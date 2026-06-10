import argparse
import os
import shutil
import warnings
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp", "wind",
    "wind_max", "wind_min", "wind_range",
]
KNOWN_REALS = ["week_sin", "week_cos"]
MAX_ENCODER_LENGTH = 13
MAX_PREDICTION_LENGTH = 5
ENCODER_ENDPOINT_OFFSETS = [6, 13, 20, 27, 34, 41, 48, 55, 62, 69, 76, 83, 90]
MONTH_LENGTHS = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
OUTPUT_COLUMNS = ["region_id", "pred_week1", "pred_week2", "pred_week3", "pred_week4", "pred_week5"]

FEATURE_COLS = [
    "prec_log",
    "prec_sum_7", "prec_sum_28", "prec_sum_91",
    "dry_days_28", "dry_days_91",
    "tmp", "tmp_mean_7", "tmp_mean_28", "tmp_mean_91", "tmp_std_28",
    "tmp_max_mean_7", "tmp_min_mean_7",
    "humidity", "humidity_mean_7", "humidity_mean_28", "humidity_mean_91",
    "humidity_min_28", "humidity_min_91",
    "vpd", "vpd_mean_7", "vpd_mean_28", "vpd_mean_91",
    "wind", "wind_mean_28", "wind_mean_91",
    "surf_pre", "surf_pre_mean_28", "surf_pre_mean_91",
    "tmp_range", "tmp_range_mean_28", "tmp_range_mean_91",
    "tmp_trend_7_28", "humidity_trend_7_28", "prec_trend_7_28",
]


@dataclass(frozen=True)
class RuntimeState:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kaggle-shaped direct weekly TFT workflow.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--train-csv", default="train.csv")
    common.add_argument("--test-csv", default="test.csv")
    common.add_argument("--sample-submission", default="sample_submission.csv")
    common.add_argument("--output-dir", default="weekly_tft_direct_outputs")
    common.add_argument("--seed", type=int, default=42)
    common.add_argument("--batch-size", type=int, default=2048)
    common.add_argument("--num-workers", type=int, default=6)
    common.add_argument("--precision", choices=("auto", "bf16-mixed", "16-mixed", "32-true"), default="auto")

    train = subparsers.add_parser("train", parents=[common])
    train.add_argument("--max-epochs", type=int, default=8)
    train.add_argument("--max-train-samples", type=int, default=500_000)
    train.add_argument("--hidden-size", type=int, default=64)
    train.add_argument("--hidden-continuous-size", type=int, default=32)
    train.add_argument("--attention-head-size", type=int, default=4)
    train.add_argument("--dropout", type=float, default=0.20)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--gradient-clip-val", type=float, default=0.5)
    train.add_argument("--early-stop-patience", type=int, default=2)

    predict = subparsers.add_parser("predict", parents=[common])
    predict.add_argument("--checkpoint", default="weekly_tft_direct_outputs/checkpoints/best.ckpt")
    predict.add_argument("--submission-name", default="weekly_tft_direct_submission.csv")

    return parser.parse_args()


def setup_runtime(seed: int) -> RuntimeState:
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "0")
    os.environ.setdefault("NCCL_SHM_DISABLE", "0")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training requested but CUDA is unavailable.")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://", timeout=timedelta(minutes=45))
        rank = dist.get_rank()
    else:
        rank = 0
        world_size = 1
    pl.seed_everything(seed, workers=True)
    return RuntimeState(distributed=distributed, rank=rank, local_rank=local_rank, world_size=world_size)


def rank_zero_print(runtime: RuntimeState, *values: object) -> None:
    if runtime.rank == 0:
        print(*values, flush=True)


def select_precision(precision: str) -> str:
    if precision != "auto":
        return precision
    if not torch.cuda.is_available():
        return "32-true"
    if torch.cuda.is_bf16_supported():
        return "bf16-mixed"
    return "16-mixed"


def split_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["date"].astype(str).str.extract(r"^(?P<year>\d+)-(?P<month>\d{2})-(?P<day>\d{2})$")
    if parts.isna().any().any():
        bad = df.loc[parts.isna().any(axis=1), "date"].head().tolist()
        raise ValueError(f"Unexpected date examples: {bad}")
    df = df.copy()
    df["year"] = parts["year"].astype(int)
    df["month"] = parts["month"].astype(int)
    df["day"] = parts["day"].astype(int)
    return df


def day_of_year(month: pd.Series, day: pd.Series) -> pd.Series:
    offsets = {m: sum(MONTH_LENGTHS[i] for i in range(1, m)) for m in range(1, 13)}
    return month.map(offsets).astype(int) + day.astype(int)


def advance_one_day(year: int, month: int, day: int) -> tuple[int, int, int]:
    day += 1
    if day > MONTH_LENGTHS[month]:
        day = 1
        month += 1
        if month > 12:
            month = 1
            year += 1
    return year, month, day


def advance_days(year: int, month: int, day: int, n_days: int) -> tuple[int, int, int]:
    for _ in range(n_days):
        year, month, day = advance_one_day(year, month, day)
    return year, month, day


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    doy = day_of_year(df["month"], df["day"])
    df["week_sin"] = np.sin(2 * np.pi * doy / 366.0)
    df["week_cos"] = np.cos(2 * np.pi * doy / 366.0)
    return df


def load_daily_csv(path: str, require_score: bool) -> pd.DataFrame:
    usecols = ["region_id", "date", *WEATHER_COLS]
    if require_score:
        usecols.append("score")
    df = pd.read_csv(path, usecols=usecols)
    if not require_score:
        df["score"] = np.nan
    df = split_date_parts(df)
    df = df.sort_values(["region_id", "year", "month", "day"]).reset_index(drop=True)
    df["day_index_region"] = df.groupby("region_id", sort=False).cumcount()
    return df


def rolling_by_region(df: pd.DataFrame, source: str, window: int, stat: str) -> pd.Series:
    grouped = df.groupby("region_id", sort=False)[source]
    if stat == "mean":
        return grouped.transform(lambda x: x.rolling(window, min_periods=1).mean())
    if stat == "sum":
        return grouped.transform(lambda x: x.rolling(window, min_periods=1).sum())
    if stat == "min":
        return grouped.transform(lambda x: x.rolling(window, min_periods=1).min())
    if stat == "std":
        return grouped.transform(lambda x: x.rolling(window, min_periods=2).std()).fillna(0.0)
    raise ValueError(stat)


def add_trailing_weather_features(daily: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["prec_raw"] = daily["prec"].clip(lower=0)
    daily["prec_log"] = np.log1p(daily["prec_raw"])
    daily["is_dry_day"] = (daily["prec_raw"] < 0.1).astype(float)

    e_s = 0.611 * np.exp((17.27 * daily["tmp"]) / (daily["tmp"] + 237.3))
    e_a = 0.611 * np.exp((17.27 * daily["dp_tmp"]) / (daily["dp_tmp"] + 237.3))
    daily["vpd"] = e_s - e_a

    for window in [7, 28, 91]:
        daily[f"prec_sum_{window}"] = rolling_by_region(daily, "prec_raw", window, "sum")
        daily[f"tmp_mean_{window}"] = rolling_by_region(daily, "tmp", window, "mean")
        daily[f"humidity_mean_{window}"] = rolling_by_region(daily, "humidity", window, "mean")
        daily[f"vpd_mean_{window}"] = rolling_by_region(daily, "vpd", window, "mean")

    for window in [28, 91]:
        daily[f"dry_days_{window}"] = rolling_by_region(daily, "is_dry_day", window, "sum")
        daily[f"humidity_min_{window}"] = rolling_by_region(daily, "humidity", window, "min")
        daily[f"wind_mean_{window}"] = rolling_by_region(daily, "wind", window, "mean")
        daily[f"surf_pre_mean_{window}"] = rolling_by_region(daily, "surf_pre", window, "mean")
        daily[f"tmp_range_mean_{window}"] = rolling_by_region(daily, "tmp_range", window, "mean")

    daily["tmp_std_28"] = rolling_by_region(daily, "tmp", 28, "std")
    daily["tmp_max_mean_7"] = rolling_by_region(daily, "tmp_max", 7, "mean")
    daily["tmp_min_mean_7"] = rolling_by_region(daily, "tmp_min", 7, "mean")
    daily["tmp_trend_7_28"] = daily["tmp_mean_7"] - daily["tmp_mean_28"]
    daily["humidity_trend_7_28"] = daily["humidity_mean_7"] - daily["humidity_mean_28"]
    daily["prec_trend_7_28"] = daily["prec_sum_7"] - (daily["prec_sum_28"] / 4.0)
    daily = add_calendar_features(daily)
    return daily


def fill_weekly_features(df: pd.DataFrame, medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    fill_cols = FEATURE_COLS + KNOWN_REALS
    df[fill_cols] = df.groupby("region_id", sort=False)[fill_cols].transform(lambda x: x.ffill().bfill())
    if medians is None:
        medians = df[FEATURE_COLS].median(numeric_only=True).fillna(0.0)
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(medians)
    df[KNOWN_REALS] = df[KNOWN_REALS].fillna(0.0)
    return df, medians


def make_train_weekly(train_csv: str) -> pd.DataFrame:
    daily = load_daily_csv(train_csv, require_score=True)
    daily = add_trailing_weather_features(daily)
    weekly = daily[daily["score"].notna()].copy()
    weekly["week_idx"] = weekly.groupby("region_id", sort=False).cumcount()
    keep_cols = ["region_id", "week_idx", "score", *KNOWN_REALS, *FEATURE_COLS]
    weekly = weekly[keep_cols].reset_index(drop=True)
    weekly, _ = fill_weekly_features(weekly)
    weekly["score"] = weekly["score"].astype(float).clip(0, 5)
    return weekly


def split_train_val(weekly: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    weekly = weekly.copy()
    max_week = weekly.groupby("region_id", sort=False)["week_idx"].transform("max")
    val_start = max_week - (MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH) + 1
    train_cutoff = max_week - MAX_PREDICTION_LENGTH
    train_df = weekly[weekly["week_idx"] <= train_cutoff].copy()
    val_df = weekly[weekly["week_idx"] >= val_start].copy()
    return train_df, val_df


def build_training_dataset(train_df: pd.DataFrame, max_train_samples: int = 0, seed: int = 42) -> TimeSeriesDataSet:
    dataset = TimeSeriesDataSet(
        train_df,
        time_idx="week_idx",
        target="score",
        group_ids=["region_id"],
        min_encoder_length=MAX_ENCODER_LENGTH,
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_prediction_length=MAX_PREDICTION_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=["region_id"],
        time_varying_known_reals=KNOWN_REALS,
        time_varying_unknown_reals=FEATURE_COLS,
        target_normalizer=None,
        add_relative_time_idx=True,
        add_encoder_length=True,
    )
    if max_train_samples and len(dataset) > max_train_samples:
        dataset.index = dataset.index.sample(n=max_train_samples, random_state=seed).reset_index(drop=True)
    return dataset


def build_model(training_dataset: TimeSeriesDataSet, args: argparse.Namespace) -> TemporalFusionTransformer:
    return TemporalFusionTransformer.from_dataset(
        training_dataset,
        hidden_size=args.hidden_size,
        hidden_continuous_size=args.hidden_continuous_size,
        attention_head_size=args.attention_head_size,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        loss=MAE(),
        output_size=1,
        reduce_on_plateau_patience=2,
        log_interval=-1,
    )


def make_loader(dataset: TimeSeriesDataSet, train: bool, batch_size: int, num_workers: int):
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "persistent_workers": num_workers > 0,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return dataset.to_dataloader(train=train, **kwargs)


def train_mode(args: argparse.Namespace) -> None:
    runtime = setup_runtime(args.seed)
    out_dir = Path(args.output_dir)
    ckpt_dir = out_dir / "checkpoints"
    log_dir = out_dir / "lightning_logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    try:
        rank_zero_print(runtime, "Building weekly direct TFT dataset...")
        weekly = make_train_weekly(args.train_csv)
        train_df, val_df = split_train_val(weekly)
        del weekly
        rank_zero_print(runtime, f"Training rows: {len(train_df):,}; validation rows: {len(val_df):,}; regions: {train_df['region_id'].nunique():,}")

        training_dataset = build_training_dataset(train_df, args.max_train_samples, args.seed)
        validation_dataset = TimeSeriesDataSet.from_dataset(training_dataset, val_df, predict=True, stop_randomization=True)
        rank_zero_print(runtime, f"Training samples after index subsampling: {len(training_dataset):,}; validation samples: {len(validation_dataset):,}")

        train_loader = make_loader(training_dataset, train=True, batch_size=args.batch_size, num_workers=args.num_workers)
        val_loader = make_loader(validation_dataset, train=False, batch_size=args.batch_size * 2, num_workers=args.num_workers)
        model = build_model(training_dataset, args)
        if runtime.rank == 0:
            print("Encoder variables:", model.encoder_variables, flush=True)
            print("Decoder variables:", model.decoder_variables, flush=True)

        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="weekly-direct-{epoch:02d}-{val_loss:.5f}",
            monitor="val_loss",
            mode="min",
            save_top_k=2,
            save_last=True,
        )
        callbacks = [
            LearningRateMonitor(logging_interval="epoch"),
            checkpoint_callback,
            EarlyStopping(monitor="val_loss", patience=args.early_stop_patience, min_delta=1e-4, mode="min"),
        ]
        strategy = DDPStrategy(process_group_backend="nccl", timeout=timedelta(minutes=45)) if runtime.distributed else "auto"
        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=runtime.world_size if runtime.distributed else 1,
            strategy=strategy,
            precision=select_precision(args.precision),
            max_epochs=args.max_epochs,
            gradient_clip_val=args.gradient_clip_val,
            callbacks=callbacks,
            logger=TensorBoardLogger(str(log_dir), name="weekly_direct_tft"),
            enable_progress_bar=runtime.rank == 0,
            enable_model_summary=runtime.rank == 0,
            num_sanity_val_steps=0,
            check_val_every_n_epoch=1,
            log_every_n_steps=50,
        )
        trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

        if runtime.rank == 0:
            best_path = Path(checkpoint_callback.best_model_path)
            best_copy = ckpt_dir / "best.ckpt"
            if best_path.exists():
                shutil.copy2(best_path, best_copy)
            print("Training complete.", flush=True)
            print(f"Best checkpoint: {best_path}", flush=True)
            print(f"Best checkpoint copy: {best_copy}", flush=True)
            print(f"Best val_loss: {checkpoint_callback.best_model_score}", flush=True)
    finally:
        if runtime.distributed and dist.is_initialized():
            dist.destroy_process_group()


def build_test_weekly(test_csv: str, train_weekly: pd.DataFrame) -> pd.DataFrame:
    daily = load_daily_csv(test_csv, require_score=False)
    daily = add_trailing_weather_features(daily)
    train_max_week = train_weekly.groupby("region_id", sort=False)["week_idx"].max()
    feature_medians = train_weekly[FEATURE_COLS].median(numeric_only=True).fillna(0.0)

    rows = []
    for region, group in daily.groupby("region_id", sort=False):
        group = group.sort_values("day_index_region").reset_index(drop=True)
        if len(group) < 91:
            raise ValueError(f"Region {region} has only {len(group)} test rows; expected 91.")
        start_week = int(train_max_week.loc[region]) + 1
        encoder = group.iloc[ENCODER_ENDPOINT_OFFSETS].copy()
        for j, (_, row) in enumerate(encoder.iterrows()):
            out = {"region_id": region, "week_idx": start_week + j, "score": 0.0}
            for col in KNOWN_REALS + FEATURE_COLS:
                out[col] = row[col]
            rows.append(out)

        last = group.iloc[90]
        year, month, day = int(last["year"]), int(last["month"]), int(last["day"])
        for h in range(1, MAX_PREDICTION_LENGTH + 1):
            fy, fm, fd = advance_days(year, month, day, h * 7)
            doy = sum(MONTH_LENGTHS[i] for i in range(1, fm)) + fd
            out = {
                "region_id": region,
                "week_idx": start_week + MAX_ENCODER_LENGTH + h - 1,
                "score": 0.0,
                "week_sin": float(np.sin(2 * np.pi * doy / 366.0)),
                "week_cos": float(np.cos(2 * np.pi * doy / 366.0)),
            }
            for col in FEATURE_COLS:
                out[col] = float(feature_medians[col])
            rows.append(out)

    pred_df = pd.DataFrame(rows)
    pred_df, _ = fill_weekly_features(pred_df, medians=feature_medians)
    pred_df["score"] = 0.0
    pred_df = pred_df[["region_id", "week_idx", "score", *KNOWN_REALS, *FEATURE_COLS]].reset_index(drop=True)
    if pred_df.isna().sum().sum() != 0:
        missing = pred_df.isna().sum()
        raise ValueError(f"Prediction dataframe has missing values:\n{missing[missing > 0]}")
    return pred_df


def prediction_output_to_array(prediction) -> np.ndarray:
    output = getattr(prediction, "output", prediction)
    if isinstance(output, list):
        output = torch.cat([x.detach().cpu() for x in output], dim=0)
    elif isinstance(output, torch.Tensor):
        output = output.detach().cpu()
    else:
        output = torch.as_tensor(output)
    if output.ndim == 3 and output.shape[-1] == 1:
        output = output[..., 0]
    if output.ndim != 2 or output.shape[1] != MAX_PREDICTION_LENGTH:
        raise ValueError(f"Unexpected prediction shape: {tuple(output.shape)}")
    return output.numpy()


def extract_regions(prediction, prediction_dataset: TimeSeriesDataSet, pred_df: pd.DataFrame) -> list[str]:
    index = getattr(prediction, "index", None)
    if isinstance(index, pd.DataFrame) and "region_id" in index.columns:
        return index["region_id"].astype(str).tolist()
    try:
        decoded = prediction_dataset.decoded_index
        if isinstance(decoded, pd.DataFrame) and "region_id" in decoded.columns:
            return decoded["region_id"].astype(str).tolist()
    except Exception:
        pass
    return pred_df.groupby("region_id", sort=False).size().index.astype(str).tolist()


def predict_mode(args: argparse.Namespace) -> None:
    pl.seed_everything(args.seed, workers=True)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = Path(args.checkpoint)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")

    print("Rebuilding training dataset metadata...", flush=True)
    train_weekly = make_train_weekly(args.train_csv)
    train_df, _ = split_train_val(train_weekly)
    training_dataset = build_training_dataset(train_df, max_train_samples=0, seed=args.seed)
    pred_df = build_test_weekly(args.test_csv, train_weekly)
    prediction_dataset = TimeSeriesDataSet.from_dataset(training_dataset, pred_df, predict=True, stop_randomization=True)
    print(f"Prediction samples: {len(prediction_dataset):,}", flush=True)

    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint)
    model.eval()
    prediction = model.predict(
        prediction_dataset,
        mode="prediction",
        return_index=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        trainer_kwargs={
            "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
            "devices": 1,
            "precision": "32-true",
            "logger": False,
            "enable_progress_bar": True,
        },
    )
    pred = np.clip(prediction_output_to_array(prediction), 0.0, 5.0)
    regions = extract_regions(prediction, prediction_dataset, pred_df)
    if len(regions) != pred.shape[0]:
        raise ValueError(f"Region count mismatch: {len(regions)} vs {pred.shape[0]}")

    wide = pd.DataFrame(pred, columns=OUTPUT_COLUMNS[1:])
    wide.insert(0, "region_id", regions)
    sample = pd.read_csv(args.sample_submission)
    submission = sample[["region_id"]].merge(wide, on="region_id", how="left", validate="one_to_one")
    submission = submission[OUTPUT_COLUMNS]
    if submission.isna().sum().sum() != 0:
        missing = submission.isna().sum()
        raise ValueError(f"Submission has missing values:\n{missing[missing > 0]}")
    if not submission["region_id"].equals(sample["region_id"]):
        raise ValueError("Submission region order does not match sample_submission.csv")

    pred_path = out_dir / "weekly_tft_direct_predictions_wide.csv"
    sub_path = out_dir / args.submission_name
    wide.to_csv(pred_path, index=False)
    submission.to_csv(sub_path, index=False)
    print(f"Saved predictions: {pred_path}", flush=True)
    print(f"Saved submission: {sub_path}", flush=True)
    print(submission.head(15).to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train_mode(args)
    elif args.mode == "predict":
        predict_mode(args)
    else:
        raise ValueError(args.mode)


if __name__ == "__main__":
    main()
