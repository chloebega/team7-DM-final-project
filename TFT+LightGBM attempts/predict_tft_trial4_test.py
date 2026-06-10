import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

import lightning.pytorch as pl
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet

warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")

TRAIN_CSV = Path("train.csv")
TEST_CSV = Path("test.csv")
PROCESSED_TRAIN_PATH = Path("data/checkpoints/processed_tft_df.parquet")
DEFAULT_CHECKPOINT = Path("tft_best_trial4_final/trial4-full-epoch=03-val_loss=0.04502.ckpt")
OUTPUT_DIR = Path("tft_trial4_inference_outputs")

MAX_ENCODER_LENGTH = 91
MAX_PREDICTION_LENGTH = 35
TRAIN_CUTOFF = 5114
WEEKLY_OFFSETS = [6, 13, 20, 27, 34]

WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp", "wind",
    "wind_max", "wind_min", "wind_range",
]
CONTINUOUS_FEATURES = [
    "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp",
    "wind", "wind_max", "wind_min", "wind_range",
    "vpd", "sin_month", "cos_month", "prec",
]
FINAL_COLS = [
    "region_id", "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp", "wind", "wind_max",
    "wind_min", "wind_range", "score", "day_index_region", "sin_dayofyear",
    "cos_dayofyear", "vpd", "sin_month", "cos_month",
]
KNOWN_REALS = ["day_index_region", "sin_dayofyear", "cos_dayofyear", "sin_month", "cos_month"]
UNKNOWN_REALS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp", "wind",
    "wind_max", "wind_min", "wind_range", "vpd", "score",
]
MONTH_LENGTHS = {1: 31, 2: 29, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TFT trial 4 inference on test.csv.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--future-weather-window", type=int, default=28)
    parser.add_argument("--limit-regions", type=int, default=0, help="Smoke-test on the first N test regions if > 0.")
    return parser.parse_args()


def split_date_parts(df: pd.DataFrame) -> pd.DataFrame:
    parts = df["date"].astype(str).str.extract(r"^(?P<year>\d+)-(?P<month>\d{2})-(?P<day>\d{2})$")
    if parts.isna().any().any():
        bad = df.loc[parts.isna().any(axis=1), "date"].head().tolist()
        raise ValueError(f"Unexpected date format examples: {bad}")
    df = df.copy()
    df["year"] = parts["year"].astype(int)
    df["month"] = parts["month"].astype(int)
    df["day"] = parts["day"].astype(int)
    return df


def day_of_year_from_month_day(month: pd.Series, day: pd.Series) -> pd.Series:
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


def add_calendar_and_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    df = split_date_parts(df)
    day_of_year = day_of_year_from_month_day(df["month"], df["day"])
    df["sin_dayofyear"] = np.sin(2 * np.pi * day_of_year / 365.25)
    df["cos_dayofyear"] = np.cos(2 * np.pi * day_of_year / 365.25)

    e_s = 0.611 * np.exp((17.27 * df["tmp"]) / (df["tmp"] + 237.3))
    e_a = 0.611 * np.exp((17.27 * df["dp_tmp"]) / (df["dp_tmp"] + 237.3))
    df["vpd"] = e_s - e_a
    df["prec"] = np.log1p(df["prec"])

    df["sin_month"] = np.sin((2 * np.pi * df["month"]) / 12)
    df["cos_month"] = np.cos((2 * np.pi * df["month"]) / 12)
    return df


def load_raw_train() -> pd.DataFrame:
    usecols = ["region_id", "date", *WEATHER_COLS, "score"]
    train = pd.read_csv(TRAIN_CSV, usecols=usecols)
    train = split_date_parts(train)
    train = train.sort_values(["region_id", "year", "month", "day"]).reset_index(drop=True)
    train["day_index_region"] = train.groupby("region_id").cumcount()
    return train


def build_extended_test(train_raw: pd.DataFrame, limit_regions: int, future_weather_window: int) -> pd.DataFrame:
    test = pd.read_csv(TEST_CSV)
    test = split_date_parts(test)
    test = test.sort_values(["region_id", "year", "month", "day"]).reset_index(drop=True)

    if limit_regions > 0:
        keep_regions = test["region_id"].drop_duplicates().head(limit_regions).tolist()
        test = test[test["region_id"].isin(keep_regions)].copy()
        print(f"Smoke-test mode: keeping {len(keep_regions)} regions.", flush=True)

    train_counts = train_raw.groupby("region_id").size()
    test["observed_offset"] = test.groupby("region_id").cumcount()
    test["day_index_region"] = test["region_id"].map(train_counts).astype(int) + test["observed_offset"].astype(int)
    test["is_future"] = 0
    test["horizon_day"] = 0

    future_rows = []
    for region, group in test.groupby("region_id", sort=False):
        group = group.sort_values("day_index_region")
        last = group.iloc[-1]
        weather_tail = group[WEATHER_COLS].tail(future_weather_window)
        future_weather = weather_tail.mean(skipna=True)
        future_weather = future_weather.fillna(group[WEATHER_COLS].ffill().iloc[-1]).fillna(0.0)

        year, month, day = int(last["year"]), int(last["month"]), int(last["day"])
        for horizon_day in range(1, MAX_PREDICTION_LENGTH + 1):
            year, month, day = advance_one_day(year, month, day)
            row = {
                "region_id": region,
                "date": f"{year:04d}-{month:02d}-{day:02d}",
                "day_index_region": int(last["day_index_region"]) + horizon_day,
                "is_future": 1,
                "horizon_day": horizon_day,
            }
            for col in WEATHER_COLS:
                row[col] = float(future_weather[col])
            future_rows.append(row)

    future = pd.DataFrame(future_rows)
    extended = pd.concat([test.drop(columns=["observed_offset"]), future], ignore_index=True, sort=False)
    extended = split_date_parts(extended)
    extended = extended.sort_values(["region_id", "day_index_region"]).reset_index(drop=True)
    return extended


def attach_score_proxy(extended: pd.DataFrame, train_raw: pd.DataFrame) -> pd.DataFrame:
    # The trained model requires a non-null score column, and score is also part of decoder_cont.
    # We use the most recent interpolated training target path as a proxy for the unavailable test score history.
    train_score = train_raw[["region_id", "day_index_region", "score"]].copy()
    global_fill = float(train_score["score"].dropna().median())
    if not np.isfinite(global_fill):
        global_fill = 0.0
    train_score["score_proxy"] = train_score.groupby("region_id")["score"].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    ).fillna(global_fill)

    proxy_groups = {region: group.sort_values("day_index_region")["score_proxy"].to_numpy() for region, group in train_score.groupby("region_id")}
    extended = extended.copy()
    extended["score"] = np.nan

    for region, indices in extended.groupby("region_id").groups.items():
        group = extended.loc[indices].sort_values("day_index_region")
        observed_mask = group["is_future"].to_numpy() == 0
        n_observed = int(observed_mask.sum())
        history = proxy_groups.get(region, np.array([global_fill], dtype=float))
        observed_proxy = history[-n_observed:] if len(history) >= n_observed else np.pad(history, (n_observed - len(history), 0), constant_values=global_fill)
        if len(observed_proxy) == 0:
            observed_proxy = np.array([global_fill], dtype=float)
        filled = np.empty(len(group), dtype=float)
        filled[observed_mask] = observed_proxy[-n_observed:]
        filled[~observed_mask] = float(observed_proxy[-1])
        extended.loc[group.index, "score"] = filled

    extended["score"] = extended["score"].clip(0, 5)
    if extended["score"].isna().any():
        raise ValueError("Score proxy still contains NaNs after filling.")
    return extended


def apply_train_region_scaling(extended: pd.DataFrame, train_raw: pd.DataFrame) -> pd.DataFrame:
    train_features = add_calendar_and_weather_features(train_raw.copy())
    extended = add_calendar_and_weather_features(extended.copy())

    train_groups = train_features.groupby("region_id").groups
    missing_regions = []
    for region, test_indices in extended.groupby("region_id").groups.items():
        train_indices = train_groups.get(region)
        if train_indices is None:
            missing_regions.append(region)
            continue
        scaler = StandardScaler()
        train_subset = train_features.loc[train_indices, CONTINUOUS_FEATURES].ffill().bfill().fillna(0.0)
        test_subset = extended.loc[test_indices, CONTINUOUS_FEATURES].ffill().bfill().fillna(0.0)
        scaler.fit(train_subset)
        extended.loc[test_indices, CONTINUOUS_FEATURES] = scaler.transform(test_subset)

    if missing_regions:
        raise ValueError(f"Regions in test but missing from train: {missing_regions[:10]}")

    inference_df = extended[FINAL_COLS].copy()
    inference_df["region_id"] = inference_df["region_id"].astype(str)
    if inference_df.isna().sum().sum() != 0:
        missing = inference_df.isna().sum()
        missing = missing[missing > 0]
        raise ValueError(f"Inference dataframe still has missing values:\n{missing}")
    return inference_df


def build_training_dataset() -> TimeSeriesDataSet:
    print("Loading processed training parquet to rebuild the training dataset metadata...", flush=True)
    tft_df = pd.read_parquet(PROCESSED_TRAIN_PATH)
    train_df = tft_df[tft_df["day_index_region"] <= TRAIN_CUTOFF].copy()
    del tft_df

    return TimeSeriesDataSet(
        train_df,
        time_idx="day_index_region",
        target="score",
        group_ids=["region_id"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=["region_id"],
        time_varying_known_reals=KNOWN_REALS,
        time_varying_unknown_reals=UNKNOWN_REALS,
    )


def prediction_output_to_tensor(prediction) -> torch.Tensor:
    output = getattr(prediction, "output", prediction)
    if isinstance(output, list):
        output = torch.cat([x.detach().cpu() for x in output], dim=0)
    elif isinstance(output, torch.Tensor):
        output = output.detach().cpu()
    else:
        output = torch.as_tensor(output)
    if output.ndim == 3 and output.shape[-1] == 1:
        output = output[..., 0]
    if output.ndim != 2 or output.shape[1] < MAX_PREDICTION_LENGTH:
        raise ValueError(f"Unexpected prediction tensor shape: {tuple(output.shape)}")
    return output[:, :MAX_PREDICTION_LENGTH]


def extract_prediction_regions(prediction, prediction_dataset: TimeSeriesDataSet, fallback_df: pd.DataFrame) -> list[str]:
    index = getattr(prediction, "index", None)
    if isinstance(index, pd.DataFrame) and "region_id" in index.columns:
        return index["region_id"].astype(str).tolist()
    try:
        decoded = prediction_dataset.decoded_index
        if isinstance(decoded, pd.DataFrame) and "region_id" in decoded.columns:
            return decoded["region_id"].astype(str).tolist()
    except Exception:
        pass
    return fallback_df.groupby("region_id", sort=False).size().index.astype(str).tolist()


def save_outputs(prediction, prediction_dataset: TimeSeriesDataSet, inference_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pred = prediction_output_to_tensor(prediction).numpy()
    regions = extract_prediction_regions(prediction, prediction_dataset, inference_df)
    if len(regions) != pred.shape[0]:
        raise ValueError(f"Region count {len(regions)} does not match prediction rows {pred.shape[0]}.")

    daily_cols = [f"day_{i:02d}" for i in range(1, MAX_PREDICTION_LENGTH + 1)]
    daily = pd.DataFrame(pred, columns=daily_cols)
    daily.insert(0, "region_id", regions)
    daily.to_csv(output_dir / "tft_trial4_daily_raw_predictions.csv", index=False)

    weekly = np.clip(pred[:, WEEKLY_OFFSETS], 0.0, 5.0)
    weekly_wide = pd.DataFrame(weekly, columns=[f"week_{i}" for i in range(1, 6)])
    weekly_wide.insert(0, "region_id", regions)
    weekly_wide.to_csv(output_dir / "tft_trial4_weekly_clipped_predictions_wide.csv", index=False)

    long_rows = []
    for row_region, row_values in zip(regions, weekly):
        for week_idx, value in enumerate(row_values, start=1):
            long_rows.append({
                "region_id": row_region,
                "horizon_week": week_idx,
                "day_offset": week_idx * 7,
                "prediction": float(value),
            })
    weekly_long = pd.DataFrame(long_rows)
    weekly_long.to_csv(output_dir / "tft_trial4_weekly_clipped_predictions_long.csv", index=False)

    print("Saved prediction files:", flush=True)
    print(f"  {output_dir / 'tft_trial4_daily_raw_predictions.csv'}", flush=True)
    print(f"  {output_dir / 'tft_trial4_weekly_clipped_predictions_wide.csv'}", flush=True)
    print(f"  {output_dir / 'tft_trial4_weekly_clipped_predictions_long.csv'}", flush=True)
    print("Weekly preview:", flush=True)
    print(weekly_wide.head().to_string(index=False), flush=True)


def main() -> None:
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    output_dir = Path(args.output_dir)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    print(f"Using checkpoint: {checkpoint.resolve()}", flush=True)
    print("Loading raw train and test data...", flush=True)
    train_raw = load_raw_train()
    extended_test = build_extended_test(train_raw, args.limit_regions, args.future_weather_window)
    extended_test = attach_score_proxy(extended_test, train_raw)
    inference_df = apply_train_region_scaling(extended_test, train_raw)
    del train_raw, extended_test

    print(f"Inference dataframe shape: {inference_df.shape}", flush=True)
    print(f"Regions: {inference_df['region_id'].nunique()}, rows per region should be {MAX_ENCODER_LENGTH + MAX_PREDICTION_LENGTH}.", flush=True)
    print(inference_df.groupby("region_id").size().describe().to_string(), flush=True)

    training_dataset = build_training_dataset()
    prediction_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        inference_df,
        predict=True,
        stop_randomization=True,
    )
    print(f"Prediction dataset samples: {len(prediction_dataset)}", flush=True)

    model = TemporalFusionTransformer.load_from_checkpoint(checkpoint)
    model.eval()

    trainer_kwargs = {
        "accelerator": "gpu" if torch.cuda.is_available() else "cpu",
        "devices": 1,
        "logger": False,
        "enable_progress_bar": True,
        "precision": "32-true",
    }
    print("Running model.predict...", flush=True)
    prediction = model.predict(
        prediction_dataset,
        mode="prediction",
        return_index=True,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        trainer_kwargs=trainer_kwargs,
    )
    save_outputs(prediction, prediction_dataset, inference_df, output_dir)
    print("Inference complete.", flush=True)


if __name__ == "__main__":
    main()
