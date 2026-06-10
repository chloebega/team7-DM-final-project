import argparse
import os
import warnings
from dataclasses import dataclass
from datetime import timedelta

import lightning.pytorch as pl
import optuna
import pandas as pd
import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from optuna_integration import TorchDistributedTrial
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE


warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")
optuna.logging.set_verbosity(optuna.logging.INFO)

# PyTorch Forecasting's default attention mask bias is -1e9, which overflows
# when the attention path is autocast to float16.
ATTENTION_MASK_BIAS = -1e4


@dataclass(frozen=True)
class RuntimeState:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DDP-safe Optuna sweep for the TFT model."
    )
    parser.add_argument(
        "--data-path",
        default="data/checkpoints/processed_tft_df.parquet",
        help="Processed training parquet created by the notebook.",
    )
    parser.add_argument(
        "--model-path",
        default="optuna_tuning",
        help="Directory for trial checkpoints and the Optuna SQLite study.",
    )
    parser.add_argument("--study-name", default="tft_ddp_sweep")
    parser.add_argument("--n-trials", type=int, default=15)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--n-regions", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", default="lightning_logs")
    parser.add_argument(
        "--log-name",
        default=None,
        help="TensorBoard logger name. Defaults to the study name.",
    )
    parser.add_argument(
        "--precision",
        choices=("auto", "bf16-mixed", "16-mixed", "32-true"),
        default="auto",
        help="Lightning precision. Auto uses bf16 if CUDA reports support, else fp16 on GPU.",
    )
    parser.add_argument(
        "--single-gpu",
        action="store_true",
        help="Force one-process training even if multiple GPUs are visible.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build the data module and model, then exit before training.",
    )
    return parser.parse_args()


def setup_runtime(args: argparse.Namespace) -> RuntimeState:
    os.environ.setdefault("PYTHONFAULTHANDLER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    os.environ.setdefault("NCCL_P2P_DISABLE", "0")
    os.environ.setdefault("NCCL_SHM_DISABLE", "0")

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1 and not args.single_gpu

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training was requested, but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        rank = dist.get_rank()
    else:
        rank = 0
        world_size = 1

    pl.seed_everything(args.seed, workers=True)
    return RuntimeState(
        distributed=distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def rank_zero_print(runtime: RuntimeState, *values: object) -> None:
    if runtime.rank == 0:
        print(*values, flush=True)


def select_precision(args: argparse.Namespace) -> str:
    if args.precision != "auto":
        return args.precision
    if not torch.cuda.is_available():
        return "32-true"
    if torch.cuda.is_bf16_supported():
        return "bf16-mixed"
    return "16-mixed"


def build_dataloaders(args: argparse.Namespace, runtime: RuntimeState):
    rank_zero_print(runtime, "Loading processed TFT data...")
    tft_df = pd.read_parquet(args.data_path)

    val_df = tft_df[tft_df["day_index_region"] > 5023].copy()
    train_df = tft_df[tft_df["day_index_region"] <= 5114].copy()

    unique_regions = pd.Series(train_df["region_id"].unique())
    sample_size = min(args.n_regions, len(unique_regions))
    sampled_regions = unique_regions.sample(n=sample_size, random_state=args.seed).tolist()

    tune_train_df = train_df[train_df["region_id"].isin(sampled_regions)].copy()
    tune_val_df = val_df[val_df["region_id"].isin(sampled_regions)].copy()

    rank_zero_print(
        runtime,
        f"Using {sample_size} regions, "
        f"{len(tune_train_df):,} training rows, "
        f"{len(tune_val_df):,} validation rows.",
    )

    static_categoricals = ["region_id"]
    time_varying_known_reals = [
        "day_index_region",
        "sin_dayofyear",
        "cos_dayofyear",
        "sin_month",
        "cos_month",
    ]
    time_varying_unknown_reals = [
        "prec",
        "surf_pre",
        "humidity",
        "tmp",
        "dp_tmp",
        "wb_tmp",
        "tmp_max",
        "tmp_min",
        "tmp_range",
        "surf_tmp",
        "wind",
        "wind_max",
        "wind_min",
        "wind_range",
        "vpd",
        "score",
    ]

    training_dataset = TimeSeriesDataSet(
        tune_train_df,
        time_idx="day_index_region",
        target="score",
        group_ids=static_categoricals,
        max_encoder_length=91,
        max_prediction_length=35,
        static_categoricals=static_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
    )
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        tune_val_df,
        predict=True,
        stop_randomization=True,
    )

    loader_kwargs = {
        "num_workers": args.num_workers,
        "persistent_workers": args.num_workers > 0,
    }
    train_dataloader = training_dataset.to_dataloader(
        train=True,
        batch_size=args.batch_size,
        **loader_kwargs,
    )
    val_dataloader = validation_dataset.to_dataloader(
        train=False,
        batch_size=args.batch_size * 2,
        **loader_kwargs,
    )
    return training_dataset, train_dataloader, val_dataloader


def make_trial_model(training_dataset: TimeSeriesDataSet, trial) -> TemporalFusionTransformer:
    hidden_size = trial.suggest_int("hidden_size", 32, 128, log=True)
    hidden_continuous_size = trial.suggest_int(
        "hidden_continuous_size",
        16,
        min(64, hidden_size),
        log=True,
    )
    attention_head_size = trial.suggest_int("attention_head_size", 4, 4)
    dropout = trial.suggest_float("dropout", 0.1, 0.3)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)

    return TemporalFusionTransformer.from_dataset(
        training_dataset,
        hidden_size=hidden_size,
        hidden_continuous_size=hidden_continuous_size,
        attention_head_size=attention_head_size,
        dropout=dropout,
        learning_rate=learning_rate,
        loss=MAE(),
        reduce_on_plateau_patience=2,
        log_interval=-1,
        mask_bias=ATTENTION_MASK_BIAS,
    )


def objective(
    optuna_trial,
    args: argparse.Namespace,
    runtime: RuntimeState,
    training_dataset: TimeSeriesDataSet,
    train_dataloader,
    val_dataloader,
) -> float:
    if optuna_trial is None and (not runtime.distributed or runtime.rank == 0):
        optuna_trial = optuna.trial.FixedTrial(
            {
                "gradient_clip_val": 0.1,
                "hidden_size": 64,
                "hidden_continuous_size": 32,
                "attention_head_size": 4,
                "dropout": 0.2,
                "learning_rate": 1e-3,
            }
        )

    trial = (
        TorchDistributedTrial(optuna_trial if runtime.rank == 0 else None)
        if runtime.distributed
        else optuna_trial
    )

    pl.seed_everything(args.seed + trial.number, workers=True)
    gradient_clip_val = trial.suggest_float("gradient_clip_val", 0.01, 2.0, log=True)

    model = make_trial_model(training_dataset, trial)
    if args.dry_run:
        rank_zero_print(
            runtime,
            f"Dry run model parameters: {sum(p.numel() for p in model.parameters()):,}",
        )
        return 0.0

    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(args.model_path, f"trial_{trial.number}"),
        filename="{epoch}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
    )
    callbacks = [
        LearningRateMonitor(logging_interval="epoch"),
        checkpoint_callback,
        EarlyStopping(monitor="val_loss", patience=3, min_delta=1e-4, mode="min"),
    ]

    logger = TensorBoardLogger(
        args.log_dir,
        name=args.log_name or args.study_name,
        version=trial.number,
    )
    strategy = (
        DDPStrategy(
            process_group_backend="nccl",
            timeout=timedelta(minutes=30),
        )
        if runtime.distributed
        else "auto"
    )

    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=runtime.world_size if runtime.distributed else 1,
        strategy=strategy,
        precision=select_precision(args),
        max_epochs=args.max_epochs,
        gradient_clip_val=gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        enable_progress_bar=runtime.rank == 0,
        enable_model_summary=runtime.rank == 0,
        num_sanity_val_steps=0,
    )

    trainer.fit(
        model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    val_loss = trainer.callback_metrics["val_loss"].detach().float().cpu().item()
    rank_zero_print(runtime, f"Trial {trial.number} val_loss={val_loss:.6f}")
    return val_loss


def run_sweep(
    args: argparse.Namespace,
    runtime: RuntimeState,
    training_dataset: TimeSeriesDataSet,
    train_dataloader,
    val_dataloader,
) -> optuna.Study | None:
    os.makedirs(args.model_path, exist_ok=True)

    if args.dry_run:
        objective(None, args, runtime, training_dataset, train_dataloader, val_dataloader)
        return None

    if runtime.distributed and runtime.rank != 0:
        for _ in range(args.n_trials):
            objective(None, args, runtime, training_dataset, train_dataloader, val_dataloader)
        return None

    storage = f"sqlite:///{os.path.join(args.model_path, 'tft_ddp_sweep.db')}"
    study = optuna.create_study(
        direction="minimize",
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: objective(
            trial,
            args,
            runtime,
            training_dataset,
            train_dataloader,
            val_dataloader,
        ),
        n_trials=args.n_trials,
    )
    return study


def main() -> None:
    args = parse_args()
    runtime = setup_runtime(args)
    rank_zero_print(
        runtime,
        f"Runtime: distributed={runtime.distributed}, "
        f"rank={runtime.rank}, world_size={runtime.world_size}, "
        f"precision={select_precision(args)}",
    )

    training_dataset, train_dataloader, val_dataloader = build_dataloaders(args, runtime)
    study = run_sweep(args, runtime, training_dataset, train_dataloader, val_dataloader)

    if runtime.rank == 0 and study is not None:
        print("\nTuning complete!", flush=True)
        print("Best parameters:", study.best_trial.params, flush=True)
        print("Best val_loss:", study.best_value, flush=True)

    if runtime.distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
