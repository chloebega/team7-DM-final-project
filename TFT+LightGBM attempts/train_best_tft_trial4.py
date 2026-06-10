import os
import warnings
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("PYTHONFAULTHANDLER", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
os.environ.setdefault("NCCL_P2P_DISABLE", "0")
os.environ.setdefault("NCCL_SHM_DISABLE", "0")

import lightning.pytorch as pl
import optuna
import pandas as pd
import torch
import torch.distributed as dist
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import MAE


warnings.filterwarnings("ignore")
torch.set_float32_matmul_precision("medium")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True


DATA_PATH = Path("data/checkpoints/processed_tft_df.parquet")
STUDY_DB = Path("optuna_tuning_gpu_full/tft_ddp_sweep.db")
STUDY_NAME = "tft_full_gpu_sweep"
EXPECTED_TRIAL_NUMBER = 4

CHECKPOINT_DIR = Path("tft_best_trial4_final")
LOG_DIR = "lightning_logs"
LOG_NAME = "tft_best_trial4_full"

SEED = 42
MAX_EPOCHS = 12
EARLY_STOP_PATIENCE = 4

# Per-process values. With torchrun --nproc_per_node=2, these are per 3090.
TRAIN_BATCH_SIZE = 2048
VAL_BATCH_SIZE = 1024
NUM_WORKERS = 6
PREFETCH_FACTOR = 2
PRECISION = "auto"

# PyTorch Forecasting's default attention mask bias is -1e9, which overflows
# when the attention path is autocast to float16.
ATTENTION_MASK_BIAS = -1e4

MAX_ENCODER_LENGTH = 91
MAX_PREDICTION_LENGTH = 35
TRAIN_CUTOFF = 5114
VAL_ENCODER_START = 5023


@dataclass(frozen=True)
class RuntimeState:
    distributed: bool
    rank: int
    local_rank: int
    world_size: int


def setup_runtime() -> RuntimeState:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("Distributed training was requested, but CUDA is not available.")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timedelta(minutes=45),
            )
        rank = dist.get_rank()
    else:
        rank = 0
        world_size = 1

    pl.seed_everything(SEED + EXPECTED_TRIAL_NUMBER, workers=True)
    return RuntimeState(
        distributed=distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
    )


def rank_zero_print(runtime: RuntimeState, *values: object) -> None:
    if runtime.rank == 0:
        print(*values, flush=True)


def select_precision() -> str:
    if PRECISION != "auto":
        return PRECISION
    if not torch.cuda.is_available():
        return "32-true"
    if torch.cuda.is_bf16_supported():
        return "bf16-mixed"
    return "16-mixed"


def load_trial4_params(runtime: RuntimeState) -> dict[str, float | int]:
    if not STUDY_DB.exists():
        raise FileNotFoundError(f"Missing Optuna study database: {STUDY_DB}")

    study = optuna.load_study(
        study_name=STUDY_NAME,
        storage=f"sqlite:///{STUDY_DB.resolve().as_posix()}",
    )
    best_trial = study.best_trial
    if best_trial.number != EXPECTED_TRIAL_NUMBER:
        raise RuntimeError(
            f"Expected Optuna trial {EXPECTED_TRIAL_NUMBER} to be best, "
            f"but the best completed trial is {best_trial.number} "
            f"with value {best_trial.value}."
        )

    params = dict(best_trial.params)
    fixed_params = {
        "gradient_clip_val": float(params["gradient_clip_val"]),
        "hidden_size": int(params["hidden_size"]),
        "hidden_continuous_size": int(params["hidden_continuous_size"]),
        "attention_head_size": int(params["attention_head_size"]),
        "dropout": float(params["dropout"]),
        "learning_rate": float(params["learning_rate"]),
    }
    rank_zero_print(
        runtime,
        f"Loaded Optuna trial {best_trial.number} from {STUDY_DB} "
        f"(val_loss={best_trial.value:.8f}).",
    )
    rank_zero_print(runtime, "Trial parameters:", fixed_params)
    return fixed_params


def build_dataloaders(runtime: RuntimeState):
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing processed training data: {DATA_PATH}")

    rank_zero_print(runtime, "Loading processed TFT data...")
    tft_df = pd.read_parquet(DATA_PATH)
    train_df = tft_df[tft_df["day_index_region"] <= TRAIN_CUTOFF].copy()
    val_df = tft_df[tft_df["day_index_region"] > VAL_ENCODER_START].copy()
    del tft_df

    rank_zero_print(
        runtime,
        f"Using all {train_df['region_id'].nunique():,} regions, "
        f"{len(train_df):,} training rows, {len(val_df):,} validation rows.",
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
        train_df,
        time_idx="day_index_region",
        target="score",
        group_ids=static_categoricals,
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=static_categoricals,
        time_varying_known_reals=time_varying_known_reals,
        time_varying_unknown_reals=time_varying_unknown_reals,
    )
    validation_dataset = TimeSeriesDataSet.from_dataset(
        training_dataset,
        val_df,
        predict=True,
        stop_randomization=True,
    )

    loader_kwargs = {
        "num_workers": NUM_WORKERS,
        "persistent_workers": NUM_WORKERS > 0,
        "pin_memory": torch.cuda.is_available(),
    }
    if NUM_WORKERS > 0:
        loader_kwargs["prefetch_factor"] = PREFETCH_FACTOR

    train_dataloader = training_dataset.to_dataloader(
        train=True,
        batch_size=TRAIN_BATCH_SIZE,
        **loader_kwargs,
    )
    val_dataloader = validation_dataset.to_dataloader(
        train=False,
        batch_size=VAL_BATCH_SIZE,
        **loader_kwargs,
    )

    rank_zero_print(
        runtime,
        f"Dataset samples: train={len(training_dataset):,}, "
        f"val={len(validation_dataset):,}. "
        f"Batch sizes per GPU: train={TRAIN_BATCH_SIZE}, val={VAL_BATCH_SIZE}; "
        f"workers per GPU={NUM_WORKERS}.",
    )
    return training_dataset, train_dataloader, val_dataloader


def build_model(training_dataset: TimeSeriesDataSet, params: dict[str, float | int]):
    return TemporalFusionTransformer.from_dataset(
        training_dataset,
        hidden_size=int(params["hidden_size"]),
        hidden_continuous_size=int(params["hidden_continuous_size"]),
        attention_head_size=int(params["attention_head_size"]),
        dropout=float(params["dropout"]),
        learning_rate=float(params["learning_rate"]),
        loss=MAE(),
        reduce_on_plateau_patience=2,
        log_interval=-1,
        mask_bias=ATTENTION_MASK_BIAS,
    )


def train() -> None:
    runtime = setup_runtime()
    rank_zero_print(
        runtime,
        f"Runtime: distributed={runtime.distributed}, rank={runtime.rank}, "
        f"local_rank={runtime.local_rank}, world_size={runtime.world_size}, "
        f"precision={select_precision()}",
    )

    try:
        params = load_trial4_params(runtime)
        training_dataset, train_dataloader, val_dataloader = build_dataloaders(runtime)
        model = build_model(training_dataset, params)

        checkpoint_callback = ModelCheckpoint(
            dirpath=CHECKPOINT_DIR,
            filename="trial4-full-{epoch:02d}-{val_loss:.5f}",
            monitor="val_loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        )
        callbacks = [
            LearningRateMonitor(logging_interval="epoch"),
            checkpoint_callback,
            EarlyStopping(
                monitor="val_loss",
                patience=EARLY_STOP_PATIENCE,
                min_delta=1e-4,
                mode="min",
            ),
        ]
        logger = TensorBoardLogger(LOG_DIR, name=LOG_NAME)
        strategy = (
            DDPStrategy(
                process_group_backend="nccl",
                timeout=timedelta(minutes=45),
            )
            if runtime.distributed
            else "auto"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        trainer = pl.Trainer(
            accelerator="gpu" if torch.cuda.is_available() else "cpu",
            devices=runtime.world_size if runtime.distributed else 1,
            strategy=strategy,
            precision=select_precision(),
            max_epochs=MAX_EPOCHS,
            gradient_clip_val=float(params["gradient_clip_val"]),
            accumulate_grad_batches=1,
            callbacks=callbacks,
            logger=logger,
            enable_progress_bar=runtime.rank == 0,
            enable_model_summary=runtime.rank == 0,
            num_sanity_val_steps=0,
            check_val_every_n_epoch=1,
            log_every_n_steps=50,
            deterministic=False,
        )
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_dataloader,
        )

        if runtime.rank == 0:
            best_score = checkpoint_callback.best_model_score
            best_score_text = (
                f"{best_score.detach().float().cpu().item():.8f}"
                if best_score is not None
                else "n/a"
            )
            print("\nFinal training complete!", flush=True)
            print(f"Best checkpoint: {checkpoint_callback.best_model_path}", flush=True)
            print(f"Best val_loss: {best_score_text}", flush=True)
    finally:
        if runtime.distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    train()
