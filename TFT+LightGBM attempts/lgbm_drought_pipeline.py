from __future__ import annotations

import argparse
import json
import math
import os
import resource
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning
from sklearn.metrics import mean_absolute_error

try:
    import lightgbm as lgb
except ImportError as exc:
    raise SystemExit(
        "LightGBM is required. Run `%pip install -q lightgbm` in the notebook first."
    ) from exc


WEATHER_COLS = [
    "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp",
    "tmp_max", "tmp_min", "tmp_range", "surf_tmp", "wind",
    "wind_max", "wind_min", "wind_range",
]
WINDOWS = [7, 14, 28, 56, 91]
TEST_WEEKLY_OFFSETS = [6, 13, 20, 27, 34, 41, 48, 55, 62, 69, 76, 83, 90]
OLDER_TRUE_LAGS = [13, 26, 52, 104]
MONTH_OFFSETS = np.array([0, 0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335], dtype=np.int16)
RANDOM_SEED = 42
warnings.filterwarnings("ignore", category=PerformanceWarning)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def memory_gb() -> float:
    usage_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return usage_kb / 1024 / 1024


@contextmanager
def timed_step(name: str):
    start = time.perf_counter()
    log(f"START {name}")
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        log(f"DONE  {name} in {elapsed:,.1f}s; peak RSS {memory_gb():.2f} GB")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Leakage-aware LightGBM drought forecasting pipeline")
    parser.add_argument("--mode", choices=["smoke", "full"], default="full")
    parser.add_argument("--train-path", default="train.csv")
    parser.add_argument("--test-path", default="test.csv")
    parser.add_argument("--sample-path", default="sample_submission.csv")
    parser.add_argument("--out-dir", default="lgbm_outputs")
    parser.add_argument("--max-regions", type=int, default=None, help="Limit to the first N sample regions for smoke testing")
    parser.add_argument("--num-boost-round", type=int, default=1000)
    parser.add_argument("--early-stopping-rounds", type=int, default=150)
    parser.add_argument("--nowcast-rounds", type=int, default=700)
    parser.add_argument("--min-forecast-week", type=int, default=100)
    parser.add_argument("--valid-origin-offsets", default="25,20,15,10,5")
    parser.add_argument("--no-sample-weights", action="store_true")
    parser.add_argument(
        "--sort-input",
        action="store_true",
        help="Force a full region/date sort. Slow; only use if the CSV row order is not already region-local chronological.",
    )
    return parser.parse_args()


def compact_float_frame(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        if col not in {"year"}:
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


def read_csv_maybe_filtered(path: str, *, has_score: bool, region_filter: set[str] | None) -> pd.DataFrame:
    usecols = ["region_id", "date"] + WEATHER_COLS + (["score"] if has_score else [])
    dtype = {"region_id": "string", "date": "string"}
    dtype.update({col: "float32" for col in WEATHER_COLS})
    if has_score:
        dtype["score"] = "float32"

    if region_filter is None:
        return pd.read_csv(path, usecols=usecols, dtype=dtype)

    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(path, usecols=usecols, dtype=dtype, chunksize=500_000):
        keep = chunk["region_id"].isin(region_filter)
        if keep.any():
            chunks.append(chunk.loc[keep].copy())
    if not chunks:
        raise ValueError(f"No rows from {path} matched the selected regions")
    return pd.concat(chunks, ignore_index=True)


def parse_date_parts_fast(df: pd.DataFrame) -> pd.DataFrame:
    date_str = df["date"].astype("string")
    # Synthetic years can have more than four digits. Month/day are fixed at the right side.
    df["year"] = date_str.str[:-6].astype("int32")
    df["month"] = date_str.str[-5:-3].astype("int16")
    df["day"] = date_str.str[-2:].astype("int16")
    month_values = df["month"].to_numpy(dtype=np.int16)
    df["doy"] = (MONTH_OFFSETS[month_values] + df["day"].to_numpy(dtype=np.int16)).astype("int16")
    df["week_of_year"] = (((df["doy"].astype("int16") - 1) // 7) + 1).astype("int16")
    df["month_sin"] = np.sin(2 * np.pi * df["month"].astype("float32") / 12.0).astype("float32")
    df["month_cos"] = np.cos(2 * np.pi * df["month"].astype("float32") / 12.0).astype("float32")
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"].astype("float32") / 366.0).astype("float32")
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"].astype("float32") / 366.0).astype("float32")
    df["season_bin"] = ((df["month"].astype("int16") % 12) // 3).astype("int8")
    return df


def prepare_daily_order(df: pd.DataFrame, region_order: list[str], *, label: str, sort_input: bool) -> pd.DataFrame:
    log(f"{label}: assigning region codes")
    region_cat = pd.Categorical(df["region_id"], categories=region_order, ordered=True)
    if (region_cat.codes < 0).any():
        missing = pd.Series(df.loc[region_cat.codes < 0, "region_id"]).astype(str).unique()[:5].tolist()
        raise ValueError(f"{label}: region ids not found in sample submission order: {missing}")
    df["region_id"] = region_cat
    df["region_idx"] = region_cat.codes.astype("int32")

    log(f"{label}: parsing synthetic date parts with fixed-width string slicing")
    df = parse_date_parts_fast(df)

    date_key = (
        df["year"].to_numpy(dtype=np.int64) * 10_000
        + df["month"].to_numpy(dtype=np.int64) * 100
        + df["day"].to_numpy(dtype=np.int64)
    )
    codes = df["region_idx"].to_numpy(dtype=np.int32)
    same_region = codes[1:] == codes[:-1]
    block_starts = np.r_[0, np.flatnonzero(~same_region) + 1]
    block_codes = codes[block_starts]
    regions_are_contiguous = len(block_codes) == len(np.unique(block_codes))
    dates_are_ordered_within_region = bool(np.all(date_key[1:][same_region] >= date_key[:-1][same_region]))

    if sort_input or not (regions_are_contiguous and dates_are_ordered_within_region):
        reason = "--sort-input was provided" if sort_input else "CSV order failed the region/date sanity check"
        log(f"{label}: sorting by region/date because {reason}")
        df = df.sort_values(["region_id", "year", "month", "day"], kind="mergesort").reset_index(drop=True)
    else:
        log(f"{label}: CSV order verified; skipping expensive full-frame sort")
        df = df.reset_index(drop=True)

    log(f"{label}: creating region-local day index")
    df["day_idx"] = df.groupby("region_id", observed=True, sort=False).cumcount().astype("int32")
    return df


def add_daily_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    eps = np.float32(1e-6)
    df["prec_clip"] = df["prec"].clip(lower=0).astype("float32")
    df["prec_log1p"] = np.log1p(df["prec_clip"]).astype("float32")
    df["is_dry_day"] = (df["prec_clip"] < 0.1).astype("float32")
    df["is_wet_day_1mm"] = (df["prec_clip"] >= 1.0).astype("float32")
    df["is_wet_day_5mm"] = (df["prec_clip"] >= 5.0).astype("float32")
    df["is_heavy_precip"] = (df["prec_clip"] >= 10.0).astype("float32")

    df["diurnal_range"] = (df["tmp_max"] - df["tmp_min"]).astype("float32")
    df["heat_day_30"] = (df["tmp_max"] >= 30.0).astype("float32")
    df["heat_day_35"] = (df["tmp_max"] >= 35.0).astype("float32")
    df["cold_day_0"] = (df["tmp_min"] <= 0.0).astype("float32")

    tmp_safe = df["tmp"].clip(lower=-80, upper=80).astype("float32")
    dew_safe = df["dp_tmp"].clip(lower=-80, upper=80).astype("float32")
    e_s = 0.611 * np.exp((17.27 * tmp_safe) / (tmp_safe + 237.3))
    e_a = 0.611 * np.exp((17.27 * dew_safe) / (dew_safe + 237.3))
    df["vpd"] = np.maximum(e_s - e_a, 0).astype("float32")
    df["humidity_deficit"] = (100.0 - df["humidity"]).astype("float32")
    df["temp_dewpoint_spread"] = (df["tmp"] - df["dp_tmp"]).astype("float32")
    df["tmp_minus_wb"] = (df["tmp"] - df["wb_tmp"]).astype("float32")
    df["surf_tmp_minus_tmp"] = (df["surf_tmp"] - df["tmp"]).astype("float32")
    df["wind_stress"] = (df["wind"] * df["vpd"]).astype("float32")
    df["evap_proxy"] = (df["vpd"] * (1.0 + df["wind"].clip(lower=0))).astype("float32")
    df["water_balance"] = (df["prec_clip"] - df["evap_proxy"]).astype("float32")
    df["stress_daily"] = (
        df["vpd"].fillna(0) + 0.03 * df["tmp"].fillna(0) + 0.01 * df["humidity_deficit"].fillna(0)
        - df["prec_log1p"].fillna(0)
    ).astype("float32")
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


def train_endpoint_rows(daily: pd.DataFrame) -> pd.DataFrame:
    endpoints = daily.loc[daily["score"].notna(), [
        "region_id", "region_idx", "date", "year", "month", "day", "doy", "week_of_year",
        "month_sin", "month_cos", "doy_sin", "doy_cos", "season_bin", "day_idx", "score",
    ]].copy()
    endpoints["daily_row_pos"] = endpoints.index.astype("int64")
    endpoints["source"] = "train"
    endpoints["week_idx"] = endpoints.groupby("region_id", observed=True).cumcount().astype("int16")
    endpoints["endpoint_seq"] = endpoints["week_idx"].astype("int16")
    return endpoints.reset_index(drop=True)


def test_endpoint_rows(daily: pd.DataFrame) -> pd.DataFrame:
    pieces: list[pd.DataFrame] = []
    for seq, offset in enumerate(TEST_WEEKLY_OFFSETS):
        part = daily.groupby("region_id", observed=True, sort=False).nth(offset).reset_index()
        part = part[[
            "region_id", "region_idx", "date", "year", "month", "day", "doy", "week_of_year",
            "month_sin", "month_cos", "doy_sin", "doy_cos", "season_bin", "day_idx",
        ]].copy()
        part["daily_row_pos"] = daily.groupby("region_id", observed=True, sort=False).nth(offset).index.to_numpy(dtype="int64")
        part["source"] = "test"
        part["score"] = np.nan
        part["week_idx"] = seq
        part["endpoint_seq"] = seq
        pieces.append(part)
    endpoints = pd.concat(pieces, ignore_index=True)
    endpoints = endpoints.sort_values(["region_id", "endpoint_seq"], kind="mergesort").reset_index(drop=True)
    return endpoints


def rolling_plan() -> dict[str, dict[int, list[str]]]:
    plan: dict[str, dict[int, list[str]]] = {}

    def add(col: str, windows: Iterable[int], aggs: Iterable[str]) -> None:
        spec = plan.setdefault(col, {})
        for window in windows:
            existing = spec.setdefault(window, [])
            for agg in aggs:
                if agg not in existing:
                    existing.append(agg)

    add("prec_clip", WINDOWS, ["sum", "mean", "max", "std"])
    for col in ["is_dry_day", "is_wet_day_1mm", "is_wet_day_5mm", "is_heavy_precip", "heat_day_30", "heat_day_35"]:
        add(col, WINDOWS, ["sum"])
    for col in ["tmp", "tmp_max", "tmp_min", "tmp_range", "surf_tmp", "humidity", "surf_pre", "wind", "wind_max", "vpd", "humidity_deficit", "wind_stress", "water_balance", "stress_daily"]:
        add(col, WINDOWS, ["mean"])
    for col in ["tmp", "humidity", "vpd", "wind", "water_balance", "stress_daily"]:
        add(col, [28, 91], ["std"])
    for col in ["humidity"]:
        add(col, [28, 91], ["min"])
    for col in ["tmp_max", "vpd", "wind_max"]:
        add(col, [28, 91], ["max"])
    return plan


def build_weekly_weather_features(daily: pd.DataFrame, endpoints: pd.DataFrame, label: str) -> pd.DataFrame:
    log(f"Building weekly weather features for {label}: {len(daily):,} daily rows -> {len(endpoints):,} endpoints")
    weekly = endpoints.copy()
    endpoint_pos = weekly["daily_row_pos"].to_numpy(dtype=np.int64)
    new_features: dict[str, np.ndarray] = {}

    current_cols = [
        "prec", "surf_pre", "humidity", "tmp", "dp_tmp", "wb_tmp", "tmp_max", "tmp_min", "tmp_range",
        "surf_tmp", "wind", "wind_max", "wind_min", "wind_range", "prec_clip", "prec_log1p", "vpd",
        "humidity_deficit", "temp_dewpoint_spread", "tmp_minus_wb", "surf_tmp_minus_tmp", "wind_stress",
        "water_balance", "stress_daily",
    ]
    for col in current_cols:
        if col in daily.columns:
            new_features[f"{col}_current"] = daily[col].iloc[endpoint_pos].to_numpy(dtype=np.float32)

    for col, specs in rolling_plan().items():
        if col not in daily.columns:
            continue
        group = daily.groupby("region_id", observed=True, sort=False)[col]
        for window, aggs in specs.items():
            rolled = group.rolling(window=window, min_periods=1).agg(aggs)
            rolled.index = rolled.index.droplevel(0)
            if isinstance(rolled, pd.Series):
                agg = aggs[0]
                new_features[f"{col}_{agg}_{window}d"] = rolled.reindex(endpoint_pos).to_numpy(dtype=np.float32)
            else:
                aligned = rolled.reindex(endpoint_pos)
                for agg in aggs:
                    new_features[f"{col}_{agg}_{window}d"] = aligned[agg].to_numpy(dtype=np.float32)
            del rolled
        log(f"  finished rolling features for {col}")

    for halflife in [7, 30, 90]:
        col = f"stress_ewm_h{halflife}d"
        ewm = daily.groupby("region_id", observed=True, sort=False)["stress_daily"].transform(
            lambda s: s.ewm(halflife=halflife, adjust=False, min_periods=1).mean()
        )
        new_features[col] = ewm.iloc[endpoint_pos].to_numpy(dtype=np.float32)
        del ewm

    if new_features:
        weekly = pd.concat([weekly.reset_index(drop=True), pd.DataFrame(new_features)], axis=1)
    add_weather_trend_features(weekly)
    return compact_float_frame(weekly.copy())


def add_weather_trend_features(df: pd.DataFrame) -> None:
    pairs = [
        ("prec_clip_sum_7d", "prec_clip_sum_28d", 7 / 28, "prec_sum_7_vs_28"),
        ("prec_clip_sum_28d", "prec_clip_sum_91d", 28 / 91, "prec_sum_28_vs_91"),
        ("tmp_mean_7d", "tmp_mean_28d", 1.0, "tmp_mean_7_minus_28"),
        ("tmp_mean_28d", "tmp_mean_91d", 1.0, "tmp_mean_28_minus_91"),
        ("vpd_mean_7d", "vpd_mean_28d", 1.0, "vpd_mean_7_minus_28"),
        ("humidity_mean_7d", "humidity_mean_28d", 1.0, "humidity_mean_7_minus_28"),
        ("water_balance_mean_28d", "water_balance_mean_91d", 1.0, "water_balance_28_minus_91"),
        ("stress_daily_mean_28d", "stress_daily_mean_91d", 1.0, "stress_28_minus_91"),
    ]
    for recent, longer, scale, out in pairs:
        if recent in df.columns and longer in df.columns:
            df[out] = (df[recent] - df[longer] * scale).astype("float32")
    for window in [28, 56, 91]:
        sum_col = f"prec_clip_sum_{window}d"
        wet_col = f"is_wet_day_1mm_sum_{window}d"
        if sum_col in df.columns and wet_col in df.columns:
            df[f"prec_intensity_{window}d"] = (df[sum_col] / np.maximum(df[wet_col], 1)).astype("float32")


def add_weather_climatology(train_weekly: pd.DataFrame, frames: list[pd.DataFrame]) -> list[pd.DataFrame]:
    selected = [
        "prec_clip_sum_28d", "prec_clip_sum_91d", "tmp_mean_28d", "vpd_mean_28d",
        "humidity_mean_28d", "water_balance_mean_91d", "stress_daily_mean_91d",
    ]
    selected = [col for col in selected if col in train_weekly.columns]
    if not selected:
        return frames
    grouped = train_weekly.groupby(["region_id", "month"], observed=True)[selected].agg(["mean", "std"])
    grouped.columns = [f"{feat}_{stat}" for feat, stat in grouped.columns]

    updated: list[pd.DataFrame] = []
    for frame in frames:
        out = frame.copy()
        keys = pd.MultiIndex.from_frame(out[["region_id", "month"]])
        stats = grouped.reindex(keys).reset_index(drop=True)
        for feat in selected:
            mean_col = f"{feat}_mean"
            std_col = f"{feat}_std"
            if mean_col not in stats:
                continue
            mean_values = stats[mean_col].to_numpy(dtype=np.float32)
            std_values = stats[std_col].fillna(0).to_numpy(dtype=np.float32)
            out[f"{feat}_anom_rmonth"] = (out[feat].to_numpy(dtype=np.float32) - mean_values).astype("float32")
            out[f"{feat}_z_rmonth"] = ((out[feat].to_numpy(dtype=np.float32) - mean_values) / (std_values + 1e-3)).astype("float32")
        updated.append(out)
    return updated


def add_targets(weekly: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly.sort_values(["region_id", "week_idx"], kind="mergesort").copy()
    group = weekly.groupby("region_id", observed=True, sort=False)["score"]
    for horizon in range(1, 6):
        weekly[f"target_h{horizon}"] = group.shift(-horizon).astype("float32")
    return weekly


def numeric_feature_columns(frame: pd.DataFrame, extra_exclude: Iterable[str] = ()) -> list[str]:
    exclude = {
        "score", "daily_row_pos", "target_h1", "target_h2", "target_h3", "target_h4", "target_h5",
        *extra_exclude,
    }
    cols = []
    for col in frame.columns:
        if col in exclude or col.startswith("target_h"):
            continue
        if pd.api.types.is_numeric_dtype(frame[col]):
            cols.append(col)
    return cols


def lgb_params(task: str, rows: int) -> dict:
    min_leaf = 50 if rows < 150_000 else 200
    leaves = 63 if rows < 150_000 else 127
    return {
        "objective": "regression_l1",
        "metric": "l1",
        "boosting_type": "gbdt",
        "learning_rate": 0.03 if task == "forecast" else 0.04,
        "num_leaves": leaves,
        "max_depth": -1,
        "min_data_in_leaf": min_leaf,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 5.0,
        "max_bin": 255,
        "verbosity": -1,
        "seed": RANDOM_SEED,
        "feature_pre_filter": False,
        "num_threads": max(1, min(os.cpu_count() or 1, 20)),
    }


def train_lgb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_valid: pd.DataFrame | None,
    y_valid: np.ndarray | None,
    params: dict,
    num_boost_round: int,
    early_stopping_rounds: int,
    categorical_cols: list[str],
    weights: np.ndarray | None = None,
    log_period: int = 100,
) -> lgb.Booster:
    train_set = lgb.Dataset(
        X_train,
        label=y_train,
        weight=weights,
        feature_name=list(X_train.columns),
        categorical_feature=[col for col in categorical_cols if col in X_train.columns],
        free_raw_data=False,
    )
    callbacks = [lgb.log_evaluation(period=log_period)]
    valid_sets = None
    valid_names = None
    if X_valid is not None and y_valid is not None and len(X_valid) > 0:
        valid_set = lgb.Dataset(
            X_valid,
            label=y_valid,
            reference=train_set,
            feature_name=list(X_train.columns),
            categorical_feature=[col for col in categorical_cols if col in X_train.columns],
            free_raw_data=False,
        )
        valid_sets = [valid_set]
        valid_names = ["valid"]
        callbacks.append(lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=True))
    return lgb.train(
        params,
        train_set,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )


def make_nowcast_predictions(
    weekly: pd.DataFrame,
    test_history: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[pd.DataFrame, pd.DataFrame, lgb.Booster, list[str]]:
    log("Training stage-1 nowcast models")
    nowcast_cols = numeric_feature_columns(weekly, extra_exclude={"week_idx", "endpoint_seq"})
    categorical_cols = ["region_idx", "season_bin"]
    X_all = weekly[nowcast_cols]
    y_all = weekly["score"].to_numpy(dtype=np.float32)
    params = lgb_params("nowcast", len(weekly))

    weeks = np.array(sorted(weekly["week_idx"].unique()))
    max_week = int(weeks.max())
    first_val_week = max(40, min(args.min_forecast_week - 20, int(max_week * 0.12)))
    boundaries = np.unique(np.linspace(first_val_week, max_week + 1, 6, dtype=int))
    score_hat = np.full(len(weekly), np.nan, dtype=np.float32)
    best_iterations: list[int] = []

    for start, end in zip(boundaries[:-1], boundaries[1:]):
        train_mask = weekly["week_idx"].to_numpy() < start
        valid_mask = (weekly["week_idx"].to_numpy() >= start) & (weekly["week_idx"].to_numpy() < end)
        if train_mask.sum() == 0 or valid_mask.sum() == 0:
            continue
        log(f"  nowcast fold weeks [{start}, {end}) with {train_mask.sum():,} train / {valid_mask.sum():,} valid rows")
        model = train_lgb(
            X_all.loc[train_mask], y_all[train_mask],
            X_all.loc[valid_mask], y_all[valid_mask],
            params, args.nowcast_rounds, max(20, args.early_stopping_rounds // 2), categorical_cols,
            log_period=200,
        )
        pred = model.predict(X_all.loc[valid_mask], num_iteration=model.best_iteration)
        score_hat[valid_mask] = np.clip(pred, 0, 5).astype(np.float32)
        if model.best_iteration:
            best_iterations.append(int(model.best_iteration))

    final_rounds = int(np.median(best_iterations)) if best_iterations else max(100, args.nowcast_rounds // 2)
    final_rounds = max(50, final_rounds)
    log(f"  fitting final nowcast model for {final_rounds} rounds on all weekly labels")
    final_nowcast = train_lgb(
        X_all, y_all, None, None, params, final_rounds, args.early_stopping_rounds, categorical_cols, log_period=0
    )
    missing = np.isnan(score_hat)
    if missing.any():
        score_hat[missing] = np.clip(final_nowcast.predict(X_all.loc[missing]), 0, 5).astype(np.float32)
    weekly = weekly.copy()
    weekly["score_hat"] = score_hat.astype("float32")
    weekly["score_hat_filled_by_final"] = missing.astype("int8")

    test_history = test_history.copy()
    for col in nowcast_cols:
        if col not in test_history.columns:
            test_history[col] = np.nan
    test_pred = final_nowcast.predict(test_history[nowcast_cols], num_iteration=final_nowcast.best_iteration)
    test_history["score_hat"] = np.clip(test_pred, 0, 5).astype("float32")
    test_history["score_hat_filled_by_final"] = 0
    return weekly, test_history, final_nowcast, nowcast_cols


def add_training_score_state(weekly: pd.DataFrame) -> pd.DataFrame:
    weekly = weekly.sort_values(["region_id", "week_idx"], kind="mergesort").copy()
    group = weekly.groupby("region_id", observed=True, sort=False)
    for lag in range(13):
        weekly[f"score_hat_lag_{lag}"] = group["score_hat"].shift(lag).astype("float32")
    for window in [4, 8, 13]:
        weekly[f"score_hat_mean_{window}"] = group["score_hat"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        ).astype("float32")
        weekly[f"score_hat_max_{window}"] = group["score_hat"].transform(
            lambda s: s.rolling(window, min_periods=1).max()
        ).astype("float32")
    weekly["score_hat_slope_4"] = (weekly["score_hat_lag_0"] - weekly["score_hat_lag_3"]).astype("float32")
    weekly["score_hat_last_minus_mean_13"] = (weekly["score_hat_lag_0"] - weekly["score_hat_mean_13"]).astype("float32")

    for lag in OLDER_TRUE_LAGS:
        col = f"true_score_lag_{lag}"
        weekly[col] = group["score"].shift(lag).astype("float32")
        weekly[f"{col}_missing"] = weekly[col].isna().astype("int8")
    weekly["last_known_train_score"] = weekly["true_score_lag_13"].astype("float32")
    weekly["true_score_mean_lag13_26"] = group["score"].transform(
        lambda s: s.shift(13).rolling(14, min_periods=1).mean()
    ).astype("float32")
    weekly["true_score_mean_lag13_52"] = group["score"].transform(
        lambda s: s.shift(13).rolling(40, min_periods=1).mean()
    ).astype("float32")
    weekly["true_score_hist_mean_lag13"] = group["score"].transform(
        lambda s: s.shift(13).expanding(min_periods=1).mean()
    ).astype("float32")
    weekly["true_score_hist_zero_rate_lag13"] = group["score"].transform(
        lambda s: (s.shift(13) == 0).expanding(min_periods=1).mean()
    ).astype("float32")
    weekly["true_score_hist_severe_rate_lag13"] = group["score"].transform(
        lambda s: (s.shift(13) >= 2).expanding(min_periods=1).mean()
    ).astype("float32")
    return weekly


def add_test_score_state(test_history: pd.DataFrame, train_weekly: pd.DataFrame) -> pd.DataFrame:
    test_history = test_history.sort_values(["region_id", "endpoint_seq"], kind="mergesort").copy()
    group = test_history.groupby("region_id", observed=True, sort=False)
    for lag in range(13):
        test_history[f"score_hat_lag_{lag}"] = group["score_hat"].shift(lag).astype("float32")
    for window in [4, 8, 13]:
        test_history[f"score_hat_mean_{window}"] = group["score_hat"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        ).astype("float32")
        test_history[f"score_hat_max_{window}"] = group["score_hat"].transform(
            lambda s: s.rolling(window, min_periods=1).max()
        ).astype("float32")
    test_history["score_hat_slope_4"] = (test_history["score_hat_lag_0"] - test_history["score_hat_lag_3"]).astype("float32")
    test_history["score_hat_last_minus_mean_13"] = (test_history["score_hat_lag_0"] - test_history["score_hat_mean_13"]).astype("float32")

    score_matrix = train_weekly.pivot(index="region_id", columns="week_idx", values="score").sort_index(axis=1)
    max_week = int(score_matrix.columns.max())
    for lag in OLDER_TRUE_LAGS:
        train_week_for_lag = max_week + 13 - lag
        col = f"true_score_lag_{lag}"
        if train_week_for_lag in score_matrix.columns:
            mapping = score_matrix[train_week_for_lag]
            test_history[col] = test_history["region_id"].map(mapping).astype("float32")
        else:
            test_history[col] = np.nan
        test_history[f"{col}_missing"] = test_history[col].isna().astype("int8")

    last_14 = score_matrix.loc[:, max(max_week - 13, 0):max_week]
    last_40 = score_matrix.loc[:, max(max_week - 39, 0):max_week]
    test_history["last_known_train_score"] = test_history["true_score_lag_13"].astype("float32")
    test_history["true_score_mean_lag13_26"] = test_history["region_id"].map(last_14.mean(axis=1)).astype("float32")
    test_history["true_score_mean_lag13_52"] = test_history["region_id"].map(last_40.mean(axis=1)).astype("float32")
    test_history["true_score_hist_mean_lag13"] = test_history["region_id"].map(score_matrix.mean(axis=1)).astype("float32")
    test_history["true_score_hist_zero_rate_lag13"] = test_history["region_id"].map((score_matrix == 0).mean(axis=1)).astype("float32")
    test_history["true_score_hist_severe_rate_lag13"] = test_history["region_id"].map((score_matrix >= 2).mean(axis=1)).astype("float32")
    return test_history


def sample_weights(frame: pd.DataFrame, target: np.ndarray, max_week: int) -> np.ndarray:
    recency = 1.0 + 0.30 * (frame["week_idx"].to_numpy(dtype=np.float32) / max(max_week, 1))
    severity = 1.0 + 0.20 * (target >= 1) + 0.45 * (target >= 2) + 0.70 * (target >= 3)
    return (recency * severity).astype("float32")


def train_forecast_models(
    weekly: pd.DataFrame,
    test_final: pd.DataFrame,
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[dict[int, lgb.Booster], pd.DataFrame, pd.DataFrame, list[str]]:
    feature_cols = numeric_feature_columns(weekly)
    for col in feature_cols:
        if col not in test_final.columns:
            test_final[col] = np.nan
    categorical_cols = [col for col in ["region_idx", "season_bin"] if col in feature_cols]
    max_week = int(weekly["week_idx"].max())
    valid_offsets = [int(x) for x in args.valid_origin_offsets.split(",") if x.strip()]
    valid_origins = sorted({max_week - offset for offset in valid_offsets if max_week - offset >= args.min_forecast_week})
    if not valid_origins:
        valid_origins = [max_week - 5]
    min_valid_origin = min(valid_origins)
    log(f"Forecast validation origins: {valid_origins}")

    models: dict[int, lgb.Booster] = {}
    valid_rows: list[pd.DataFrame] = []
    pred_frame = test_final[["region_id"]].copy()
    params = lgb_params("forecast", len(weekly))

    model_dir = out_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    for horizon in range(1, 6):
        target_col = f"target_h{horizon}"
        y = weekly[target_col].to_numpy(dtype=np.float32)
        valid_mask = weekly["week_idx"].isin(valid_origins).to_numpy() & np.isfinite(y)
        train_mask = (
            (weekly["week_idx"].to_numpy() >= args.min_forecast_week)
            & (weekly["week_idx"].to_numpy() < min_valid_origin)
            & np.isfinite(y)
        )
        final_mask = (weekly["week_idx"].to_numpy() >= args.min_forecast_week) & np.isfinite(y)
        log(f"Training horizon {horizon}: {train_mask.sum():,} train / {valid_mask.sum():,} valid / {final_mask.sum():,} final rows")

        weights = None
        if not args.no_sample_weights:
            weights = sample_weights(weekly.loc[train_mask], y[train_mask], max_week)
        model_valid = train_lgb(
            weekly.loc[train_mask, feature_cols], y[train_mask],
            weekly.loc[valid_mask, feature_cols], y[valid_mask],
            params, args.num_boost_round, args.early_stopping_rounds, categorical_cols, weights=weights,
        )
        valid_pred = np.clip(model_valid.predict(weekly.loc[valid_mask, feature_cols], num_iteration=model_valid.best_iteration), 0, 5)
        fold = weekly.loc[valid_mask, ["region_id", "week_idx", "month", target_col]].copy()
        fold = fold.rename(columns={target_col: "target"})
        fold["horizon"] = horizon
        fold["prediction"] = valid_pred.astype("float32")
        fold["abs_error"] = np.abs(fold["target"] - fold["prediction"]).astype("float32")
        fold["baseline_zero"] = 0.0
        if "score_hat_lag_0" in weekly.columns:
            fold["baseline_nowcast"] = weekly.loc[valid_mask, "score_hat_lag_0"].to_numpy(dtype=np.float32)
        if "true_score_lag_13" in weekly.columns:
            fold["baseline_last_known"] = weekly.loc[valid_mask, "true_score_lag_13"].to_numpy(dtype=np.float32)
        valid_rows.append(fold)

        best_iter = int(model_valid.best_iteration or args.num_boost_round)
        best_iter = max(best_iter, 50)
        final_weights = None
        if not args.no_sample_weights:
            final_weights = sample_weights(weekly.loc[final_mask], y[final_mask], max_week)
        log(f"  fitting final horizon {horizon} model for {best_iter} rounds")
        model_final = train_lgb(
            weekly.loc[final_mask, feature_cols], y[final_mask],
            None, None, params, best_iter, args.early_stopping_rounds, categorical_cols, weights=final_weights, log_period=0,
        )
        models[horizon] = model_final
        model_final.save_model(str(model_dir / f"lgbm_h{horizon}.txt"))
        importance = pd.DataFrame({
            "feature": feature_cols,
            "gain": model_final.feature_importance(importance_type="gain"),
            "split": model_final.feature_importance(importance_type="split"),
        }).sort_values("gain", ascending=False)
        importance.to_csv(out_dir / f"feature_importance_h{horizon}.csv", index=False)
        pred = np.clip(model_final.predict(test_final[feature_cols], num_iteration=model_final.best_iteration), 0, 5)
        pred_frame[f"pred_week{horizon}"] = pred.astype("float32")

    valid_pred_df = pd.concat(valid_rows, ignore_index=True)
    summary = valid_pred_df.groupby("horizon", as_index=False)["abs_error"].mean().rename(columns={"abs_error": "mae"})
    summary.loc[len(summary)] = [0, valid_pred_df["abs_error"].mean()]
    return models, pred_frame, valid_pred_df, summary, feature_cols


def write_submission(pred_frame: pd.DataFrame, sample: pd.DataFrame, out_dir: Path, smoke: bool) -> Path:
    pred_frame = pred_frame.copy()
    pred_frame["region_id"] = pred_frame["region_id"].astype(str)
    sample = sample.copy()
    sample["region_id"] = sample["region_id"].astype(str)
    submission = sample[["region_id"]].merge(pred_frame, on="region_id", how="left")
    submission = submission[sample.columns.tolist()]
    pred_cols = [col for col in submission.columns if col != "region_id"]
    values = submission[pred_cols].to_numpy(dtype=np.float32)
    assert submission.shape == sample.shape, (submission.shape, sample.shape)
    assert submission.columns.tolist() == sample.columns.tolist()
    assert submission["region_id"].equals(sample["region_id"])
    assert np.isfinite(values).all(), "Submission contains NaN or infinite predictions"
    assert (values >= 0).all() and (values <= 5).all(), "Submission predictions are outside [0, 5]"
    output = out_dir / ("lgbm_submission_smoke.csv" if smoke else "lgbm_submission.csv")
    submission.to_csv(output, index=False)
    return output


def main() -> None:
    args = parse_args()
    if args.mode == "smoke" and args.max_regions is None:
        args.max_regions = 96
    if args.mode == "smoke":
        args.num_boost_round = min(args.num_boost_round, 120)
        args.nowcast_rounds = min(args.nowcast_rounds, 100)
        args.early_stopping_rounds = min(args.early_stopping_rounds, 30)

    out_dir = Path(args.out_dir)
    if args.mode == "smoke" or args.max_regions is not None:
        out_dir = out_dir / "smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "validation").mkdir(exist_ok=True)

    sample = pd.read_csv(args.sample_path)
    region_order = sample["region_id"].astype(str).tolist()
    region_filter = None
    if args.max_regions is not None:
        selected = region_order[: args.max_regions]
        region_filter = set(selected)
        sample = sample[sample["region_id"].astype(str).isin(region_filter)].reset_index(drop=True)
        region_order = selected
        log(f"Smoke/filtered run using {len(region_order)} regions")

    with timed_step("load train/test CSV files"):
        train_daily = read_csv_maybe_filtered(args.train_path, has_score=True, region_filter=region_filter)
        test_daily = read_csv_maybe_filtered(args.test_path, has_score=False, region_filter=region_filter)
        log(f"Loaded train {train_daily.shape}, test {test_daily.shape}")

    with timed_step("prepare train order/date/index"):
        train_daily = prepare_daily_order(train_daily, region_order, label="train", sort_input=args.sort_input)
    with timed_step("prepare test order/date/index"):
        test_daily = prepare_daily_order(test_daily, region_order, label="test", sort_input=args.sort_input)

    with timed_step("derive train daily weather features"):
        train_daily = add_daily_derived_features(train_daily)
    with timed_step("derive test daily weather features"):
        test_daily = add_daily_derived_features(test_daily)

    with timed_step("extract weekly endpoint rows"):
        train_endpoints = train_endpoint_rows(train_daily)
        test_endpoints = test_endpoint_rows(test_daily)
        log(f"Weekly endpoints: train {train_endpoints.shape}, test history {test_endpoints.shape}")

    with timed_step("build train weekly rolling weather features"):
        weekly_train = build_weekly_weather_features(train_daily, train_endpoints, "train")
    with timed_step("build test weekly rolling weather features"):
        weekly_test_history = build_weekly_weather_features(test_daily, test_endpoints, "test")

    with timed_step("add climatology and horizon targets"):
        weekly_train, weekly_test_history = add_weather_climatology(weekly_train, [weekly_train, weekly_test_history])
        weekly_train = add_targets(weekly_train)

    with timed_step("train nowcast model and impute test score history"):
        weekly_train, weekly_test_history, nowcast_model, nowcast_cols = make_nowcast_predictions(weekly_train, weekly_test_history, args)

    with timed_step("add score-state features"):
        weekly_train = add_training_score_state(weekly_train)
        weekly_test_history = add_test_score_state(weekly_test_history, weekly_train)
        test_final = weekly_test_history.loc[weekly_test_history["endpoint_seq"] == max(TEST_WEEKLY_OFFSETS) // 7].copy()
        if len(test_final) != len(sample):
            # Endpoint sequence is ordinal 12, while max(TEST_WEEKLY_OFFSETS)//7 also equals 12 for the 91-day window.
            test_final = weekly_test_history.loc[weekly_test_history["endpoint_seq"] == 12].copy()
        test_final = test_final.sort_values("region_idx", kind="mergesort").reset_index(drop=True)

    with timed_step("train five horizon models and predict test"):
        models, pred_frame, valid_pred_df, valid_summary, feature_cols = train_forecast_models(
            weekly_train, test_final, args, out_dir
        )
    valid_pred_df.to_csv(out_dir / "validation" / "validation_predictions.csv", index=False)
    valid_summary.to_csv(out_dir / "validation" / "validation_mae_by_horizon.csv", index=False)
    log("Validation MAE by horizon, with horizon 0 as overall:")
    print(valid_summary.to_string(index=False))

    nowcast_model.save_model(str(out_dir / "models" / "lgbm_nowcast.txt"))
    with open(out_dir / "feature_columns.json", "w") as f:
        json.dump(feature_cols, f, indent=2)
    with open(out_dir / "nowcast_feature_columns.json", "w") as f:
        json.dump(nowcast_cols, f, indent=2)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    output = write_submission(pred_frame, sample, out_dir, smoke=(args.mode == "smoke" or args.max_regions is not None))
    log(f"Saved submission: {output}")
    log("Head of submission:")
    print(pd.read_csv(output).head(10).to_string(index=False))


if __name__ == "__main__":
    main()
