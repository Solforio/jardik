"""Feature engineering for day-ahead hourly load forecasting.

The forecast origin is LOCAL MIDNIGHT: at that moment the 24 (or 23 / 25 on DST days) slots of the
coming day are forecast. A feature of slot T may use
  * the load at or before the origin (lags >= hours_ahead + 1, statistics ending at the origin);
  * calendar information;
  * the weather of slot T and of the day, which stands for the weather forecast (known in
    advance). The weather in the dataset is observed, not forecast: see the limitations.
It must never use load at or after the origin. `tests/unit/test_features.py` proves this by
changing all load from the origin onwards and checking that no feature of the day moves.

Imputed values are never used as features: interpolation looks ahead of a gap, so a value imputed
before the origin could carry information from after it.

Normalisation: every building has a different size, so load-like features are divided by `scale_kwh`
(the mean load of the 168 hours before the origin) and the model predicts a ratio that is
multiplied back. One model can then serve buildings of very different size.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

REQUIRED = [
    "building_id",
    "ts_utc",
    "ts_local",
    "target_kwh",
    "is_missing",
    "is_imputed",
    "hour_local",
    "dayofweek_local",
    "month_local",
    "is_weekend_local",
]
DEFAULT_WEATHER = ["temp_c", "dewpoint_c", "wind_ms", "pressure_msl_hpa"]
CALENDAR = [
    "hour_local",
    "dayofweek_local",
    "month_local",
    "is_weekend_local",
    "hours_ahead",
    "doy_sin",
    "doy_cos",
]
ORIGIN_STATS = ["n_last", "n_mean_24", "n_std_168", "n_max_24", "n_min_24"]
# Same-slot values of the last weeks: the base forecast a residual model corrects.
BASE_LAGS = (168, 336, 504, 672)
BASE = ["n_base"]
TEMPERATURE = ["temp_lag_24", "temp_day_mean", "temp_delta_24", "hdd", "cdd"]
STATIC = ["log_scale", "floor_area_m2", "intensity_kwh_m2"]


class FeatureConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lags: tuple[int, ...] = (24, 48, 168, 336, 504, 672)
    scale_window: int = Field(default=168, gt=0)
    stat_window: int = Field(default=24, gt=0)
    min_history_obs: int = Field(default=72, gt=0)
    weather_columns: tuple[str, ...] = tuple(DEFAULT_WEATHER)
    hdd_base_c: float = 18.0
    cdd_base_c: float = 21.0


def model_feature_columns(config: FeatureConfig) -> list[str]:
    """Columns the models are trained on (normalised load features, never raw kWh lags)."""
    return [
        *[f"n_lag_{k}" for k in config.lags],
        *ORIGIN_STATS,
        *BASE,
        *CALENDAR,
        *config.weather_columns,
        *TEMPERATURE,
        *STATIC,
    ]


def _at_origin(series: pd.Series, positions: np.ndarray, hours_ahead: np.ndarray) -> np.ndarray:
    """Value of a (shifted) rolling statistic at the origin slot of each row."""
    origin = positions - hours_ahead
    values = series.to_numpy(dtype=float)
    out = np.full(len(positions), np.nan)
    ok = origin >= 0
    out[ok] = values[origin[ok]]
    return out


def _base_forecast(
    columns: dict[str, np.ndarray], cfg: FeatureConfig, scale_kwh: np.ndarray
) -> np.ndarray:
    """Mean of the same slot in the last weeks; else yesterday's value; else the scale.

    Every ingredient is a lag or a statistic that ends at the origin, so it is causal.
    """
    weekly = [columns[f"lag_{k}"] for k in cfg.lags if k in BASE_LAGS]
    base = np.full(len(scale_kwh), np.nan)
    if weekly:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # rows where every lag is missing
            base = np.nanmean(np.column_stack(weekly), axis=1)
    if 24 in cfg.lags:
        base = np.where(np.isnan(base), columns["lag_24"], base)
    return np.where(np.isnan(base), scale_kwh, base)


def build_features(merged: pd.DataFrame, config: FeatureConfig | None = None) -> pd.DataFrame:
    """One feature row per slot of `merged_training_dataset` (same rows, building/time order).

    Rows without enough history have `has_history = False` (no scale); rows whose target is
    missing or imputed have `is_measured = False`. Both are kept so the grid stays complete.
    """
    cfg = config or FeatureConfig()
    absent = [c for c in REQUIRED if c not in merged.columns]
    if absent:
        raise KeyError(f"merged dataset lacks required column(s): {absent}")

    df = merged.sort_values(["building_id", "ts_utc"], kind="stable").reset_index(drop=True)
    local_day = df["ts_local"].dt.normalize()
    hours_ahead = df.groupby(["building_id", local_day]).cumcount().to_numpy()

    usable_history = ~df["is_missing"].astype(bool) & ~df["is_imputed"].astype(bool)
    history = df["target_kwh"].where(usable_history)
    n_rows = len(df)

    columns: dict[str, np.ndarray] = {
        "scale_kwh": np.full(n_rows, np.nan),
        "n_last": np.full(n_rows, np.nan),
        "n_mean_24": np.full(n_rows, np.nan),
        "n_std_168": np.full(n_rows, np.nan),
        "n_max_24": np.full(n_rows, np.nan),
        "n_min_24": np.full(n_rows, np.nan),
        **{f"lag_{k}": np.full(n_rows, np.nan) for k in cfg.lags},
    }

    for _, rows in df.groupby("building_id").indices.items():
        positions = np.arange(len(rows))
        ahead = hours_ahead[rows]
        s = pd.Series(history.to_numpy()[rows])
        half = max(1, cfg.stat_window // 2)

        stats = {
            "scale_kwh": s.rolling(cfg.scale_window, min_periods=cfg.min_history_obs)
            .mean()
            .shift(1),
            "last": s.shift(1),
            "mean_24": s.rolling(cfg.stat_window, min_periods=half).mean().shift(1),
            "std_168": s.rolling(cfg.scale_window, min_periods=cfg.min_history_obs).std().shift(1),
            "max_24": s.rolling(cfg.stat_window, min_periods=half).max().shift(1),
            "min_24": s.rolling(cfg.stat_window, min_periods=half).min().shift(1),
        }
        scale = _at_origin(stats["scale_kwh"], positions, ahead)
        columns["scale_kwh"][rows] = scale
        columns["n_last"][rows] = _at_origin(stats["last"], positions, ahead)
        columns["n_mean_24"][rows] = _at_origin(stats["mean_24"], positions, ahead)
        columns["n_std_168"][rows] = _at_origin(stats["std_168"], positions, ahead)
        columns["n_max_24"][rows] = _at_origin(stats["max_24"], positions, ahead)
        columns["n_min_24"][rows] = _at_origin(stats["min_24"], positions, ahead)
        for k in cfg.lags:
            lag = s.shift(k).to_numpy(dtype=float, copy=True)
            lag[ahead >= k] = np.nan  # slot T-k would be at or after the origin: not known yet
            columns[f"lag_{k}"][rows] = lag

    scale_kwh = columns["scale_kwh"]
    base = _base_forecast(columns, cfg, scale_kwh)
    out: dict[str, Any] = {
        "building_id": df["building_id"],
        "site_id": df["site_id"] if "site_id" in df.columns else df["building_id"],
        "ts_utc": df["ts_utc"],
        "ts_local": df["ts_local"],
        "target_kwh": df["target_kwh"].astype(float),
        "is_measured": usable_history & df["target_kwh"].notna(),
        "has_history": pd.Series(np.isfinite(scale_kwh) & (scale_kwh > 0), index=df.index),
        "hours_ahead": pd.Series(hours_ahead, index=df.index).astype("int64"),
        "scale_kwh": scale_kwh,
        "base_kwh": base,
    }
    for k in cfg.lags:
        out[f"lag_{k}"] = columns[f"lag_{k}"]
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in cfg.lags:
            out[f"n_lag_{k}"] = (columns[f"lag_{k}"] / scale_kwh).astype("float32")
        for name in ORIGIN_STATS:
            out[name] = (columns[name] / scale_kwh).astype("float32")
        out["n_base"] = (base / scale_kwh).astype("float32")

    doy = df["ts_local"].dt.dayofyear.to_numpy(dtype=float)
    for name in ("hour_local", "dayofweek_local", "month_local"):
        out[name] = df[name].astype("int64")
    out["is_weekend_local"] = df["is_weekend_local"].astype(int)
    out["doy_sin"] = np.sin(2 * np.pi * doy / 365.25).astype("float32")
    out["doy_cos"] = np.cos(2 * np.pi * doy / 365.25).astype("float32")

    for name in cfg.weather_columns:
        out[name] = df[name].astype("float32") if name in df.columns else np.float32("nan")
    temp = (
        df["temp_c"].astype(float) if "temp_c" in df.columns else pd.Series(np.nan, index=df.index)
    )
    by_building = temp.groupby(df["building_id"])
    temp_lag_24 = by_building.shift(24)
    out["temp_lag_24"] = temp_lag_24.astype("float32")
    out["temp_day_mean"] = (
        temp.groupby([df["building_id"], local_day]).transform("mean").astype("float32")
    )
    out["temp_delta_24"] = (temp - temp_lag_24).astype("float32")
    out["hdd"] = np.maximum(0, cfg.hdd_base_c - temp).astype("float32")
    out["cdd"] = np.maximum(0, temp - cfg.cdd_base_c).astype("float32")

    area = df["floor_area_m2"].astype(float) if "floor_area_m2" in df.columns else np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log_scale"] = np.log(scale_kwh).astype("float32")
        out["floor_area_m2"] = pd.Series(area, index=df.index).astype("float32")
        out["intensity_kwh_m2"] = (scale_kwh / area).astype("float32")
    return pd.DataFrame(out)
