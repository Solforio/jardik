"""Forecast accuracy metrics.

Conventions
  * pairs where the actual value or the forecast is missing are ignored; with no pair left the
    result is NaN (never a silent zero);
  * relative metrics (CV(RMSE), NMBE) divide by the mean of the actual values and are fractions
    (0.25 = 25 %); they are NaN when that mean is zero;
  * NMBE follows ASHRAE Guideline 14: sum(actual - forecast) / (n * mean(actual)). A POSITIVE NMBE
    therefore means the model under-forecasts, a negative one that it over-forecasts.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

ArrayLike = Sequence[float] | np.ndarray | pd.Series


def _pairs(actual: ArrayLike, pred: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float)
    p = np.asarray(pred, dtype=float)
    if a.shape != p.shape:
        raise ValueError(f"actual and forecast must have the same length ({a.size} vs {p.size})")
    keep = np.isfinite(a) & np.isfinite(p)
    return a[keep], p[keep]


def mae(actual: ArrayLike, pred: ArrayLike) -> float:
    a, p = _pairs(actual, pred)
    return float(np.mean(np.abs(a - p))) if a.size else float("nan")


def rmse(actual: ArrayLike, pred: ArrayLike) -> float:
    a, p = _pairs(actual, pred)
    return float(np.sqrt(np.mean((a - p) ** 2))) if a.size else float("nan")


def cv_rmse(actual: ArrayLike, pred: ArrayLike) -> float:
    """Coefficient of variation of the RMSE: RMSE divided by the mean of the actual values."""
    a, p = _pairs(actual, pred)
    if not a.size or np.mean(a) == 0:
        return float("nan")
    return float(np.sqrt(np.mean((a - p) ** 2)) / np.mean(a))


def nmbe(actual: ArrayLike, pred: ArrayLike) -> float:
    """Normalised mean bias error, ASHRAE Guideline 14 sign (positive = under-forecast)."""
    a, p = _pairs(actual, pred)
    if not a.size or np.mean(a) == 0:
        return float("nan")
    return float(np.sum(a - p) / (a.size * np.mean(a)))


def evaluate_point(actual: ArrayLike, pred: ArrayLike) -> dict[str, float | int]:
    a, _ = _pairs(actual, pred)
    return {
        "n": int(a.size),
        "mae": mae(actual, pred),
        "rmse": rmse(actual, pred),
        "cv_rmse": cv_rmse(actual, pred),
        "nmbe": nmbe(actual, pred),
    }


def pinball_loss(actual: ArrayLike, pred: ArrayLike, tau: float) -> float:
    """Quantile (pinball) loss of a forecast of the `tau` quantile."""
    if not 0.0 < tau < 1.0:
        raise ValueError(f"tau must be strictly between 0 and 1, got {tau}")
    a, p = _pairs(actual, pred)
    if not a.size:
        return float("nan")
    diff = a - p
    return float(np.mean(np.maximum(tau * diff, (tau - 1.0) * diff)))


def mean_pinball_loss(actual: ArrayLike, quantile_forecasts: pd.DataFrame) -> float:
    """Average pinball loss over the quantile columns (column labels are the quantiles).

    Only rows where every quantile forecast exists are used, so all quantiles see the same rows.
    """
    a = np.asarray(actual, dtype=float)
    q = quantile_forecasts.to_numpy(dtype=float)
    keep = np.isfinite(a) & np.isfinite(q).all(axis=1)
    if not keep.any():
        return float("nan")
    losses = [
        pinball_loss(a[keep], q[keep, i], float(tau))
        for i, tau in enumerate(quantile_forecasts.columns)
    ]
    return float(np.mean(losses))


def interval_coverage(actual: ArrayLike, low: ArrayLike, high: ArrayLike) -> float:
    """Share of actual values inside [low, high]."""
    a = np.asarray(actual, dtype=float)
    lo, hi = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    keep = np.isfinite(a) & np.isfinite(lo) & np.isfinite(hi)
    if not keep.any():
        return float("nan")
    return float(np.mean((a[keep] >= lo[keep]) & (a[keep] <= hi[keep])))


def interval_width(low: ArrayLike, high: ArrayLike) -> float:
    lo, hi = np.asarray(low, dtype=float), np.asarray(high, dtype=float)
    keep = np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean(hi[keep] - lo[keep])) if keep.any() else float("nan")


def skill_score(metric: float, baseline_metric: float) -> float:
    """1 - metric / baseline: positive when the model beats the baseline (error metrics)."""
    if not np.isfinite(metric) or not np.isfinite(baseline_metric) or baseline_metric == 0:
        return float("nan")
    return float(1.0 - metric / baseline_metric)
