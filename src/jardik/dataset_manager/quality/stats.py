"""Per-series quality statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _longest_run(mask: np.ndarray) -> int:
    if not mask.any():
        return 0
    edges = np.diff(np.concatenate(([False], mask, [False])).astype(int))
    return int((np.flatnonzero(edges == -1) - np.flatnonzero(edges == 1)).max())


def _near_zero_ratio(measured: pd.Series) -> float:
    """Share of measured values at or below 1 % of the typical (median) value.

    A partial or dead meter shows long stretches of readings like 0.0001 kWh that exact-zero
    checks miss; a real office never drops that far below its median load.
    """
    if measured.empty:
        return float("nan")
    threshold = max(0.01 * float(measured.median()), 0.0)
    return float((measured <= threshold).mean())


def summarise_series(
    raw: pd.Series, filled: pd.Series, imputed: pd.Series, step_s: int = 3600
) -> dict[str, float | int]:
    """Counts and ratios for one building over its expected time slots.

    raw      values as measured, NaN where missing (one entry per expected slot)
    filled   values after the optional imputation
    imputed  True where a value was imputed

    missing_ratio           raw gaps before imputation
    imputed_ratio           share of slots that carry an imputed value
    residual_missing_ratio  gaps that remain after imputation (what a user can not use)
    """
    expected = len(raw)
    missing = int(raw.isna().sum())
    n_imputed = int(imputed.sum())
    residual = int(filled.isna().sum())
    present = expected - missing
    measured = raw.dropna()

    def ratio(count: int) -> float:
        return count / expected if expected else float("nan")

    return {
        "expected_rows": expected,
        "present_rows": present,
        "missing_rows": missing,
        "imputed_rows": n_imputed,
        "residual_missing_rows": residual,
        "missing_ratio": ratio(missing),
        "imputed_ratio": ratio(n_imputed),
        "residual_missing_ratio": ratio(residual),
        "longest_gap_h": _longest_run(filled.isna().to_numpy()) * step_s / 3600.0,
        "zero_ratio": float((measured == 0).mean()) if len(measured) else float("nan"),
        "near_zero_ratio": _near_zero_ratio(measured),
    }
