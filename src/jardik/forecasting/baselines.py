"""Baseline forecasters. Every model beyond these has to beat them on held-out data.

All three read lag columns of the feature frame, so they are strictly causal and are scored by
exactly the same backtest code as the learned models.
"""

from __future__ import annotations

import pandas as pd

from jardik.core.interfaces import Forecaster


class ColumnBaseline:
    """Forecast = mean of the available values of some lag columns (NaN if none is available)."""

    supports_quantiles = False

    def __init__(self, name: str, columns: tuple[str, ...]) -> None:
        self.name = name
        self.columns = columns

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None) -> None:
        return None  # nothing to learn

    def predict(self, features: pd.DataFrame) -> pd.Series:
        missing = [c for c in self.columns if c not in features.columns]
        if missing:
            raise KeyError(f"{self.name} needs feature column(s) {missing}")
        return features[list(self.columns)].mean(axis=1, skipna=True).astype(float)

    def predict_quantiles(self, features: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
        raise NotImplementedError(f"{self.name} is a point forecast baseline: no quantiles")


def default_baselines() -> dict[str, Forecaster]:
    """The baselines of the plan: yesterday, last week, and the mean of the last four weeks."""
    models: list[ColumnBaseline] = [
        ColumnBaseline("naive_lag_24", ("lag_24",)),
        ColumnBaseline("seasonal_naive_168", ("lag_168",)),
        ColumnBaseline("hourly_weekday_mean_4w", ("lag_168", "lag_336", "lag_504", "lag_672")),
    ]
    return {m.name: m for m in models}  # type: ignore[misc]
