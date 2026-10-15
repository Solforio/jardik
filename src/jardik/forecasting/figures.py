"""Figures of the backtest report (matplotlib, saved as PNG)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no window: figures are only written to files
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

CHALLENGER_COLOUR = "#1f77b4"
BASELINE_COLOURS = ["#7f7f7f", "#bcbd22", "#8c564b", "#9467bd", "#e377c2"]
DPI = 130


def _colour(model: str, challenger: str, order: list[str]) -> str:
    if model == challenger:
        return CHALLENGER_COLOUR
    others = [m for m in order if m != challenger]
    return BASELINE_COLOURS[others.index(model) % len(BASELINE_COLOURS)]


def _save(fig: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def metric_bars(
    metrics: pd.DataFrame, path: Path, challenger: str, metric: str = "mae", unit: str = "kWh/h"
) -> Path:
    """Error of every model in every scenario (lower is better)."""
    scenarios = list(dict.fromkeys(metrics["scenario"]))
    models = list(dict.fromkeys(metrics["model"]))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        values = [
            metrics[(metrics["scenario"] == s) & (metrics["model"] == model)][metric].iloc[0]
            for s in scenarios
        ]
        ax.bar(
            np.arange(len(scenarios)) + i * width,
            values,
            width,
            label=model,
            color=_colour(model, challenger, models),
        )
    ax.set_xticks(np.arange(len(scenarios)) + 0.4 - width / 2, scenarios)
    ax.set_ylabel(f"{metric.upper()} ({unit})")
    ax.set_title(f"{metric.upper()} by scenario (lower is better)")
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    return _save(fig, path)


def _median_building(predictions: pd.DataFrame, scenario: str, model: str) -> str:
    rows = predictions[predictions["scenario"] == scenario]
    error = (rows[f"pred__{model}"] - rows["target_kwh"]).abs().groupby(rows["building_id"]).mean()
    error = error.dropna().sort_values()
    return str(error.index[len(error) // 2])


def example_week(
    predictions: pd.DataFrame,
    path: Path,
    challenger: str,
    baseline: str,
    scenario: str,
    interval: tuple[float, float] = (0.1, 0.9),
) -> dict[str, str]:
    """One week of a typical test building: actual load, both forecasts and the 80 % band."""
    building = _median_building(predictions, scenario, challenger)
    rows = predictions[
        (predictions["scenario"] == scenario) & (predictions["building_id"] == building)
    ]
    rows = rows.sort_values("ts_utc")
    earliest = rows["ts_utc"].min() + pd.Timedelta(days=14)
    mondays = rows[
        (rows["ts_utc"] >= earliest)
        & (rows["ts_utc"].dt.dayofweek == 0)
        & (rows["ts_utc"].dt.hour == 0)
    ]
    start = mondays["ts_utc"].iloc[0] if len(mondays) else rows["ts_utc"].iloc[0]
    week = rows[(rows["ts_utc"] >= start) & (rows["ts_utc"] < start + pd.Timedelta(days=7))]

    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    low, high = (f"q{q:g}__{challenger}" for q in interval)
    if low in week.columns and high in week.columns:
        ax.fill_between(
            week["ts_utc"],
            week[low],
            week[high],
            color=CHALLENGER_COLOUR,
            alpha=0.18,
            label=f"{challenger} {int((interval[1] - interval[0]) * 100)} % interval",
        )
    ax.plot(week["ts_utc"], week["target_kwh"], color="black", lw=1.4, label="actual")
    ax.plot(
        week["ts_utc"],
        week[f"pred__{challenger}"],
        color=CHALLENGER_COLOUR,
        lw=1.2,
        label=challenger,
    )
    ax.plot(
        week["ts_utc"], week[f"pred__{baseline}"], color="#7f7f7f", lw=1.0, ls="--", label=baseline
    )
    ax.set_ylabel("kWh per hour")
    ax.set_title(f"One test week, {building} (UTC)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    _save(fig, path)
    return {"building_id": building, "week_start": start.isoformat()}


def error_by_hour(
    predictions: pd.DataFrame, path: Path, challenger: str, baseline: str, scenario: str
) -> Path:
    """Mean absolute error by local hour of the day: when is each model wrong?"""
    rows = predictions[predictions["scenario"] == scenario].dropna(
        subset=["target_kwh", f"pred__{challenger}", f"pred__{baseline}"]
    )
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for model, colour, style in ((challenger, CHALLENGER_COLOUR, "-"), (baseline, "#7f7f7f", "--")):
        err = (rows[f"pred__{model}"] - rows["target_kwh"]).abs().groupby(rows["hour_local"]).mean()
        ax.plot(err.index, err.values, color=colour, ls=style, marker="o", ms=3, label=model)
    ax.set_xlabel("local hour of day")
    ax.set_ylabel("MAE (kWh/h)")
    ax.set_title(f"Error by hour of day, {scenario}")
    ax.set_xticks(range(0, 24, 2))
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    return _save(fig, path)


def building_scatter(
    per_building: pd.DataFrame, path: Path, challenger: str, baseline: str, scenario: str
) -> Path:
    """CV(RMSE) of every test building: below the diagonal the challenger wins."""
    rows = per_building[per_building["scenario"] == scenario]
    wide = rows.pivot(index="building_id", columns="model", values="cv_rmse").dropna(
        subset=[challenger, baseline]
    )
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    ax.scatter(
        wide[baseline] * 100, wide[challenger] * 100, s=18, color=CHALLENGER_COLOUR, alpha=0.8
    )
    top = (
        float(max(wide[baseline].max(), wide[challenger].max()) * 100 * 1.05) if len(wide) else 1.0
    )
    ax.plot([0, top], [0, top], color="black", lw=0.8, ls=":")
    wins = int((wide[challenger] < wide[baseline]).sum())
    ax.set_xlabel(f"{baseline}: CV(RMSE) %")
    ax.set_ylabel(f"{challenger}: CV(RMSE) %")
    ax.set_title(f"{challenger} wins in {wins} of {len(wide)} buildings")
    ax.grid(alpha=0.3)
    return _save(fig, path)


def feature_importance(importance: pd.DataFrame, path: Path, top: int = 15) -> Path:
    """Share of the total gain of the mean model, per feature."""
    table = importance.head(top).iloc[::-1]
    share = table["gain"] / importance["gain"].sum() * 100
    fig, ax = plt.subplots(figsize=(7, 0.28 * len(table) + 1))
    ax.barh(table["feature"], share, color=CHALLENGER_COLOUR)
    ax.set_xlabel("share of total gain (%)")
    ax.set_title("What drives the model")
    ax.grid(axis="x", alpha=0.3)
    return _save(fig, path)
