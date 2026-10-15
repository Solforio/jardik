"""Backtest engine: fit every model of the ladder on the same split and score them identically.

All models implement the `Forecaster` protocol and read the same feature frame, so "each step
must beat the previous one on held-out data" is a mechanical check (the gate), not a judgement.

Fairness rules
  * models are fitted only on the training scenario (validation data only stops the boosting);
  * every model is scored on exactly the same rows: those with a measured target and a forecast
    from every model. `row_coverage` says what share of the eligible rows that is.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from jardik.core.interfaces import Forecaster
from jardik.core.logging import get_logger
from jardik.forecasting.metrics import (
    evaluate_point,
    interval_coverage,
    interval_width,
    mean_pinball_loss,
    skill_score,
)
from jardik.forecasting.splits import SplitSpec, assign_roles, partition

log = get_logger("forecasting.backtest")

EVAL_SCENARIOS = ("val", "test_time", "test_building", "test_both")
DEFAULT_QUANTILES = (0.1, 0.5, 0.9)


@dataclass
class BacktestResult:
    spec: SplitSpec
    roles: pd.Series
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    per_building: pd.DataFrame
    macro: pd.DataFrame
    skill: pd.DataFrame
    gate: dict[str, Any]
    models: dict[str, Forecaster]


def _quantile_label(q: float) -> str:
    return f"q{q:g}"


def _predict_scenario(
    frame: pd.DataFrame, models: Mapping[str, Forecaster], quantiles: tuple[float, ...]
) -> pd.DataFrame:
    out = frame[["building_id", "ts_utc", "target_kwh", "hour_local"]].copy()
    for name, model in models.items():
        out[f"pred__{name}"] = model.predict(frame).astype(float)
        if model.supports_quantiles:
            q = model.predict_quantiles(frame, list(quantiles))
            for column in q.columns:
                out[f"{_quantile_label(float(column))}__{name}"] = q[column].to_numpy(float)
    return out


def _score(
    predictions: pd.DataFrame,
    models: Mapping[str, Forecaster],
    quantiles: tuple[float, ...],
    interval: tuple[float, float],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metrics: list[dict[str, Any]] = []
    per_building: list[dict[str, Any]] = []
    macro: list[dict[str, Any]] = []
    low, high = (_quantile_label(q) for q in interval)

    for scenario, frame in predictions.groupby("scenario", sort=False):
        point_cols = [f"pred__{name}" for name in models]
        common = frame["target_kwh"].notna() & frame[point_cols].notna().all(axis=1)
        rows = frame[common]
        for name, model in models.items():
            pred = rows[f"pred__{name}"]
            record: dict[str, Any] = {
                "scenario": scenario,
                "model": name,
                "eligible_rows": len(frame),
                "row_coverage": len(rows) / len(frame) if len(frame) else float("nan"),
                **evaluate_point(rows["target_kwh"], pred),
                "pinball": float("nan"),
                "coverage_80": float("nan"),
                "interval_width": float("nan"),
            }
            if model.supports_quantiles:
                q_cols = {q: f"{_quantile_label(q)}__{name}" for q in quantiles}
                q_frame = pd.DataFrame({q: rows[col] for q, col in q_cols.items()})
                record["pinball"] = mean_pinball_loss(rows["target_kwh"], q_frame)
                record["coverage_80"] = interval_coverage(
                    rows["target_kwh"], rows[f"{low}__{name}"], rows[f"{high}__{name}"]
                )
                record["interval_width"] = interval_width(
                    rows[f"{low}__{name}"], rows[f"{high}__{name}"]
                )
            metrics.append(record)

            buildings = []
            for building, group in rows.groupby("building_id"):
                row = {"scenario": scenario, "model": name, "building_id": building}
                row.update(evaluate_point(group["target_kwh"], group[f"pred__{name}"]))
                buildings.append(row)
            per_building.extend(buildings)
            table = pd.DataFrame(buildings)
            macro.append(
                {
                    "scenario": scenario,
                    "model": name,
                    "buildings": len(table),
                    "mae_mean": table["mae"].mean() if len(table) else float("nan"),
                    "mae_median": table["mae"].median() if len(table) else float("nan"),
                    "cv_rmse_mean": table["cv_rmse"].mean() if len(table) else float("nan"),
                    "cv_rmse_median": table["cv_rmse"].median() if len(table) else float("nan"),
                    "nmbe_mean": table["nmbe"].mean() if len(table) else float("nan"),
                }
            )
    return pd.DataFrame(metrics), pd.DataFrame(per_building), pd.DataFrame(macro)


def _skill_table(metrics: pd.DataFrame, macro: pd.DataFrame, baselines: list[str]) -> pd.DataFrame:
    """Skill of every non-baseline model against the best baseline of each scenario."""
    rows: list[dict[str, Any]] = []
    for scenario, group in metrics.groupby("scenario", sort=False):
        indexed = group.set_index("model")
        base = [b for b in baselines if b in indexed.index]
        if not base:
            continue
        best = indexed.loc[base, "mae"].idxmin()
        macro_rows = macro[macro["scenario"] == scenario].set_index("model")
        for model in indexed.index:
            if model in baselines:
                continue
            rows.append(
                {
                    "scenario": scenario,
                    "model": model,
                    "vs": best,
                    "mae_skill": skill_score(indexed.loc[model, "mae"], indexed.loc[best, "mae"]),
                    "rmse_skill": skill_score(
                        indexed.loc[model, "rmse"], indexed.loc[best, "rmse"]
                    ),
                    "cv_rmse_skill": skill_score(
                        indexed.loc[model, "cv_rmse"], indexed.loc[best, "cv_rmse"]
                    ),
                    "median_cv_rmse_skill": skill_score(
                        macro_rows.loc[model, "cv_rmse_median"],
                        macro_rows.loc[best, "cv_rmse_median"],
                    ),
                }
            )
    return pd.DataFrame(rows)


def _gate(
    metrics: pd.DataFrame,
    macro: pd.DataFrame,
    baselines: list[str],
    challenger: str,
    scenario: str,
) -> dict[str, Any]:
    """The challenger must beat the best baseline on MAE, RMSE and the median building CV(RMSE).

    The MEDIAN over buildings is used, not the mean: a building with a tiny mean load has an
    enormous CV(RMSE) and would dominate the mean.
    """
    rows = metrics[metrics["scenario"] == scenario].set_index("model")
    best = rows.loc[[b for b in baselines if b in rows.index], "mae"].idxmin()
    macro_rows = macro[macro["scenario"] == scenario].set_index("model")
    values = {
        "mae": (rows.loc[challenger, "mae"], rows.loc[best, "mae"]),
        "rmse": (rows.loc[challenger, "rmse"], rows.loc[best, "rmse"]),
        "median_cv_rmse": (
            macro_rows.loc[challenger, "cv_rmse_median"],
            macro_rows.loc[best, "cv_rmse_median"],
        ),
    }
    checks = {
        name: {
            "challenger": float(c),
            "baseline": float(b),
            "improvement": float(b - c),
            "relative": float(1 - c / b) if b else float("nan"),
        }
        for name, (c, b) in values.items()
    }
    passed = all(np.isfinite(v["improvement"]) and v["improvement"] > 0 for v in checks.values())
    parts = ", ".join(f"{k} {v['relative'] * 100:+.1f}%" for k, v in checks.items())
    verdict = "beats" if passed else "does not beat"
    return {
        "scenario": scenario,
        "challenger": challenger,
        "best_baseline": best,
        "passed": bool(passed),
        "checks": checks,
        "message": f"{challenger} {verdict} {best} on {scenario} (error reduction: {parts})",
    }


def run_backtest(
    features: pd.DataFrame,
    spec: SplitSpec,
    models: Mapping[str, Forecaster],
    *,
    challenger: str,
    baselines: list[str],
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    interval: tuple[float, float] = (0.1, 0.9),
    primary: str = "test_both",
) -> BacktestResult:
    """Fit all models on the training scenario and score them on the held-out scenarios."""
    if challenger not in models:
        raise ValueError(f"challenger '{challenger}' is not among the models: {list(models)}")
    buildings = features[["building_id", "site_id"]].drop_duplicates("building_id")
    roles = assign_roles(buildings, spec)
    parts = partition(features, roles, spec)

    for name in ("train", primary):
        if parts[name].empty:
            raise ValueError(f"scenario '{name}' has no rows: check the split dates and the data")
    for scenario in EVAL_SCENARIOS:
        log.info("scenario %-13s %8d rows", scenario, len(parts[scenario]))

    for name, model in models.items():
        log.info("fitting %s on %d rows", name, len(parts["train"]))
        model.fit(parts["train"], parts["val"] if len(parts["val"]) else None)

    frames = []
    for scenario in EVAL_SCENARIOS:
        if parts[scenario].empty:
            continue
        frame = _predict_scenario(parts[scenario], models, quantiles)
        frame.insert(0, "scenario", scenario)
        frames.append(frame)
    predictions = pd.concat(frames, ignore_index=True)

    metrics, per_building, macro = _score(predictions, models, quantiles, interval)
    skill = _skill_table(metrics, macro, baselines)
    gate = _gate(metrics, macro, baselines, challenger, primary)
    log.info(gate["message"])
    return BacktestResult(
        spec, roles, predictions, metrics, per_building, macro, skill, gate, dict(models)
    )
