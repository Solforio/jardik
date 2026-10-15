"""The extraction manifest: everything needed to understand and reproduce an extraction."""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jardik import __version__
from jardik.dataset_manager.export.merge import COLUMN_DOCS
from jardik.dataset_manager.export.writer import sha256_file
from jardik.dataset_manager.request import ExtractionRequest

ML_USE_CASES = {
    "forecasting": [
        "Day-ahead (24 h) hourly load forecasting, evaluated on held-out buildings and "
        "later periods.",
        "Model ladder benchmarks on identical splits: seasonal baselines, gradient boosting, "
        "deep models.",
        "Study of the value of weather covariates and of transfer between buildings and sites.",
        "Residual-based anomaly screening once a forecast model is trained.",
    ],
    "nilm": ["Appliance-level disaggregation (requires a dataset with appliance sub-metering)."],
    "both": [],
}


def build_limitations(
    request: ExtractionRequest,
    info: dict[str, Any],
    selected: pd.DataFrame,
    weather_screening: dict[str, dict[str, int]],
    corrections: list[dict[str, Any]],
    unmapped_usage: set[str],
) -> list[str]:
    notes: list[str] = list(info.get("notes", []))
    if info.get("coordinate_precision"):
        notes.append(f"Coordinates: {info['coordinate_precision']}.")
    notes.append(
        "Weather comes from the dataset and is observed data, not forecasts: a model trained here "
        "sees measurements, while production use will see forecasts (train/production mismatch)."
    )
    if selected["country"].isna().all():
        notes.append(
            "Country is unknown for every selected building, so holiday flags are not included."
        )
    if request.quality.imputation != "none":
        notes.append(
            f"Gaps of up to {request.quality.max_gap_hours} h were linearly interpolated "
            "and flagged "
            "(is_imputed); longer gaps are left empty."
        )
    for variable, rules in weather_screening.items():
        for rule, count in rules.items():
            notes.append(f"Weather rule applied to {count} value(s) of {variable}: {rule}.")
    for correction in corrections:
        notes.append(
            f"Declared correction: site {correction['site_id']} {correction['field']} "
            f"{correction['old']} -> {correction['new']} ({correction['reason']})."
        )
    if unmapped_usage:
        notes.append(
            "Usage labels without a mapping were assigned to 'other': "
            + ", ".join(sorted(unmapped_usage))
            + "."
        )
    notes.append(
        "Hourly slots outside the source file's coverage are reported as missing, not dropped."
    )
    return notes


def attribution(info: dict[str, Any]) -> str:
    return (
        f"Contains data from {info['name']} ({info['citation']}), licensed {info['licence']}. "
        "Adapted by Jardik: building selection, conversion of timestamps to UTC, unit and value "
        "rules, optional gap imputation and weather matching. Redistribution of this extraction "
        "must keep this attribution and the same licence."
    )


def column_schema(merged: pd.DataFrame) -> list[dict[str, str]]:
    return [
        {
            "name": c,
            "dtype": str(merged[c].dtype),
            "unit": COLUMN_DOCS.get(c, ("", ""))[0],
            "description": COLUMN_DOCS.get(c, ("", ""))[1],
        }
        for c in merged.columns
    ]


def file_entry(path: Path, directory: Path, rows: int | None = None) -> dict[str, Any]:
    return {
        "name": path.relative_to(directory).as_posix(),
        "sha256": sha256_file(path),
        "size_bytes": path.stat().st_size,
        "rows": rows,
    }


def environment() -> dict[str, str]:
    return {
        "jardik": __version__,
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "platform": platform.platform(),
    }
