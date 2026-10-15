"""Writing tables and JSON files to the extraction folder."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from jardik.dataset_manager.request import ExportRequest

SPANISH_CSV = {"sep": ";", "decimal": ","}
PLAIN_CSV = {"sep": ",", "decimal": "."}


def csv_dialect(export: ExportRequest) -> dict[str, str]:
    return SPANISH_CSV if export.csv_style == "spanish" else PLAIN_CSV


def write_table(df: pd.DataFrame, directory: Path, stem: str, export: ExportRequest) -> Path:
    """Write a table as Parquet (plain, for machines) or CSV (Spanish or plain convention)."""
    directory.mkdir(parents=True, exist_ok=True)
    if export.format == "parquet":
        path = directory / f"{stem}.parquet"
        df.to_parquet(path, index=False)
    else:
        path = directory / f"{stem}.csv"
        dialect = csv_dialect(export)
        df.to_csv(path, index=False, sep=dialect["sep"], decimal=dialect["decimal"])
    return path


def write_report_csv(df: pd.DataFrame, path: Path, export: ExportRequest) -> Path:
    """Human-facing CSV report (always CSV), following the request's CSV convention."""
    dialect = csv_dialect(export)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, sep=dialect["sep"], decimal=dialect["decimal"])
    return path


def to_jsonable(value: Any) -> Any:
    """Recursively convert numpy / pandas / date values to plain JSON types (NaN -> null)."""
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, pd.DataFrame):
        return to_jsonable(value.to_dict("records"))
    if isinstance(value, pd.Series):
        return to_jsonable(value.to_dict())
    if isinstance(value, (pd.Timestamp, datetime, date)):
        return None if value is pd.NaT else value.isoformat()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if math.isnan(number) or math.isinf(number) else number
    if isinstance(value, np.bool_):
        return bool(value)
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        to_jsonable(obj), indent=2, sort_keys=False, ensure_ascii=False, allow_nan=False
    )
    path.write_text(text + "\n", encoding="utf-8")
    return path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
