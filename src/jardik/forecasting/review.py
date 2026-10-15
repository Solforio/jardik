"""A readable copy of a forecast run: Excel workbook, PDF, figures and the JSON files."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd
from openpyxl.styles import Alignment, Font

from jardik.dataset_manager.export.review import _excel_safe, _style, _style_header_only
from jardik.dataset_manager.export.writer import PLAIN_CSV, SPANISH_CSV


def _read_csv(path: Path, csv_style: str) -> pd.DataFrame:
    dialect = SPANISH_CSV if csv_style == "spanish" else PLAIN_CSV
    return pd.read_csv(path, sep=dialect["sep"], decimal=dialect["decimal"])


def _example_week(run_dir: Path, manifest: dict[str, Any]) -> pd.DataFrame:
    """The predictions of the week shown in the report's example figure."""
    info = manifest["figures"]["example_week"]
    predictions = pd.read_parquet(run_dir / "predictions.parquet")
    start = pd.Timestamp(info["week_start"])
    rows = predictions[
        (predictions["scenario"] == manifest["gate"]["scenario"])
        & (predictions["building_id"] == info["building_id"])
        & (predictions["ts_utc"] >= start)
        & (predictions["ts_utc"] < start + pd.Timedelta(days=7))
    ]
    return rows.sort_values("ts_utc").reset_index(drop=True)


def build_run_review(run_dir: Path, dest_root: Path) -> Path:
    """Create `<dest_root>/<run_id>/` with the readable files and return it."""
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    style = manifest["config"]["csv_style"]
    dest = dest_root / manifest["run_id"]
    dest.mkdir(parents=True, exist_ok=True)

    sheets: list[tuple[str, pd.DataFrame, str]] = [
        (
            "Metrics",
            _read_csv(run_dir / "metrics.csv", style),
            "Pooled error of every model in every scenario, on identical rows.",
        ),
        (
            "Macro by model",
            _read_csv(run_dir / "macro_metrics.csv", style),
            "Building-by-building averages: every building counts equally.",
        ),
        (
            "Skill",
            _read_csv(run_dir / "skill.csv", style),
            "Improvement over the best baseline of each scenario.",
        ),
        (
            "Per building",
            _read_csv(run_dir / "per_building_metrics.csv", style),
            "Error of every model for every held-out building.",
        ),
        (
            "Feature importance",
            _read_csv(run_dir / "feature_importance.csv", style),
            "Total gain of each feature in the mean model.",
        ),
        (
            "Split roles",
            _read_csv(run_dir / "split_roles.csv", style),
            "Which buildings were used for training, validation and testing.",
        ),
        (
            "Example week",
            _example_week(run_dir, manifest),
            "Actual load and forecasts of one test building for one week.",
        ),
        (
            "Limitations",
            pd.DataFrame({"limitation": manifest["limitations"]}),
            "What these results do and do not show.",
        ),
    ]
    gate, extraction = manifest["gate"], manifest["extraction"]
    verdict = "PASSED" if gate["passed"] else "NOT PASSED"
    readme = pd.DataFrame(
        [
            ("Run", manifest["run_id"]),
            ("Created", manifest["created_at"]),
            ("Gate", f"{verdict}: {gate['message']}"),
            ("Data", f"{extraction['dataset']} ({extraction['id']})"),
            ("Licence", extraction["licence"]),
            ("Attribution", extraction["attribution"]),
            ("", ""),
            ("Sheets", ""),
            *[(name, description) for name, _, description in sheets],
            ("", ""),
            (
                "About errors",
                "Errors are in kWh per hour. NMBE follows ASHRAE Guideline 14: "
                "positive = under-forecast. Ratios are shown as percentages.",
            ),
        ],
        columns=["Item", "Value"],
    )

    with pd.ExcelWriter(dest / "forecast_tables.xlsx", engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="Read me", index=False)
        sheet = writer.book["Read me"]
        sheet.column_dimensions["A"].width = 22
        sheet.column_dimensions["B"].width = 110
        for row in sheet.iter_rows(min_row=2):
            row[1].alignment = Alignment(wrap_text=True, vertical="top")
            row[0].font = Font(bold=True)
        _style_header_only(sheet)
        for name, frame, _ in sheets:
            safe = _excel_safe(frame)
            safe.to_excel(writer, sheet_name=name, index=False)
            _style(writer.book[name], safe)

    if (run_dir / "forecast_report.pdf").is_file():
        shutil.copy2(run_dir / "forecast_report.pdf", dest / "forecast_report.pdf")
    shutil.copytree(run_dir / "figures", dest / "figures", dirs_exist_ok=True)
    json_dir = dest / "json"
    json_dir.mkdir(exist_ok=True)
    for name in ("run_manifest.json", "gate.json"):
        shutil.copy2(run_dir / name, json_dir / name)
    return dest
