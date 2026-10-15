"""Write just enough of an extraction folder for a forecast run (merged dataset + manifest)."""

import hashlib
import json

import pandas as pd


def write_fake_extraction(
    paths, merged: pd.DataFrame, extraction_id: str = "offices-test", fmt: str = "parquet"
):
    folder = paths.extraction_dir(extraction_id)
    folder.mkdir(parents=True, exist_ok=True)
    if fmt == "parquet":
        file = folder / "merged_training_dataset.parquet"
        merged.to_parquet(file, index=False)
        export = {"format": "parquet"}
    else:
        file = folder / "merged_training_dataset.csv"
        merged.to_csv(file, index=False, sep=";", decimal=",")
        export = {"format": "csv", "csv_style": "spanish", "sep": ";", "decimal": ","}
    manifest = {
        "extraction_id": extraction_id,
        "name": "offices",
        "created_at": "2026-10-16T10:00:00+00:00",
        "request": {"start": "2024-01-01", "end": "2024-06-01", "task": "forecasting"},
        "dataset": {
            "name": "Synthetic offices",
            "licence": "CC BY-SA 4.0 (Attribution-ShareAlike): for tests",
            "citation": "Made up for the tests",
        },
        "attribution": "Synthetic data for tests.",
        "export": export,
        "counts": {"buildings": int(merged["building_id"].nunique()), "load_rows": len(merged)},
        "files": [
            {
                "name": file.name,
                "sha256": hashlib.sha256(file.read_bytes()).hexdigest(),
                "size_bytes": file.stat().st_size,
                "rows": len(merged),
            }
        ],
    }
    (folder / "extraction_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return folder
