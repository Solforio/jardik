"""A forecast run: load an extraction, build features, backtest the model ladder, write results.

Outputs go to a `<run_id>.partial` folder and are moved into place only when everything,
including the report and the manifest, succeeded. The data used is identified by the SHA-256 of
the extraction's merged dataset, which is verified before use.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import lightgbm
import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from jardik import __version__
from jardik.core.catalogue import Catalogue
from jardik.core.logging import get_logger
from jardik.core.paths import JardikPaths
from jardik.dataset_manager.export.manifest import file_entry
from jardik.dataset_manager.export.writer import sha256_file, write_json, write_report_csv
from jardik.dataset_manager.request import ExportRequest
from jardik.forecasting import figures
from jardik.forecasting.backtest import BacktestResult, run_backtest
from jardik.forecasting.baselines import default_baselines
from jardik.forecasting.features import FeatureConfig, build_features, model_feature_columns
from jardik.forecasting.lightgbm_model import LGBMParams, LightGBMForecaster
from jardik.forecasting.report import build_forecast_pdf
from jardik.forecasting.splits import SCENARIOS, SplitSpec

log = get_logger("forecasting.run")

CHALLENGER = "lightgbm"


class ExtractionNotFoundError(RuntimeError):
    """The extraction is missing, incomplete or was modified after it was written."""


class RunExistsError(RuntimeError):
    """The run folder already exists; pass force to replace it."""


class ForecastRunConfig(BaseModel):
    """Everything that shapes a run. Its hash identifies the run."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[A-Za-z0-9_\-]+$")
    extraction_id: str
    split: SplitSpec
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    lightgbm: LGBMParams = Field(default_factory=LGBMParams)
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    seed: int = 42
    csv_style: Literal["spanish", "plain"] = "spanish"
    report_pdf: bool = True

    @field_validator("quantiles")
    @classmethod
    def _check_quantiles(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if any(not 0 < q < 1 for q in value):
            raise ValueError("quantiles must be strictly between 0 and 1")
        if 0.1 not in value or 0.9 not in value:
            raise ValueError("quantiles must include 0.1 and 0.9 (the 80 % interval)")
        return tuple(sorted(value))

    def config_hash(self) -> str:
        text = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    @property
    def run_id(self) -> str:
        return f"{self.name}-{self.config_hash()}"

    @classmethod
    def from_yaml(cls, path: Path) -> ForecastRunConfig:
        return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})


@dataclass
class ForecastRunResult:
    run_id: str
    output_dir: Path
    manifest: dict[str, Any]
    backtest: BacktestResult


# --- input --------------------------------------------------------------------------------------


def load_extraction(
    paths: JardikPaths, extraction_id: str
) -> tuple[pd.DataFrame, dict[str, Any], str]:
    """Read the merged dataset of an extraction after verifying its checksum."""
    folder = paths.extraction_dir(extraction_id)
    manifest_path = folder / "extraction_manifest.json"
    if not manifest_path.is_file():
        raise ExtractionNotFoundError(
            f"Extraction '{extraction_id}' not found in {folder}. "
            "Run `jardik dataset extractions` to see what is available."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = [f for f in manifest["files"] if f["name"].startswith("merged_training_dataset")]
    if not entries:
        raise ExtractionNotFoundError(
            f"Extraction '{extraction_id}' has no merged training dataset"
        )
    entry = entries[0]
    path = folder / entry["name"]
    if not path.is_file() or sha256_file(path) != entry["sha256"]:
        raise ExtractionNotFoundError(
            f"{path.name} of extraction '{extraction_id}' changed since it was extracted "
            "(checksum differs) or is missing. Extract it again."
        )

    export = manifest["export"]
    if export["format"] == "parquet":
        merged = pd.read_parquet(path)
    else:
        merged = pd.read_csv(path, sep=export["sep"], decimal=export["decimal"])
        merged["ts_utc"] = pd.to_datetime(merged["ts_utc"], utc=True)
        merged["ts_local"] = pd.to_datetime(merged["ts_local"])
    return merged, manifest, entry["sha256"]


# --- output -------------------------------------------------------------------------------------


def _models(config: ForecastRunConfig) -> dict[str, Any]:
    models: dict[str, Any] = dict(default_baselines())
    models[CHALLENGER] = LightGBMForecaster(
        model_feature_columns(config.features),
        config.lightgbm,
        config.quantiles,
        config.seed,
        name=CHALLENGER,
    )
    return models


def _limitations(extraction_manifest: dict[str, Any], config: ForecastRunConfig) -> list[str]:
    notes = [
        "The weather in the data is OBSERVED and stands in for the weather forecast. Real "
        "day-ahead "
        "errors will be larger, because an actual forecast has its own error: the results are "
        "optimistic about that part.",
        "Forecasts are issued at local midnight for the coming day. Imputed values are never "
        "used as "
        "features, to avoid information from after the origin.",
        f"One assignment of buildings to roles (seed {config.split.seed}) and one time split: the "
        "variability of the result across other assignments is not measured. Hyperparameters are "
        "defaults, not tuned.",
        "The model is trained on public data with the licence shown in this report (see "
        "LICENSES.md): whether trained weights inherit its conditions is unresolved.",
    ]
    if "Genome" in extraction_manifest["dataset"]["name"]:
        notes.append(
            "Building Data Genome 2 is mostly university and public buildings in the USA, the UK "
            "and Ireland: transfer to Spanish commercial offices is not demonstrated."
        )
    return notes


def _publish(work: Path, final: Path, force: bool) -> None:
    """Move the finished run folder into place; replace a previous one only when forced."""
    backup = final.with_name(final.name + ".old")
    if final.exists():
        if not force:
            raise RunExistsError(f"{final} already exists. Use force to replace it.")
        shutil.rmtree(backup, ignore_errors=True)
        final.rename(backup)
    try:
        work.rename(final)
    except BaseException:
        if backup.exists():
            backup.rename(final)
        raise
    shutil.rmtree(backup, ignore_errors=True)


def _write_outputs(
    config: ForecastRunConfig,
    result: BacktestResult,
    extraction_manifest: dict[str, Any],
    merged_sha: str,
    work: Path,
    final: Path,
    created_at: str,
) -> dict[str, Any]:
    export = ExportRequest(csv_style=config.csv_style)
    result.predictions.to_parquet(work / "predictions.parquet", index=False)
    for name, frame in (
        ("metrics", result.metrics),
        ("macro_metrics", result.macro),
        ("per_building_metrics", result.per_building),
        ("skill", result.skill),
    ):
        write_report_csv(frame, work / f"{name}.csv", export)
    challenger = result.models[CHALLENGER]
    importance = challenger.feature_importance()  # type: ignore[attr-defined]
    write_report_csv(importance, work / "feature_importance.csv", export)
    roles = result.roles.reset_index()
    write_report_csv(roles, work / "split_roles.csv", export)
    write_json(result.gate, work / "gate.json")

    model_dirs: list[dict[str, str]] = []
    challenger.save(work / "models" / CHALLENGER)  # type: ignore[attr-defined]
    model_dirs.append({"name": CHALLENGER, "kind": "lightgbm", "path": f"models/{CHALLENGER}"})

    gate = result.gate
    fig_dir = work / "figures"
    figures.metric_bars(result.metrics, fig_dir / "metric_bars.png", CHALLENGER)
    example = figures.example_week(
        result.predictions,
        fig_dir / "example_week.png",
        CHALLENGER,
        gate["best_baseline"],
        gate["scenario"],
    )
    figures.error_by_hour(
        result.predictions,
        fig_dir / "error_by_hour.png",
        CHALLENGER,
        gate["best_baseline"],
        gate["scenario"],
    )
    figures.building_scatter(
        result.per_building,
        fig_dir / "building_scatter.png",
        CHALLENGER,
        gate["best_baseline"],
        gate["scenario"],
    )
    figures.feature_importance(importance, fig_dir / "feature_importance.png")

    request = extraction_manifest["request"]
    scenario_rows = result.predictions["scenario"].value_counts().to_dict()
    train_rows = int(result.models[CHALLENGER].n_train_rows)  # type: ignore[attr-defined]
    manifest: dict[str, Any] = {
        "run_id": config.run_id,
        "kind": "forecast_backtest",
        "name": config.name,
        "created_at": created_at,
        "extraction_id": extraction_manifest["extraction_id"],
        "config": config.model_dump(mode="json"),
        "config_hash": config.config_hash(),
        "seed": config.seed,
        "extraction": {
            "id": extraction_manifest["extraction_id"],
            "dataset": extraction_manifest["dataset"]["name"],
            "licence": extraction_manifest["dataset"]["licence"],
            "citation": extraction_manifest["dataset"].get("citation", ""),
            "attribution": extraction_manifest["attribution"],
            "period": [str(request["start"]), str(request["end"])],
            "merged_sha256": merged_sha,
            "buildings": extraction_manifest["counts"]["buildings"],
        },
        "split": {
            "spec": config.split.model_dump(mode="json"),
            "roles": result.roles.value_counts().to_dict(),
            "scenarios": {
                **{s: int(scenario_rows.get(s, 0)) for s in SCENARIOS},
                "train": train_rows,
            },
        },
        "features": {
            "config": config.features.model_dump(mode="json"),
            "model_features": model_feature_columns(config.features),
        },
        "models": model_dirs,
        "models_fitted": list(result.models),
        "baselines": [m for m in result.models if m != CHALLENGER],
        "gate": gate,
        "metrics": result.metrics.to_dict("records"),
        "figures": {"example_week": example},
        "environment": {
            "jardik": __version__,
            "lightgbm": lightgbm.__version__,
            "pandas": pd.__version__,
        },
        "limitations": _limitations(extraction_manifest, config),
        "output_dir": str(final),
    }
    if config.report_pdf:
        build_forecast_pdf(work / "forecast_report.pdf", manifest, result, fig_dir, importance)
    manifest["files"] = [
        file_entry(path, work) for path in sorted(p for p in work.rglob("*") if p.is_file())
    ]
    write_json(manifest, work / "run_manifest.json")
    return manifest


def run_forecast(
    config: ForecastRunConfig,
    paths: JardikPaths,
    catalogue: Catalogue | None = None,
    *,
    force: bool = False,
    now: Callable[[], datetime] | None = None,
) -> ForecastRunResult:
    """Run the backtest of the model ladder on an extraction and write everything to disk."""
    clock = now or (lambda: datetime.now(UTC))
    final = paths.run_dir(config.run_id)
    if final.exists() and not force:
        raise RunExistsError(f"{final} already exists. Use force to replace it.")

    merged, extraction_manifest, merged_sha = load_extraction(paths, config.extraction_id)
    log.info("run %s on extraction %s (%d rows)", config.run_id, config.extraction_id, len(merged))
    features = build_features(merged, config.features)
    del merged
    models = _models(config)
    result = run_backtest(
        features,
        config.split,
        models,
        challenger=CHALLENGER,
        baselines=[m for m in models if m != CHALLENGER],
        quantiles=config.quantiles,
    )

    work = final.with_name(final.name + ".partial")
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        manifest = _write_outputs(
            config, result, extraction_manifest, merged_sha, work, final, clock().isoformat()
        )
        _publish(work, final, force)
    except BaseException:
        shutil.rmtree(work, ignore_errors=True)
        raise

    if catalogue is not None:
        catalogue.register_run(manifest)
    log.info("run %s done: %s", config.run_id, result.gate["message"])
    return ForecastRunResult(config.run_id, final, manifest, result)
