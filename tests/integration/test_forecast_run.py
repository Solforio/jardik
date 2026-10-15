"""A full forecast run from an extraction folder: backtest, outputs, report, catalogue."""

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError
from pypdf import PdfReader

from fake_extraction import write_fake_extraction
from jardik.core.catalogue import Catalogue
from jardik.core.paths import JardikPaths
from jardik.forecasting.features import FeatureConfig, build_features
from jardik.forecasting.lightgbm_model import LGBMParams, LightGBMForecaster
from jardik.forecasting.run import (
    ExtractionNotFoundError,
    ForecastRunConfig,
    RunExistsError,
    run_forecast,
)
from jardik.forecasting.splits import SplitSpec
from synthetic import make_merged_frame

NOW = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
SPLIT = SplitSpec(train_end=date(2024, 3, 15), val_end=date(2024, 4, 10), seed=7)
FAST = LGBMParams(n_estimators=100, num_leaves=15, min_child_samples=20, early_stopping_rounds=15)


@pytest.fixture(scope="module")
def merged() -> pd.DataFrame:
    return make_merged_frame(n_buildings=16, start="2024-01-01", days=150, seed=11)


@pytest.fixture
def paths(tmp_path: Path, merged) -> JardikPaths:
    project = JardikPaths.from_root(tmp_path / "project")
    write_fake_extraction(project, merged)
    return project


def _config(**overrides) -> ForecastRunConfig:
    base = {"name": "gbm", "extraction_id": "offices-test", "split": SPLIT, "lightgbm": FAST}
    base.update(overrides)
    return ForecastRunConfig(**base)


def _run(paths, config=None, **kwargs):
    return run_forecast(config or _config(), paths, now=lambda: NOW, **kwargs)


def _boom(*_args, **_kwargs):
    raise OSError("simulated failure while writing the report")


# --- configuration ---


def test_run_id_is_name_plus_hash_of_everything_that_matters():
    a, b = _config(), _config(seed=1)
    assert a.run_id == f"gbm-{a.config_hash()}" and a.run_id != b.run_id
    assert a.config_hash() == _config().config_hash()


def test_invalid_configurations_are_rejected():
    with pytest.raises(ValidationError):
        _config(name="has space")
    with pytest.raises(ValidationError):
        _config(quantiles=(0.5, 0.9))  # the 80 % interval needs 0.1 and 0.9
    with pytest.raises(ValidationError):
        _config(quantiles=(0.1, 0.9, 1.2))


def test_config_from_yaml(tmp_path):
    path = tmp_path / "run.yaml"
    path.write_text(
        "name: from_yaml\nextraction_id: offices-test\n"
        "split: {train_end: 2024-03-15, val_end: 2024-04-10}\n"
        "lightgbm: {num_leaves: 15}\n",
        encoding="utf-8",
    )
    config = ForecastRunConfig.from_yaml(path)
    assert config.split.val_end == date(2024, 4, 10) and config.lightgbm.num_leaves == 15


# --- outputs ---


def test_run_writes_every_expected_output(paths):
    result = _run(paths)
    names = {p.name for p in result.output_dir.iterdir()}
    expected = {
        "predictions.parquet",
        "metrics.csv",
        "macro_metrics.csv",
        "per_building_metrics.csv",
        "skill.csv",
        "feature_importance.csv",
        "split_roles.csv",
        "gate.json",
        "run_manifest.json",
        "forecast_report.pdf",
        "figures",
        "models",
    }
    assert expected <= names
    figures = {p.name for p in (result.output_dir / "figures").iterdir()}
    assert {
        "metric_bars.png",
        "example_week.png",
        "error_by_hour.png",
        "building_scatter.png",
        "feature_importance.png",
    } <= figures
    assert result.output_dir == paths.run_dir(result.run_id)


def test_the_gate_passes_and_is_recorded_identically_everywhere(paths):
    result = _run(paths)
    gate = json.loads((result.output_dir / "gate.json").read_text())
    assert gate["passed"] is True and gate == result.manifest["gate"]
    assert gate["best_baseline"] in result.manifest["baselines"]


def test_manifest_ties_the_run_to_its_data_and_configuration(paths):
    result = _run(paths)
    m = json.loads((result.output_dir / "run_manifest.json").read_text())
    extraction_manifest = paths.extraction_dir("offices-test") / "extraction_manifest.json"
    extraction = json.loads(extraction_manifest.read_text())
    assert m["run_id"] == _config().run_id and m["config_hash"] == _config().config_hash()
    assert m["created_at"] == NOW.isoformat()
    assert m["extraction"]["id"] == "offices-test"
    assert m["extraction"]["merged_sha256"] == extraction["files"][0]["sha256"]
    assert m["extraction"]["licence"].startswith("CC BY-SA 4.0")
    assert m["split"]["scenarios"]["test_both"] > 0 and m["split"]["roles"]["test"] > 0
    assert m["environment"]["lightgbm"] and m["seed"] == 42
    assert m["limitations"] and any("forecast" in x.lower() for x in m["limitations"])
    for entry in m["files"]:
        data = (result.output_dir / entry["name"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["name"]
    assert "run_manifest.json" not in {e["name"] for e in m["files"]}


def test_saved_model_reproduces_the_saved_predictions(paths, merged):
    result = _run(paths)
    model = LightGBMForecaster.load(result.output_dir / "models" / "lightgbm")
    predictions = pd.read_parquet(result.output_dir / "predictions.parquet")
    both = predictions[predictions["scenario"] == "test_both"].head(400)
    features = build_features(merged, FeatureConfig())
    rows = features.merge(both[["building_id", "ts_utc"]], on=["building_id", "ts_utc"])
    saved = both.merge(rows[["building_id", "ts_utc"]], on=["building_id", "ts_utc"])
    assert model.predict(rows).to_numpy() == pytest.approx(
        saved["pred__lightgbm"].to_numpy(), rel=1e-9
    )


def test_metric_tables_use_the_spanish_convention(paths):
    result = _run(paths)
    metrics = pd.read_csv(result.output_dir / "metrics.csv", sep=";", decimal=",")
    assert {"scenario", "model", "mae", "rmse", "cv_rmse", "nmbe", "n"} <= set(metrics.columns)
    assert metrics["mae"].dtype.kind == "f"
    raw = (result.output_dir / "metrics.csv").read_text().splitlines()[1]
    assert ";" in raw and "," in raw


def test_plain_csv_style(paths):
    result = _run(paths, _config(csv_style="plain", name="plain"))
    metrics = pd.read_csv(result.output_dir / "metrics.csv")
    assert metrics["mae"].dtype.kind == "f"


def test_pdf_report_states_the_verdict_and_the_definitions(paths):
    result = _run(paths)
    reader = PdfReader(result.output_dir / "forecast_report.pdf")
    text = "\n".join(page.extract_text() for page in reader.pages)
    for needle in (
        result.run_id,
        "Forecast backtest report",
        "PASSED",
        "Scenarios",
        "Skill",
        "Probabilistic",
        "Limitations",
        "ASHRAE",
        "Synthetic offices",
        "CC BY-SA 4.0",
    ):
        assert needle in text, needle


def test_run_is_registered_in_the_catalogue(paths):
    with Catalogue(paths.catalogue_path) as catalogue:
        result = _run(paths, catalogue=catalogue)
        [entry] = catalogue.list_runs()
        assert entry["run_id"] == result.run_id and entry["gate_passed"] is True
        assert entry["extraction_id"] == "offices-test"
        assert {m["name"] for m in catalogue.run_models(result.run_id)} == {"lightgbm"}


def test_same_configuration_gives_identical_predictions(paths):
    first = _run(paths)
    before = pd.read_parquet(first.output_dir / "predictions.parquet")
    second = _run(paths, force=True)
    after = pd.read_parquet(second.output_dir / "predictions.parquet")
    pd.testing.assert_frame_equal(before, after)


# --- safety ---


def test_rerun_requires_force(paths):
    _run(paths)
    with pytest.raises(RunExistsError):
        _run(paths)
    _run(paths, force=True)


def test_a_crash_leaves_nothing_and_a_failed_force_keeps_the_old_run(paths, monkeypatch):
    import jardik.forecasting.run as run_module

    monkeypatch.setattr(run_module, "build_forecast_pdf", _boom)
    with pytest.raises(OSError):
        _run(paths)
    assert not paths.run_dir(_config().run_id).exists()
    assert not list(paths.runs_dir.glob("*.partial"))
    monkeypatch.undo()

    _run(paths)
    manifest = paths.run_dir(_config().run_id) / "run_manifest.json"
    before = manifest.read_bytes()
    monkeypatch.setattr(run_module, "build_forecast_pdf", _boom)
    with pytest.raises(OSError):
        _run(paths, force=True)
    assert manifest.read_bytes() == before


def test_a_modified_extraction_is_refused(paths):
    folder = paths.extraction_dir("offices-test")
    with (folder / "merged_training_dataset.parquet").open("ab") as handle:
        handle.write(b"tampered")
    with pytest.raises(ExtractionNotFoundError, match="changed since"):
        _run(paths)


def test_unknown_extraction_gives_a_helpful_error(tmp_path):
    empty = JardikPaths.from_root(tmp_path / "empty")
    with pytest.raises(ExtractionNotFoundError, match="dataset extractions"):
        run_forecast(_config(), empty)


def test_csv_extractions_can_be_used_too(tmp_path, merged):
    project = JardikPaths.from_root(tmp_path / "p")
    write_fake_extraction(project, merged, "offices-csv", fmt="csv")
    result = run_forecast(_config(extraction_id="offices-csv"), project, now=lambda: NOW)
    assert result.manifest["gate"]["passed"] is True
