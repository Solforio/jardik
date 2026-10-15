from pathlib import Path

import pytest

from jardik.core.catalogue import Catalogue

INFO = {
    "dataset_id": "bdg2",
    "name": "Building Data Genome 2",
    "version": "v1.0",
    "licence_id": "bdg2",
    "licence": "CC BY-SA 4.0",
    "source_url": "https://github.com/buds-lab/building-data-genome-project-2",
    "native_granularity_s": 3600,
}
FILES = [
    {
        "name": "metadata.csv",
        "path": "data/raw/bdg2/metadata.csv",
        "sha256": "ab" * 32,
        "size_bytes": 10,
    }
]


def _manifest(extraction_id="offices-abc", n=2) -> dict:
    return {
        "extraction_id": extraction_id,
        "name": "offices",
        "dataset_id": "bdg2",
        "request_hash": "abc",
        "created_at": "2026-10-16T10:00:00+00:00",
        "jardik_version": "0.1.0",
        "output_dir": "outputs/extractions/" + extraction_id,
        "request": {"name": "offices"},
        "counts": {"buildings": n, "load_rows": 100},
        "selected_buildings": [
            {"building_id": f"bdg2:b{i}", "site_id": "S", "usage_category": "office", "rank": i}
            for i in range(n)
        ],
    }


def test_tables_are_created_and_empty(tmp_path: Path):
    with Catalogue(tmp_path / "cat.duckdb") as cat:
        assert cat.list_datasets() == [] and cat.list_extractions() == []


def test_register_and_read_dataset(tmp_path: Path):
    with Catalogue(tmp_path / "cat.duckdb") as cat:
        cat.register_dataset(INFO, FILES)
        [ds] = cat.list_datasets()
        assert ds["dataset_id"] == "bdg2" and ds["licence_id"] == "bdg2"
        assert cat.get_dataset("bdg2")["info"]["licence"] == "CC BY-SA 4.0"
        assert cat.dataset_files("bdg2")[0]["sha256"] == "ab" * 32
        assert cat.get_dataset("nope") is None


def test_reregistering_a_dataset_replaces_its_files(tmp_path: Path):
    with Catalogue(tmp_path / "cat.duckdb") as cat:
        cat.register_dataset(INFO, FILES)
        cat.register_dataset(
            INFO, [{"name": "weather.csv", "path": "w", "sha256": "cd" * 32, "size_bytes": 5}]
        )
        assert [f["name"] for f in cat.dataset_files("bdg2")] == ["weather.csv"]
        assert len(cat.list_datasets()) == 1


def test_register_extraction_and_its_buildings(tmp_path: Path):
    with Catalogue(tmp_path / "cat.duckdb") as cat:
        cat.register_extraction(_manifest(n=3))
        [ext] = cat.list_extractions()
        assert (
            ext["extraction_id"] == "offices-abc"
            and ext["n_buildings"] == 3
            and ext["n_rows"] == 100
        )
        assert [b["building_id"] for b in cat.extraction_buildings("offices-abc")] == [
            "bdg2:b0",
            "bdg2:b1",
            "bdg2:b2",
        ]
        assert cat.get_extraction("offices-abc")["request"]["name"] == "offices"


def test_rerunning_an_extraction_replaces_it(tmp_path: Path):
    with Catalogue(tmp_path / "cat.duckdb") as cat:
        cat.register_extraction(_manifest(n=3))
        cat.register_extraction(_manifest(n=1))
        assert len(cat.list_extractions()) == 1
        assert len(cat.extraction_buildings("offices-abc")) == 1


def test_data_persists_across_connections(tmp_path: Path):
    path = tmp_path / "cat.duckdb"
    with Catalogue(path) as cat:
        cat.register_extraction(_manifest())
    with Catalogue(path) as cat:
        assert cat.get_extraction("offices-abc") is not None


def test_list_is_newest_first(tmp_path: Path):
    with Catalogue(tmp_path / "cat.duckdb") as cat:
        old = _manifest("old-1") | {"created_at": "2026-01-01T00:00:00+00:00"}
        new = _manifest("new-1") | {"created_at": "2026-06-01T00:00:00+00:00"}
        cat.register_extraction(old)
        cat.register_extraction(new)
        assert [e["extraction_id"] for e in cat.list_extractions()] == ["new-1", "old-1"]


def test_use_after_close_is_an_error(tmp_path: Path):
    cat = Catalogue(tmp_path / "cat.duckdb")
    cat.close()
    with pytest.raises(RuntimeError, match="closed"):
        cat.list_datasets()


def test_parent_folder_is_created(tmp_path: Path):
    with Catalogue(tmp_path / "deep" / "dir" / "cat.duckdb") as cat:
        assert cat.list_datasets() == []


# --- forecast runs ----------------------------------------------------------------------------


def _run_manifest(run_id="gbm-abc", passed=True, created="2026-10-16T10:00:00+00:00") -> dict:
    return {
        "run_id": run_id,
        "kind": "forecast_backtest",
        "name": "gbm",
        "created_at": created,
        "extraction_id": "offices-abc",
        "config_hash": "abc",
        "output_dir": "outputs/runs/" + run_id,
        "config": {"name": "gbm", "seed": 42},
        "gate": {"passed": passed, "challenger": "lightgbm", "best_baseline": "seasonal_naive_168"},
        "metrics": [{"scenario": "test_both", "model": "lightgbm", "mae": 1.5}],
        "models": [{"name": "lightgbm", "kind": "lightgbm", "path": "models/lightgbm"}],
    }


def test_register_and_read_a_run(tmp_path: Path):
    with Catalogue(tmp_path / "cat.duckdb") as cat:
        cat.register_run(_run_manifest())
        [run] = cat.list_runs()
        assert run["run_id"] == "gbm-abc" and run["gate_passed"] is True
        assert run["challenger"] == "lightgbm" and run["best_baseline"] == "seasonal_naive_168"
        full = cat.get_run("gbm-abc")
        assert full["config"]["seed"] == 42 and full["metrics"][0]["mae"] == 1.5
        assert [m["name"] for m in cat.run_models("gbm-abc")] == ["lightgbm"]
        assert cat.get_run("nope") is None


def test_rerunning_replaces_the_run_and_its_models(tmp_path: Path):
    with Catalogue(tmp_path / "cat.duckdb") as cat:
        cat.register_run(_run_manifest(passed=True))
        second = _run_manifest(passed=False) | {"models": []}
        cat.register_run(second)
        assert len(cat.list_runs()) == 1
        assert cat.list_runs()[0]["gate_passed"] is False
        assert cat.run_models("gbm-abc") == []


def test_runs_are_listed_newest_first_and_filtered_by_extraction(tmp_path: Path):
    with Catalogue(tmp_path / "cat.duckdb") as cat:
        cat.register_run(_run_manifest("old-1", created="2026-01-01T00:00:00+00:00"))
        cat.register_run(_run_manifest("new-1", created="2026-06-01T00:00:00+00:00"))
        other = _run_manifest("other-1") | {"extraction_id": "other-extraction"}
        cat.register_run(other)
        assert [r["run_id"] for r in cat.list_runs()] == ["other-1", "new-1", "old-1"]
        assert {r["run_id"] for r in cat.list_runs(extraction_id="offices-abc")} == {
            "old-1",
            "new-1",
        }
