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
