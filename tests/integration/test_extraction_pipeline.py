"""End-to-end Dataset Manager run on a miniature BDG2: selection, quality, matching, export."""

import hashlib
import json
import re
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest
from pypdf import PdfReader

from jardik.core.catalogue import Catalogue
from jardik.core.paths import JardikPaths
from jardik.core.schemas import validate_load, validate_weather_match
from jardik.dataset_manager.adapters.bdg2 import Bdg2Adapter, Bdg2Config, SiteOverride
from jardik.dataset_manager.pipeline import (
    ExtractionExistsError,
    ExtractionRequestError,
    NoBuildingsError,
    run_extraction,
)
from jardik.dataset_manager.request import ExtractionRequest

NOW = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)


@pytest.fixture
def adapter(mini_bdg2: Path) -> Bdg2Adapter:
    return Bdg2Adapter(
        Bdg2Config(
            data_dir=mini_bdg2,
            usage_map={"Office": "office", "Retail": "shop"},
            site_overrides={"Dub": SiteOverride(lng=-6.26, reason="longitude sign error (Dublin)")},
            bbox_by_timezone={
                "US/Eastern": {"min_lat": 24, "max_lat": 50, "min_lon": -85, "max_lon": -66},
                "Europe/Dublin": {"min_lat": 51, "max_lat": 56, "min_lon": -11, "max_lon": -5.5},
            },
        )
    )


@pytest.fixture
def paths(tmp_path: Path) -> JardikPaths:
    return JardikPaths.from_root(tmp_path / "project")


def _request(**overrides) -> ExtractionRequest:
    base = {
        "name": "offices",
        "start": date(2016, 3, 8),
        "end": date(2016, 3, 20),
        "usage_categories": ["office"],
        "n_buildings": 4,
        "seed": 1,
        "export": {"format": "parquet"},
    }
    base.update(overrides)
    return ExtractionRequest(**base)


def _run(request, adapter, paths, **kwargs):
    return run_extraction(request, adapter, paths, now=lambda: NOW, **kwargs)


def _read(result, name):
    return pd.read_parquet(result.output_dir / f"{name}.parquet")


# --- the happy path -----------------------------------------------------------------------------


def test_extraction_writes_every_expected_output(adapter, paths):
    result = _run(_request(), adapter, paths)
    expected = {
        "selected_buildings.parquet",
        "building_metadata.parquet",
        "load_timeseries.parquet",
        "weather_timeseries.parquet",
        "weather_match.parquet",
        "merged_training_dataset.parquet",
        "excluded_buildings.parquet",
        "debug_counts.csv",
        "distribution_report.json",
        "quality_report.json",
        "extraction_manifest.json",
        "extraction_report.pdf",
    }
    assert expected <= {p.name for p in result.output_dir.iterdir()}
    assert result.output_dir == paths.extraction_dir(result.extraction_id)


def test_selects_the_requested_number_of_valid_buildings_deterministically(adapter, paths):
    first = _run(_request(), adapter, paths)
    selected = _read(first, "selected_buildings")
    assert len(selected) == 4 and selected["building_id"].is_unique
    assert "bdg2:East_office_f" not in set(selected["building_id"])  # too much missing data
    again = _run(_request(), adapter, paths, force=True)
    pd.testing.assert_frame_equal(selected, _read(again, "selected_buildings"))
    different_seed = _run(_request(seed=99, name="offices_b"), adapter, paths)
    assert set(_read(different_seed, "selected_buildings")["building_id"]) != set(
        selected["building_id"]
    )


def test_load_and_weather_frames_obey_their_contracts(adapter, paths):
    result = _run(_request(), adapter, paths)
    validate_load(_read(result, "load_timeseries"))
    validate_weather_match(_read(result, "weather_match"))


def test_merged_dataset_has_target_weather_static_and_calendar_columns(adapter, paths):
    result = _run(_request(), adapter, paths)
    merged = _read(result, "merged_training_dataset")
    for column in (
        "building_id",
        "ts_utc",
        "ts_local",
        "target_kwh",
        "is_missing",
        "is_imputed",
        "temp_c",
        "dewpoint_c",
        "wind_ms",
        "pressure_msl_hpa",
        "weather_time_offset_min",
        "usage_category",
        "site_id",
        "floor_area_m2",
        "timezone",
        "hour_local",
        "dayofweek_local",
        "month_local",
        "is_weekend_local",
        "location_quality",
        "spatial_link",
    ):
        assert column in merged.columns, column
    assert not merged.duplicated(["building_id", "ts_utc"]).any()
    assert len(merged) == 4 * 12 * 24  # 12 days x 24 h x 4 buildings, nothing dropped
    assert merged["temp_c"].notna().all()
    assert (merged["weather_time_offset_min"] == 0).all()


def test_funnel_reconciles_with_the_selection(adapter, paths):
    result = _run(_request(), adapter, paths)
    funnel = pd.read_csv(result.output_dir / "debug_counts.csv", sep=";")
    assert funnel.iloc[0]["step"] == "dataset_catalogue" and funnel.iloc[0]["remaining"] == 12
    for before, after in zip(funnel["remaining"], funnel["remaining"][1:], strict=False):
        assert after <= before
    assert funnel.iloc[-1]["remaining"] == 4 == len(_read(result, "selected_buildings"))
    assert (
        (funnel["remaining"].shift(1) - funnel["remaining"])
        .dropna()
        .eq(funnel["removed"][1:])
        .all()
    )


def test_quality_screen_excludes_buildings_with_too_much_missing_data(adapter, paths):
    result = _run(_request(n_buildings=None), adapter, paths)
    excluded = _read(result, "excluded_buildings")
    row = excluded[excluded["building_id"] == "bdg2:East_office_f"].iloc[0]
    assert row["step"] == "max_missing_ratio" and "0." in row["detail"]
    assert result.manifest["counts"]["buildings"] == 10  # 11 offices, one excluded


def test_short_gaps_are_imputed_when_requested_and_flagged(adapter, paths):
    request = _request(
        n_buildings=None,
        quality={"imputation": "linear_short_gaps", "max_gap_hours": 3, "max_imputed_ratio": 0.05},
    )
    result = _run(request, adapter, paths)
    load = _read(result, "load_timeseries")
    b = load[load["building_id"] == "bdg2:East_office_b"]  # a 2 hour gap -> imputed
    assert b["is_imputed"].sum() == 2 and (b[b["is_imputed"]]["quality_flag"] == "imputed").all()
    a = load[load["building_id"] == "bdg2:East_office_a"]  # a 4 hour gap -> left missing
    assert a["is_imputed"].sum() == 0 and a["is_missing"].sum() == 4
    stats = json.loads((result.output_dir / "quality_report.json").read_text())["buildings"]
    imputed_stats = {s["building_id"]: s for s in stats}["bdg2:East_office_b"]
    assert imputed_stats["imputed_rows"] == 2 and imputed_stats["missing_rows"] == 2


def test_unlocated_site_is_matched_through_the_dataset_link(adapter, paths):
    result = _run(_request(area={"site_ids": ["Nox"]}, n_buildings=None), adapter, paths)
    match = _read(result, "weather_match")
    assert set(match["spatial_link"]) == {"dataset_link"}
    assert set(match["location_quality"]) == {"unknown"}
    assert match["distance_km"].isna().all()


def test_dst_days_keep_all_records_in_load_and_match(adapter, paths):
    request = _request(start=date(2016, 3, 10), end=date(2016, 3, 16), area={"site_ids": ["East"]})
    result = _run(request, adapter, paths)  # US DST starts 2016-03-13
    report = json.loads((result.output_dir / "quality_report.json").read_text())
    days = pd.DataFrame(report["weather_matching"]["dst_days"])
    assert not days.empty
    day = days[days["local_date"] == "2016-03-13"]
    assert (day["expected_hours"] == 23).all()
    assert (day["load_rows"] == 23).all() and (day["matched_rows"] == 23).all()
    merged = _read(result, "merged_training_dataset")
    local_day = merged[merged["ts_local"].dt.date == date(2016, 3, 13)]
    assert local_day.groupby("building_id").size().eq(23).all()
    assert 2 not in set(local_day["hour_local"])  # the skipped civil hour does not exist


# --- manifest, catalogue, report ----------------------------------------------------------------


def test_manifest_is_complete_and_checksums_match_the_files(adapter, paths):
    result = _run(_request(), adapter, paths)
    manifest = json.loads((result.output_dir / "extraction_manifest.json").read_text())
    request = _request()
    assert manifest["extraction_id"] == request.extraction_id
    assert manifest["request_hash"] == request.request_hash()
    assert manifest["created_at"] == NOW.isoformat()
    assert manifest["dataset"]["licence"].startswith("CC BY-SA 4.0")
    assert manifest["dataset"]["citation"]
    assert manifest["dataset"]["timestamp_clock"] == "local_standard"
    assert manifest["corrections"][0]["site_id"] == "Dub"
    assert manifest["weather"]["matching"]["method"] == "exact"
    assert manifest["units"]["target_kwh"] == "kWh per hour"
    assert manifest["limitations"] and manifest["attribution"]
    for entry in manifest["files"]:
        data = (result.output_dir / entry["name"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == entry["sha256"], entry["name"]
    assert "extraction_manifest.json" not in {e["name"] for e in manifest["files"]}
    assert manifest["counts"]["buildings"] == 4
    assert manifest["counts"]["load_rows"] == 4 * 12 * 24


def test_extraction_is_registered_in_the_catalogue(adapter, paths):
    with Catalogue(paths.catalogue_path) as catalogue:
        result = _run(_request(), adapter, paths, catalogue=catalogue)
        [entry] = catalogue.list_extractions()
        assert entry["extraction_id"] == result.extraction_id and entry["n_buildings"] == 4
        assert len(catalogue.extraction_buildings(result.extraction_id)) == 4


def test_pdf_report_documents_the_extraction_with_spanish_numbers(adapter, paths):
    result = _run(_request(), adapter, paths)
    reader = PdfReader(result.output_dir / "extraction_report.pdf")
    text = "\n".join(page.extract_text() for page in reader.pages)
    for needle in (
        result.extraction_id,
        "Building Data Genome Project 2",
        "CC BY-SA 4.0",
        "Weather matching",
        "Geographic coverage",
        "Limitations",
        "Possible ML use cases",
        "local_standard",
    ):
        assert needle in text, needle
    assert "1.152" in text  # 4 buildings x 288 hours, thousands separator is a period
    assert "1,152" not in text


def test_csv_export_uses_the_spanish_convention(adapter, paths):
    request = _request(export={"format": "csv", "csv_style": "spanish"}, n_buildings=2)
    result = _run(request, adapter, paths)
    header = (
        (result.output_dir / "merged_training_dataset.csv").read_text(encoding="utf-8").splitlines()
    )
    assert ";" in header[0] and "," not in header[0]
    first_row = header[1]
    assert re.search(r"\d,\d", first_row) and not re.search(r"\d\.\d", first_row)  # decimal comma
    frame = pd.read_csv(result.output_dir / "merged_training_dataset.csv", sep=";", decimal=",")
    assert frame["temp_c"].dtype.kind == "f"
    manifest = json.loads((result.output_dir / "extraction_manifest.json").read_text())
    assert manifest["export"] == {
        "format": "csv",
        "csv_style": "spanish",
        "sep": ";",
        "decimal": ",",
    }


def test_plain_csv_style(adapter, paths):
    result = _run(
        _request(export={"format": "csv", "csv_style": "plain"}, n_buildings=2), adapter, paths
    )
    frame = pd.read_csv(result.output_dir / "merged_training_dataset.csv")
    assert frame["temp_c"].dtype.kind == "f"


# --- failure modes -------------------------------------------------------------------------------


def test_rerun_requires_force(adapter, paths):
    _run(_request(), adapter, paths)
    with pytest.raises(ExtractionExistsError):
        _run(_request(), adapter, paths)
    _run(_request(), adapter, paths, force=True)


def test_force_only_removes_files_it_created(adapter, paths, tmp_path):
    target = tmp_path / "custom_out"
    target.mkdir()
    (target / "notes.txt").write_text("keep me")
    request = _request(export={"format": "parquet", "output_dir": str(target)})
    _run(request, adapter, paths, force=True)
    _run(request, adapter, paths, force=True)
    assert (target / "notes.txt").read_text() == "keep me"
    assert (target / "extraction_manifest.json").is_file()


def test_nothing_eligible_is_a_clear_error(adapter, paths):
    with pytest.raises(NoBuildingsError, match="usage_category"):
        _run(_request(usage_categories=["condominium"]), adapter, paths)


def test_everything_excluded_by_quality_is_a_clear_error(adapter, paths):
    with pytest.raises(NoBuildingsError, match="max_missing_ratio"):
        _run(_request(start=date(2016, 6, 1), end=date(2016, 6, 10)), adapter, paths)


def test_asking_for_more_buildings_than_available_selects_all_and_warns(adapter, paths):
    result = _run(_request(n_buildings=50), adapter, paths)
    assert result.manifest["counts"]["buildings"] == 10
    assert any("only 10" in w for w in result.manifest["warnings"])


def test_unsupported_task_and_granularity(adapter, paths):
    with pytest.raises(ExtractionRequestError, match="nilm"):
        _run(_request(task="nilm"), adapter, paths)
    with pytest.raises(ExtractionRequestError, match="granularity"):
        _run(_request(granularity_s=900), adapter, paths)


def test_request_for_another_dataset_is_refused(adapter, paths):
    with pytest.raises(ExtractionRequestError, match="dataset"):
        _run(_request(dataset_id="goiener"), adapter, paths)


# --- a failed run leaves nothing behind --------------------------------------------------------


def test_a_crash_midway_leaves_no_partial_outputs(adapter, paths, monkeypatch):
    import jardik.dataset_manager.pipeline as pipeline

    def boom(*args, **kwargs):
        raise RuntimeError("PDF engine exploded")

    monkeypatch.setattr(pipeline, "build_pdf", boom)
    request = _request()
    target = paths.extraction_dir(request.extraction_id)
    with pytest.raises(RuntimeError, match="exploded"):
        _run(request, adapter, paths)
    assert not target.exists()
    assert not list(target.parent.glob("*.partial"))
    monkeypatch.undo()
    _run(request, adapter, paths)  # no --force needed after a failed run
    assert (target / "extraction_manifest.json").is_file()


def test_a_failed_force_rerun_keeps_the_previous_extraction_intact(adapter, paths, monkeypatch):
    import jardik.dataset_manager.pipeline as pipeline

    request = _request()
    first = _run(request, adapter, paths)
    before = (first.output_dir / "extraction_manifest.json").read_bytes()
    monkeypatch.setattr(
        pipeline, "build_pdf", lambda *a, **k: (_ for _ in ()).throw(OSError("disk"))
    )
    with pytest.raises(OSError):
        _run(request, adapter, paths, force=True)
    assert (first.output_dir / "extraction_manifest.json").read_bytes() == before


# --- dead or partial meters are screened out -----------------------------------------------------


def test_a_building_with_mostly_near_zero_readings_is_excluded(mini_bdg2, tmp_path):
    csv = mini_bdg2 / "electricity_cleaned.csv"
    frame = pd.read_csv(csv)
    frame.loc[frame.index % 2 == 0, "East_office_c"] = 0.0001  # half of all hours "off"
    frame.to_csv(csv, index=False)
    adapter = Bdg2Adapter(
        Bdg2Config(data_dir=mini_bdg2, usage_map={"Office": "office", "Retail": "shop"})
    )
    paths = JardikPaths.from_root(tmp_path / "project")
    result = run_extraction(
        _request(n_buildings=None, area={"site_ids": ["East"]}), adapter, paths, now=lambda: NOW
    )
    excluded = pd.read_parquet(result.output_dir / "excluded_buildings.parquet")
    row = excluded[excluded["building_id"] == "bdg2:East_office_c"].iloc[0]
    assert row["step"] == "max_near_zero_ratio" and "near_zero_ratio=0.5" in row["detail"]
    assert "bdg2:East_office_c" not in set(
        pd.read_parquet(result.output_dir / "selected_buildings.parquet")["building_id"]
    )
    funnel = pd.read_csv(result.output_dir / "debug_counts.csv", sep=";")
    assert "max_near_zero_ratio" in set(funnel["step"])


def test_the_near_zero_limit_can_be_relaxed(mini_bdg2, tmp_path):
    csv = mini_bdg2 / "electricity_cleaned.csv"
    frame = pd.read_csv(csv)
    frame.loc[frame.index % 2 == 0, "East_office_c"] = 0.0001
    frame.to_csv(csv, index=False)
    adapter = Bdg2Adapter(Bdg2Config(data_dir=mini_bdg2, usage_map={"Office": "office"}))
    request = _request(
        n_buildings=None,
        area={"site_ids": ["East"]},
        quality={"max_near_zero_ratio": 1.0},
    )
    result = run_extraction(
        request, adapter, JardikPaths.from_root(tmp_path / "p2"), now=lambda: NOW
    )
    selected = pd.read_parquet(result.output_dir / "selected_buildings.parquet")
    assert "bdg2:East_office_c" in set(selected["building_id"])
