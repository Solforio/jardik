from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from jardik.dataset_manager.request import ExtractionRequest
from jardik.dataset_manager.selection.filters import apply_metadata_filters
from jardik.dataset_manager.selection.sampling import order_candidates


def _request(**overrides) -> ExtractionRequest:
    base = {"name": "offices", "start": date(2016, 1, 1), "end": date(2016, 2, 1)}
    base.update(overrides)
    return ExtractionRequest(**base)


# --- request model ------------------------------------------------------------------------------


def test_defaults_follow_the_plan():
    req = _request()
    assert req.task == "forecasting" and req.dataset_id == "bdg2"
    assert req.granularity_s == 3600
    assert req.weather.matching.method == "exact"
    assert req.export.format == "parquet"
    assert req.quality.imputation == "none"


def test_window_is_utc_half_open():
    window = _request().window
    assert window.start == pd.Timestamp("2016-01-01", tz="UTC")
    assert window.end == pd.Timestamp("2016-02-01", tz="UTC")


@pytest.mark.parametrize(
    "overrides",
    [
        {"end": date(2016, 1, 1)},
        {"name": "has space"},
        {"granularity_s": 0},
        {"n_buildings": 0},
        {"quality": {"max_missing_ratio": 1.5}},
        {"weather": {"variables": ["nope"]}},
        {"weather": {"variables": []}},
        {"contracted_power_kw": (10.0, 5.0)},
        {"area": {"bbox": (50.0, 0.0, 40.0, 10.0)}},
        {"usage_categories": ["castle"]},
        {"unknown_field": 1},
    ],
)
def test_invalid_requests_are_rejected(overrides):
    with pytest.raises(ValidationError):
        _request(**overrides)


def test_hash_is_stable_and_ignores_the_output_folder():
    a = _request()
    b = _request(export={"output_dir": "somewhere/else"})
    c = _request(n_buildings=10)
    assert a.request_hash() == b.request_hash() != c.request_hash()
    assert len(a.request_hash()) == 12
    assert a.extraction_id == f"offices-{a.request_hash()}"


def test_yaml_roundtrip(tmp_path: Path):
    path = tmp_path / "req.yaml"
    path.write_text(
        "name: offices_2016\n"
        "start: 2016-01-01\nend: 2016-07-01\n"
        "usage_categories: [office]\n"
        "n_buildings: 20\n"
        "quality: {max_missing_ratio: 0.05, imputation: linear_short_gaps, max_gap_hours: 6}\n"
        "weather: {variables: [temp_c, dewpoint_c], matching: {method: nearest_previous}}\n",
        encoding="utf-8",
    )
    req = ExtractionRequest.from_yaml(path)
    assert req.usage_categories == ["office"] and req.n_buildings == 20
    assert req.weather.matching.method == "nearest_previous"
    assert req.quality.max_gap_hours == 6


# --- metadata filters -----------------------------------------------------------------------------


@pytest.fixture
def catalogue() -> pd.DataFrame:
    rows = [
        # id, site, usage, country, lat, lon, quality-bearing fields, power
        ("bdg2:a", "S1", "office", "ES", 40.4, -3.7, None, 40.0),
        ("bdg2:b", "S1", "office", "ES", 40.4, -3.7, None, 80.0),
        ("bdg2:c", "S2", "office", "FR", 48.8, 2.3, None, None),
        ("bdg2:d", "S2", "shop", "FR", 48.8, 2.3, None, 20.0),
        ("bdg2:e", "S3", "office", None, np.nan, np.nan, None, 10.0),
        ("bdg2:f", "S3", "other", None, np.nan, np.nan, "Murcia", 15.0),
    ]
    df = pd.DataFrame(
        rows,
        columns=[
            "building_id",
            "site_id",
            "usage_category",
            "country",
            "lat",
            "lon",
            "province",
            "contracted_power_kw",
        ],
    )
    df["region"] = None
    df["municipality"] = None
    df["postal_code"] = None
    return df


def test_funnel_starts_with_the_whole_catalogue(catalogue):
    eligible, funnel = apply_metadata_filters(catalogue, _request())
    assert len(eligible) == 6
    assert funnel[0] == {"step": "dataset_catalogue", "remaining": 6, "removed": 0, "detail": ""}


def test_usage_filter_and_reconciliation(catalogue):
    eligible, funnel = apply_metadata_filters(catalogue, _request(usage_categories=["office"]))
    assert sorted(eligible["building_id"]) == ["bdg2:a", "bdg2:b", "bdg2:c", "bdg2:e"]
    for before, after in zip(funnel, funnel[1:], strict=False):
        assert after["remaining"] == before["remaining"] - after["removed"]
    assert funnel[-1]["remaining"] == len(eligible)


def test_site_country_and_bbox_filters(catalogue):
    only_s1, _ = apply_metadata_filters(catalogue, _request(area={"site_ids": ["S1"]}))
    assert set(only_s1["site_id"]) == {"S1"}
    spain, _ = apply_metadata_filters(catalogue, _request(area={"countries": ["ES"]}))
    assert set(spain["building_id"]) == {"bdg2:a", "bdg2:b"}
    box, _ = apply_metadata_filters(catalogue, _request(area={"bbox": (45.0, 0.0, 50.0, 5.0)}))
    assert set(box["site_id"]) == {"S2"}  # unlocated buildings never pass a bbox filter


def test_province_filter_is_case_insensitive(catalogue):
    out, _ = apply_metadata_filters(catalogue, _request(area={"provinces": ["murcia"]}))
    assert out["building_id"].tolist() == ["bdg2:f"]


def test_min_location_quality(catalogue):
    out, funnel = apply_metadata_filters(catalogue, _request(min_location_quality="coordinates"))
    assert set(out["building_id"]) == {"bdg2:a", "bdg2:b", "bdg2:c", "bdg2:d"}
    assert "location_quality" in funnel[-1]["step"]
    # province-level is acceptable for 'f' (Murcia) but not for 'e' (nothing known)
    out2, _ = apply_metadata_filters(catalogue, _request(min_location_quality="province"))
    assert set(out2["building_id"]) == {"bdg2:a", "bdg2:b", "bdg2:c", "bdg2:d", "bdg2:f"}


def test_contracted_power_range_excludes_unknown_values_explicitly(catalogue):
    out, funnel = apply_metadata_filters(catalogue, _request(contracted_power_kw=(15.0, 50.0)))
    assert set(out["building_id"]) == {"bdg2:a", "bdg2:d", "bdg2:f"}
    step = funnel[-1]
    assert step["removed"] == 3 and "unknown" in step["detail"]


def test_filters_do_not_mutate_the_catalogue(catalogue):
    before = catalogue.copy()
    apply_metadata_filters(catalogue, _request(usage_categories=["office"]))
    pd.testing.assert_frame_equal(catalogue, before)


# --- sampling ----------------------------------------------------------------------------------


@pytest.fixture
def pool() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "building_id": [f"b{i:02d}" for i in range(12)],
            "site_id": ["S1"] * 6 + ["S2"] * 4 + ["S3"] * 2,
        }
    )


def test_random_order_is_deterministic_per_seed(pool):
    a = order_candidates(pool, "random", seed=1)
    assert a == order_candidates(pool, "random", seed=1)
    assert a != order_candidates(pool, "random", seed=2)
    assert sorted(a) == sorted(pool["building_id"])


def test_order_does_not_depend_on_input_row_order(pool):
    shuffled = pool.sample(frac=1.0, random_state=0)
    assert order_candidates(pool, "random", seed=5) == order_candidates(shuffled, "random", seed=5)


def test_first_sampling_is_alphabetical(pool):
    assert order_candidates(pool, "first", seed=0) == sorted(pool["building_id"])


def test_stratified_by_site_alternates_sites(pool):
    order = order_candidates(pool, "stratified_site", seed=3)
    site = dict(zip(pool["building_id"], pool["site_id"], strict=True))
    assert {site[b] for b in order[:3]} == {"S1", "S2", "S3"}
    assert sorted(order) == sorted(pool["building_id"])


def test_empty_pool_gives_empty_order():
    empty = pd.DataFrame({"building_id": [], "site_id": []})
    assert order_candidates(empty, "random", seed=0) == []
