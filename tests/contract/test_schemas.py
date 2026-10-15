import pandas as pd
import pytest
from pydantic import ValidationError

from jardik.core.schemas import (
    LOAD_CONTRACT,
    BuildingMetadata,
    ContractError,
    LocationQuality,
    collect_violations,
    derive_location_quality,
    validate_load,
    validate_weather,
    validate_weather_match,
)


def _building(**overrides):
    base = {
        "building_id": "bdg2:office_1",
        "dataset_id": "bdg2",
        "usage_category": "office",
        "timezone": "Europe/Madrid",
    }
    base.update(overrides)
    return BuildingMetadata(**base)


# --- building metadata ---------------------------------------------------------------------


def test_location_quality_ladder():
    assert derive_location_quality(40.4, -3.7, "28001", "Madrid", "Madrid", "Madrid") == (
        LocationQuality.COORDINATES
    )
    assert derive_location_quality(None, None, "28001", "Madrid", None, None) == (
        LocationQuality.POSTAL_CENTROID
    )
    assert derive_location_quality(None, None, None, "Madrid", None, None) == (
        LocationQuality.MUNICIPALITY
    )
    assert derive_location_quality(None, None, None, None, "Madrid", None) == (
        LocationQuality.PROVINCE
    )
    assert derive_location_quality(None, None, None, None, None, "Madrid") == (
        LocationQuality.REGION
    )
    assert derive_location_quality(None, None, None, None, None, None) == LocationQuality.UNKNOWN


def test_location_quality_is_derived_not_input():
    assert _building(lat=40.4, lon=-3.7).location_quality == LocationQuality.COORDINATES
    assert _building(province="Madrid").location_quality == LocationQuality.PROVINCE
    with pytest.raises(ValidationError):
        _building(location_quality="coordinates")  # extra field is forbidden


@pytest.mark.parametrize(
    "overrides",
    [
        {"building_id": "no_separator"},
        {"building_id": "other:office_1"},  # prefix differs from dataset_id
        {"timezone": "Mars/Olympus"},
        {"lat": 40.0},  # lon missing
        {"lat": 95.0, "lon": 0.0},
        {"country": "es"},
        {"floor_area_m2": 0},
        {"usage_category": "castle"},
    ],
)
def test_invalid_building_rejected(overrides):
    with pytest.raises(ValidationError):
        _building(**overrides)


# --- load contract -------------------------------------------------------------------------


def test_valid_load_frame(load_frame):
    validate_load(load_frame)


def test_load_missing_column(load_frame):
    with pytest.raises(ContractError) as err:
        validate_load(load_frame.drop(columns=["unit"]))
    assert "missing required column 'unit'" in err.value.violations


def test_load_naive_timestamps_rejected(load_frame):
    bad = load_frame.assign(ts_utc=load_frame["ts_utc"].dt.tz_localize(None))
    problems = collect_violations(bad, LOAD_CONTRACT)
    assert any("ts_utc" in p and "datetime_utc" in p for p in problems)


def test_load_duplicates_rejected(load_frame):
    bad = pd.concat([load_frame, load_frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ContractError, match="duplicated"):
        validate_load(bad)


def test_load_wrong_unit_for_meter(load_frame):
    bad = load_frame.assign(unit="Wh")
    with pytest.raises(ContractError, match="must use unit 'kWh'"):
        validate_load(bad)


def test_load_missing_flag_consistency(load_frame):
    bad = load_frame.copy()
    bad.loc[0, "value"] = float("nan")  # null but not flagged
    with pytest.raises(ContractError, match="not flagged is_missing"):
        validate_load(bad)
    ok = bad.assign(is_missing=[True, False, False, False, False, False])
    ok["quality_flag"] = ["missing", "ok", "ok", "ok", "ok", "ok"]
    validate_load(ok)


def test_load_ts_local_must_match_ts_utc(load_frame):
    bad = load_frame.copy()
    bad["ts_local"] = bad["ts_local"] + pd.Timedelta(hours=1)
    with pytest.raises(ContractError, match="same instant"):
        validate_load(bad)


def test_load_disallowed_meter_type(load_frame):
    with pytest.raises(ContractError, match="disallowed"):
        validate_load(load_frame.assign(meter_type="gas"))


# --- weather contracts ---------------------------------------------------------------------


def test_valid_weather_frame(weather_frame):
    validate_weather(weather_frame)


def test_weather_implausible_value(weather_frame):
    bad = weather_frame.assign(rh_pct=[80.0, 82.0, 150.0, 87.0, 90.0, 88.0])
    with pytest.raises(ContractError, match="rh_pct"):
        validate_weather(bad)


def test_forecast_requires_issue_time(weather_frame):
    bad = weather_frame.assign(product_type="forecast")
    with pytest.raises(ContractError, match="issued_at_utc"):
        validate_weather(bad)
    good = bad.assign(issued_at_utc=weather_frame["ts_utc"] - pd.Timedelta(hours=12))
    validate_weather(good)


def test_weather_match_contract():
    ts = pd.date_range("2024-01-15", periods=2, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "building_id": "bdg2:office_1",
            "ts_utc": ts,
            "weather_source": "bdg2_file",
            "matched_lat": 40.4,
            "matched_lon": -3.7,
            "distance_km": [1.2, 1.2],
            "weather_time_offset_min": [0, 0],
            "timezone_used": "Europe/Madrid",
            "matching_method": "exact",
            "spatial_link": "nearest_point",
            "location_quality": "coordinates",
        }
    )
    validate_weather_match(df)
    with pytest.raises(ContractError, match="distance_km"):
        validate_weather_match(df.assign(distance_km=[-1.0, 1.0]))
    with pytest.raises(ContractError, match="disallowed"):
        validate_weather_match(df.assign(matching_method="magic"))


def test_weather_match_dataset_link_may_have_unknown_coordinates():
    ts = pd.date_range("2024-01-15", periods=2, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "building_id": "bdg2:office_1",
            "ts_utc": ts,
            "weather_source": "bdg2_file",
            "matched_lat": float("nan"),
            "matched_lon": float("nan"),
            "distance_km": float("nan"),
            "weather_time_offset_min": [0, 0],
            "timezone_used": "US/Eastern",
            "matching_method": "exact",
            "spatial_link": "dataset_link",
            "location_quality": "unknown",
        }
    )
    validate_weather_match(df)
    with pytest.raises(ContractError, match="nearest_point"):
        validate_weather_match(df.assign(spatial_link="nearest_point"))


def test_pressure_msl_is_a_canonical_variable():
    from jardik.core.schemas import WEATHER_CANONICAL_UNITS, WEATHER_VARIABLES

    assert "pressure_msl_hpa" in WEATHER_VARIABLES
    assert WEATHER_CANONICAL_UNITS["pressure_msl_hpa"] == "hPa"
    assert set(WEATHER_VARIABLES) == set(WEATHER_CANONICAL_UNITS)
