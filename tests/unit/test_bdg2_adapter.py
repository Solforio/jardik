"""Building Data Genome 2 adapter, tested on a tiny synthetic copy of the real file layout."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jardik.core.interfaces import DatasetAdapter, TimeWindow
from jardik.core.schemas import BuildingMetadata, validate_load, validate_weather
from jardik.dataset_manager.adapters.base import DatasetError
from jardik.dataset_manager.adapters.bdg2 import Bdg2Adapter, Bdg2Config, SiteOverride

HOURS = pd.date_range("2016-03-01", "2016-03-20 23:00", freq="h")  # naive local standard time


def _window(start: str, end: str) -> TimeWindow:
    return TimeWindow(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"))


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    meta = pd.DataFrame(
        {
            "building_id": [
                "East_office_ann",
                "East_retail_bob",
                "Dub_office_cy",
                "Nox_education_di",
                "East_nometer_ed",
            ],
            "site_id": ["East", "East", "Dub", "Nox", "East"],
            "primaryspaceusage": ["Office", "Retail", "Office", "Education", "Office"],
            "sub_primaryspaceusage": ["Office", "Retail Store", "Office", "College\t", "Office"],
            "sqm": [1000.0, 200.0, 500.0, 3000.0, 100.0],
            "lat": [28.5, 28.5, 53.35, np.nan, 28.5],
            "lng": [-81.4, -81.4, 6.26, np.nan, -81.4],  # Dublin longitude has the wrong sign
            "timezone": ["US/Eastern", "US/Eastern", "Europe/Dublin", "US/Pacific", "US/Eastern"],
            "electricity": ["Yes", "Yes", "Yes", "Yes", np.nan],
            "solar": [np.nan, "Yes", np.nan, np.nan, np.nan],
            "yearbuilt": [1990.0, np.nan, 2005.0, 1970.0, 1980.0],
        }
    )
    meta.to_csv(tmp_path / "metadata.csv", index=False)

    rng = np.random.default_rng(0)
    elec = pd.DataFrame(
        {
            "timestamp": HOURS,
            "East_office_ann": rng.uniform(1, 5, len(HOURS)),
            "East_retail_bob": rng.uniform(1, 5, len(HOURS)),
            "Dub_office_cy": rng.uniform(1, 5, len(HOURS)),
            "Nox_education_di": rng.uniform(1, 5, len(HOURS)),
        }
    )
    elec.loc[10:13, "East_office_ann"] = np.nan  # a 4 hour gap (cleaned file marks outliers as NaN)
    elec.to_csv(tmp_path / "electricity_cleaned.csv", index=False)

    weather = []
    for site in ("East", "Dub", "Nox"):
        weather.append(
            pd.DataFrame(
                {
                    "timestamp": HOURS,
                    "site_id": site,
                    "airTemperature": np.linspace(0, 20, len(HOURS)),
                    "cloudCoverage": 4.0,
                    "dewTemperature": 5.0,
                    "precipDepth1HR": 0.0,
                    "precipDepth6HR": np.nan,
                    "seaLvlPressure": 1013.0,
                    "windDirection": 180.0,
                    "windSpeed": 3.0,
                }
            )
        )
    w = pd.concat(weather, ignore_index=True)
    w.loc[5, "precipDepth1HR"] = -1.0  # ISD convention: trace of precipitation
    w.loc[6, "cloudCoverage"] = 9.0  # sky obscured
    w.loc[7, "airTemperature"] = 400.0  # sensor glitch
    w.loc[8, "windSpeed"] = np.nan
    w.to_csv(tmp_path / "weather.csv", index=False)
    return tmp_path


def _config(raw_dir: Path, **overrides) -> Bdg2Config:
    base = {
        "data_dir": raw_dir,
        "electricity_file": "electricity_cleaned.csv",
        "usage_map": {"Office": "office", "Retail": "shop"},
        "usage_sub_map": {"Lodging/residential|Multifamily Housing": "condominium"},
        "site_overrides": {
            "Dub": SiteOverride(
                lng=-6.26, reason="longitude sign error: Dublin is west of Greenwich"
            )
        },
    }
    base.update(overrides)
    return Bdg2Config(**base)


@pytest.fixture
def adapter(raw_dir: Path) -> Bdg2Adapter:
    return Bdg2Adapter(_config(raw_dir))


def test_implements_the_dataset_adapter_protocol(adapter):
    assert isinstance(adapter, DatasetAdapter)
    assert adapter.dataset_id == "bdg2"


def test_discover_lists_only_buildings_with_an_electricity_series(adapter):
    cat = adapter.discover()
    assert sorted(cat["building_id"]) == [
        "bdg2:Dub_office_cy",
        "bdg2:East_office_ann",
        "bdg2:East_retail_bob",
        "bdg2:Nox_education_di",
    ]  # East_nometer_ed has no electricity column


def test_metadata_rows_satisfy_the_building_model(adapter):
    cat = adapter.discover()
    fields = set(BuildingMetadata.model_fields)
    for row in cat.to_dict("records"):
        clean = {k: (None if pd.isna(v) else v) for k, v in row.items() if k in fields}
        BuildingMetadata(**clean)


def test_usage_mapping_keeps_the_raw_label(adapter):
    cat = adapter.discover().set_index("building_id")
    assert cat.loc["bdg2:East_office_ann", "usage_category"] == "office"
    assert cat.loc["bdg2:East_retail_bob", "usage_category"] == "shop"
    assert cat.loc["bdg2:Nox_education_di", "usage_category"] == "other"  # not in the map
    assert cat.loc["bdg2:Nox_education_di", "usage_raw"] == "Education"
    assert cat.loc["bdg2:East_retail_bob", "usage_sub_raw"] == "Retail Store"
    assert cat.loc["bdg2:Nox_education_di", "usage_sub_raw"] == "College"  # whitespace stripped


def test_unmapped_labels_do_not_silently_become_known_categories(adapter):
    assert adapter.unmapped_usage_labels() == {"Education"}


def test_static_fields(adapter):
    row = adapter.discover().set_index("building_id").loc["bdg2:East_retail_bob"]
    assert row["site_id"] == "East" and row["weather_link"] == "East"
    assert row["floor_area_m2"] == 200.0
    assert row["has_pv"] is True or row["has_pv"] == True  # noqa: E712  (solar meter present)
    assert pd.isna(row["contracted_power_kw"]) and pd.isna(row["country"])
    assert row["licence_id"] == "bdg2"
    ann = adapter.discover().set_index("building_id").loc["bdg2:East_office_ann"]
    assert ann["year_built"] == 1990 and pd.isna(ann["has_pv"])


def test_declared_coordinate_override_is_applied_and_recorded(adapter):
    row = adapter.discover().set_index("building_id").loc["bdg2:Dub_office_cy"]
    assert row["lon"] == -6.26 and row["lat"] == 53.35
    [correction] = adapter.corrections
    assert correction["site_id"] == "Dub" and correction["field"] == "lng"
    assert correction["old"] == 6.26 and correction["new"] == -6.26
    assert "sign" in correction["reason"]


def test_load_timeseries_follows_the_contract_on_a_complete_grid(adapter):
    ids = ["bdg2:East_office_ann", "bdg2:Dub_office_cy"]
    df = adapter.load_timeseries(ids, _window("2016-03-05", "2016-03-08"))
    validate_load(df)
    assert set(df["building_id"]) == set(ids)
    assert len(df) == 2 * 72  # every hourly slot of the window, present or not
    assert (df["unit"] == "kWh").all() and (df["meter_type"] == "electricity_total").all()
    assert (df["granularity_s"] == 3600).all()


def test_timestamps_are_local_standard_time_converted_to_utc(adapter):
    df = adapter.load_timeseries(["bdg2:East_office_ann"], _window("2016-03-01", "2016-03-02"))
    df = df.sort_values("ts_utc")
    # File says 2016-03-01 00:00 (Eastern standard time, UTC-5) -> 05:00 UTC. The five UTC hours
    # of the window that precede it have no data and are reported as missing, not dropped.
    before = df["ts_utc"] < pd.Timestamp("2016-03-01 05:00", tz="UTC")
    assert before.sum() == 5 and df[before]["is_missing"].all()
    # US DST began on 2016-03-13, but the file keeps standard time: still UTC-5, nothing skipped.
    after = adapter.load_timeseries(["bdg2:East_office_ann"], _window("2016-03-14", "2016-03-15"))
    assert after["ts_utc"].is_unique and not after["is_missing"].any()
    first_file_row_utc = pd.Timestamp("2016-03-14 05:00", tz="UTC")  # = 2016-03-14 00:00 local std
    assert first_file_row_utc in set(after["ts_utc"])
    # ts_local is civil local time (DST aware): UTC-4 after the change.
    assert (after["utc_offset_min"] == -240).all()


def test_values_roundtrip_and_gaps_are_flagged(adapter, raw_dir):
    raw = pd.read_csv(raw_dir / "electricity_cleaned.csv", parse_dates=["timestamp"])
    df = adapter.load_timeseries(["bdg2:East_office_ann"], _window("2016-03-01", "2016-03-03"))
    # local standard 2016-03-01 10:00-13:00 is the NaN gap -> 15:00-18:00 UTC
    gap = df[
        (df["ts_utc"] >= pd.Timestamp("2016-03-01 15:00", tz="UTC"))
        & (df["ts_utc"] < pd.Timestamp("2016-03-01 19:00", tz="UTC"))
    ]
    assert gap["is_missing"].all() and gap["value"].isna().all()
    assert (gap["quality_flag"] == "missing").all()
    present = df[~df["is_missing"]].sort_values("ts_utc")
    assert present["value"].iloc[0] == pytest.approx(raw["East_office_ann"].iloc[0])
    assert present["ts_utc"].iloc[0] == pd.Timestamp("2016-03-01 05:00", tz="UTC")
    assert (present["quality_flag"] == "ok").all()


def test_dublin_uses_gmt_not_irish_summer_time(adapter):
    df = adapter.load_timeseries(["bdg2:Dub_office_cy"], _window("2016-03-01", "2016-03-02"))
    first_present = df[~df["is_missing"]].sort_values("ts_utc").iloc[0]
    assert first_present["ts_utc"] == pd.Timestamp("2016-03-01 00:00", tz="UTC")


def test_unknown_building_is_an_error(adapter):
    with pytest.raises(DatasetError, match="bdg2:nope"):
        adapter.load_timeseries(["bdg2:nope"], _window("2016-03-01", "2016-03-02"))


def test_weather_points_use_the_site_location(adapter):
    points = adapter.weather_points().set_index("weather_point_id")
    assert set(points.index) == {"East", "Dub", "Nox"}
    assert points.loc["East", "lat"] == pytest.approx(28.5)
    assert points.loc["Dub", "lon"] == pytest.approx(-6.26)  # corrected coordinate
    assert pd.isna(points.loc["Nox", "lat"])  # unknown location stays unknown


def test_weather_follows_the_contract_and_declared_value_rules(adapter):
    w = adapter.load_weather(
        ["East"],
        _window("2016-03-01", "2016-03-03"),
        ["temp_c", "precip_mm", "cloud_pct", "wind_ms", "pressure_msl_hpa"],
    )
    validate_weather(w)
    w = w.set_index("ts_utc")
    start = pd.Timestamp("2016-03-01 05:00", tz="UTC")  # file's first row, East is UTC-5
    assert w.loc[start + pd.Timedelta(hours=5), "precip_mm"] == 0.0  # -1 (trace) -> 0
    assert w.loc[start + pd.Timedelta(hours=6), "cloud_pct"] == 100.0  # 9 oktas, obscured
    assert w.loc[start + pd.Timedelta(hours=4), "cloud_pct"] == 50.0  # 4 oktas
    assert np.isnan(w.loc[start + pd.Timedelta(hours=7), "temp_c"])  # 400 C is not physical
    assert np.isnan(w.loc[start + pd.Timedelta(hours=8), "wind_ms"])
    assert (w["pressure_msl_hpa"].dropna() == 1013.0).all()
    assert (w["product_type"] == "observation").all() and (w["source"] == "bdg2").all()


def test_weather_screening_is_counted_not_hidden(adapter):
    adapter.load_weather(
        ["East"], _window("2016-03-01", "2016-03-03"), ["temp_c", "precip_mm", "cloud_pct"]
    )
    report = adapter.weather_screening
    assert report["temp_c"]["out_of_range_set_to_nan"] == 1
    assert report["precip_mm"]["trace_set_to_zero"] == 1
    assert report["cloud_pct"]["obscured_sky_set_to_100"] == 1


def test_weather_unknown_variable_and_point(adapter):
    window = _window("2016-03-01", "2016-03-03")
    with pytest.raises(DatasetError, match="ghi_wm2"):
        adapter.load_weather(["East"], window, ["ghi_wm2"])
    with pytest.raises(DatasetError, match="Mars"):
        adapter.load_weather(["Mars"], window, ["temp_c"])


def test_info_describes_licence_clock_and_files(adapter):
    info = adapter.info()
    assert info["dataset_id"] == "bdg2"
    assert info["licence_id"] == "bdg2" and "CC BY-SA 4.0" in info["licence"]
    assert info["native_granularity_s"] == 3600
    assert info["timestamp_clock"] == "local_standard"
    assert info["supported_tasks"] == ["forecasting"]
    assert {f["name"] for f in info["files"]} == {
        "metadata.csv",
        "electricity_cleaned.csv",
        "weather.csv",
    }
    assert all(len(f["sha256"]) == 64 for f in info["files"])
    assert "citation" in info


def test_missing_files_give_a_helpful_error(tmp_path: Path):
    with pytest.raises(DatasetError, match="dataset fetch"):
        Bdg2Adapter(Bdg2Config(data_dir=tmp_path)).discover()


def test_civil_clock_option_requires_dst_policy(raw_dir):
    # With a civil clock the repeated/skipped hour must be handled explicitly.
    cfg = _config(raw_dir, timestamp_clock="local_civil")
    adapter = Bdg2Adapter(cfg)
    with pytest.raises(Exception, match="DST"):
        adapter.load_timeseries(["bdg2:East_office_ann"], _window("2016-03-01", "2016-03-20"))


def test_missing_usage_label_is_reported_not_crashed(raw_dir):
    # Real BDG2 has buildings without a primaryspaceusage value.
    meta = pd.read_csv(raw_dir / "metadata.csv")
    meta.loc[meta["building_id"] == "East_retail_bob", "primaryspaceusage"] = np.nan
    meta.loc[meta["building_id"] == "East_retail_bob", "sub_primaryspaceusage"] = np.nan
    meta.to_csv(raw_dir / "metadata.csv", index=False)
    adapter = Bdg2Adapter(_config(raw_dir))
    row = adapter.discover().set_index("building_id").loc["bdg2:East_retail_bob"]
    assert row["usage_category"] == "other" and row["usage_raw"] == "(missing)"
    assert "(missing)" in adapter.unmapped_usage_labels()
