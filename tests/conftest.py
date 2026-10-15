"""Shared fixtures: tiny synthetic frames only, never real datasets."""

from __future__ import annotations

import pandas as pd
import pytest

from jardik.core.timeutils import add_local_columns


@pytest.fixture
def load_frame() -> pd.DataFrame:
    """Valid 6-hour hourly load frame for one office building (UTC key, Madrid local)."""
    ts = pd.date_range("2024-01-15 00:00", periods=6, freq="h", tz="UTC")
    df = pd.DataFrame(
        {
            "building_id": "bdg2:office_1",
            "meter_type": "electricity_total",
            "appliance": None,
            "ts_utc": ts,
            "value": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5],
            "unit": "kWh",
            "granularity_s": 3600,
            "is_missing": False,
            "is_imputed": False,
            "quality_flag": "ok",
        }
    )
    return add_local_columns(df, "Europe/Madrid")


@pytest.fixture
def weather_frame() -> pd.DataFrame:
    ts = pd.date_range("2024-01-15 00:00", periods=6, freq="h", tz="UTC")
    return pd.DataFrame(
        {
            "weather_point_id": "wp_1",
            "ts_utc": ts,
            "source": "bdg2_file",
            "product_type": "observation",
            "temp_c": [5.0, 4.5, 4.0, 3.5, 3.0, 3.5],
            "rh_pct": [80.0, 82.0, 85.0, 87.0, 90.0, 88.0],
        }
    )


# --- a miniature Building Data Genome 2 on disk ------------------------------------------------

import numpy as np  # noqa: E402


@pytest.fixture
def mini_bdg2(tmp_path):
    """12 buildings on 3 sites (Eastern US, Dublin with a wrong-sign longitude, unlocated Pacific
    site) with 25 days of hourly local-standard-time data crossing the US DST change."""
    data_dir = tmp_path / "bdg2_raw"
    data_dir.mkdir()
    hours = pd.date_range("2016-03-01", "2016-03-25 23:00", freq="h")
    buildings = (
        [(f"East_office_{c}", "East", "Office", "US/Eastern", 28.5, -81.4) for c in "abcdef"]
        + [(f"Dub_office_{c}", "Dub", "Office", "Europe/Dublin", 53.35, 6.26) for c in "abc"]
        + [("Dub_retail_a", "Dub", "Retail", "Europe/Dublin", 53.35, 6.26)]
        + [(f"Nox_office_{c}", "Nox", "Office", "US/Pacific", np.nan, np.nan) for c in "ab"]
    )
    meta = pd.DataFrame(
        buildings, columns=["building_id", "site_id", "primaryspaceusage", "timezone", "lat", "lng"]
    )
    meta["sub_primaryspaceusage"] = meta["primaryspaceusage"]
    meta["sqm"] = 1000.0
    meta["electricity"] = "Yes"
    meta["solar"] = np.nan
    meta["yearbuilt"] = 1990.0
    meta.to_csv(data_dir / "metadata.csv", index=False)

    rng = np.random.default_rng(7)
    elec = pd.DataFrame({"timestamp": hours})
    daily = 2 + np.sin((hours.hour.to_numpy() - 6) / 24 * 2 * np.pi)
    for name in meta["building_id"]:
        elec[name] = daily + rng.uniform(0, 0.5, len(hours))
    elec.loc[200:203, "East_office_a"] = np.nan  # 4 h gap
    elec.loc[300:301, "East_office_b"] = np.nan  # 2 h gap
    elec.loc[150:400, "East_office_f"] = np.nan  # ~40% of the file missing
    elec.to_csv(data_dir / "electricity_cleaned.csv", index=False)

    weather = []
    for site in ("East", "Dub", "Nox"):
        weather.append(
            pd.DataFrame(
                {
                    "timestamp": hours,
                    "site_id": site,
                    "airTemperature": 10 + 5 * np.sin(np.arange(len(hours)) / 24 * 2 * np.pi),
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
    pd.concat(weather, ignore_index=True).to_csv(data_dir / "weather.csv", index=False)
    return data_dir
