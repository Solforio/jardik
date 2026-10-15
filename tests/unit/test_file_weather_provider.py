from pathlib import Path

import pandas as pd
import pytest

from jardik.core.config import FileWeatherConfig, TimePolicyConfig
from jardik.core.interfaces import Location, TimeWindow, WeatherProvider
from jardik.core.schemas import ContractError
from jardik.core.timeutils import TimeError
from jardik.providers.base import (
    ProviderConfigError,
    ProviderError,
    ProviderNotSupportedError,
)
from jardik.providers.weather.file import FileWeatherProvider

MADRID = Location(40.4168, -3.7038)


def _window(start: str, end: str) -> TimeWindow:
    return TimeWindow(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"))


def _write(tmp_path: Path, text: str, name: str = "weather.csv") -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def _cfg(path: Path, **overrides) -> FileWeatherConfig:
    base = {
        "path": path,
        "source": "bdg2_file",
        "timestamp_col": "timestamp",
        "columns": {"temp_c": "airTemperature", "rh_pct": "humidity"},
        "units": {"temp_c": "degC", "rh_pct": "%"},
        "lat": 40.4168,
        "lon": -3.7038,
        "point_id": "site_a",
    }
    base.update(overrides)
    return FileWeatherConfig(**base)


UTC_CSV = (
    "timestamp,airTemperature,humidity\n"
    "2024-01-15T00:00:00Z,5.0,80\n"
    "2024-01-15T01:00:00Z,4.0,82\n"
    "2024-01-15T02:00:00Z,3.0,85\n"
    "2024-01-15T03:00:00Z,2.0,87\n"
)


def test_implements_protocol_and_reads_history(tmp_path: Path):
    provider = FileWeatherProvider(_cfg(_write(tmp_path, UTC_CSV)), TimePolicyConfig())
    assert isinstance(provider, WeatherProvider)
    df = provider.get_history(MADRID, _window("2024-01-15 01:00", "2024-01-15 03:00"), ["temp_c"])
    assert df["temp_c"].tolist() == [4.0, 3.0]  # window is half-open [start, end)
    assert df["weather_point_id"].unique().tolist() == ["site_a"]
    assert df["source"].unique().tolist() == ["bdg2_file"]
    assert df["product_type"].unique().tolist() == ["observation"]
    assert str(df["ts_utc"].dt.tz) == "UTC"
    assert "rh_pct" not in df.columns  # only requested variables


def test_units_are_converted_to_canonical(tmp_path: Path):
    csv = "timestamp,airTemperature,humidity\n2024-01-15T00:00:00Z,50,0.5\n"
    cfg = _cfg(_write(tmp_path, csv), units={"temp_c": "degF", "rh_pct": "fraction"})
    df = FileWeatherProvider(cfg, TimePolicyConfig()).get_history(
        MADRID, _window("2024-01-15", "2024-01-16"), ["temp_c", "rh_pct"]
    )
    assert df["temp_c"].iloc[0] == pytest.approx(10.0)
    assert df["rh_pct"].iloc[0] == pytest.approx(50.0)


def test_units_must_be_declared_for_every_mapped_column(tmp_path: Path):
    cfg = _cfg(_write(tmp_path, UTC_CSV), units={"temp_c": "degC"})
    with pytest.raises(ProviderConfigError, match="rh_pct"):
        FileWeatherProvider(cfg, TimePolicyConfig())


def test_unknown_canonical_variable_in_mapping_rejected(tmp_path: Path):
    cfg = _cfg(
        _write(tmp_path, UTC_CSV),
        columns={"temperature": "airTemperature"},
        units={"temperature": "degC"},
    )
    with pytest.raises(ProviderConfigError, match="temperature"):
        FileWeatherProvider(cfg, TimePolicyConfig())


def test_naive_timestamps_need_a_timezone(tmp_path: Path):
    csv = "timestamp,airTemperature,humidity\n2024-01-15 00:00:00,5,80\n"
    provider = FileWeatherProvider(_cfg(_write(tmp_path, csv)), TimePolicyConfig())
    with pytest.raises(ProviderConfigError, match="timezone"):
        provider.get_history(MADRID, _window("2024-01-14", "2024-01-16"), ["temp_c"])


def test_naive_timestamps_localised_to_utc(tmp_path: Path):
    csv = "timestamp,airTemperature,humidity\n2024-07-15 12:00:00,25,40\n"
    cfg = _cfg(_write(tmp_path, csv), timezone="Europe/Madrid")
    df = FileWeatherProvider(cfg, TimePolicyConfig()).get_history(
        MADRID, _window("2024-07-15", "2024-07-16"), ["temp_c"]
    )
    assert df["ts_utc"].iloc[0] == pd.Timestamp("2024-07-15 10:00", tz="UTC")  # CEST = UTC+2


def test_ambiguous_dst_hour_follows_policy(tmp_path: Path):
    csv = (
        "timestamp,airTemperature,humidity\n"
        "2024-10-27 01:30:00,15,70\n2024-10-27 02:30:00,14,71\n"
        "2024-10-27 02:30:00,13,72\n2024-10-27 03:30:00,12,73\n"
    )
    cfg = _cfg(_write(tmp_path, csv), timezone="Europe/Madrid")
    window = _window("2024-10-26", "2024-10-29")
    with pytest.raises(TimeError):
        FileWeatherProvider(cfg, TimePolicyConfig()).get_history(MADRID, window, ["temp_c"])
    # The file repeats 02:30, so "infer"-style handling is not available; a flat policy would
    # duplicate instants. Declaring dst for all rows must still be reported as duplicates.
    with pytest.raises(ContractError, match="duplicated"):
        FileWeatherProvider(cfg, TimePolicyConfig(ambiguous="dst")).get_history(
            MADRID, window, ["temp_c"]
        )


def test_mixed_offset_timestamps_parse_to_utc(tmp_path: Path):
    csv = (
        "timestamp,airTemperature,humidity\n"
        "2024-03-31T01:30:00+01:00,10,70\n2024-03-31T03:30:00+02:00,11,71\n"
    )
    df = FileWeatherProvider(_cfg(_write(tmp_path, csv)), TimePolicyConfig()).get_history(
        MADRID, _window("2024-03-31", "2024-04-01"), ["temp_c"]
    )
    assert df["ts_utc"].tolist() == [
        pd.Timestamp("2024-03-31 00:30", tz="UTC"),
        pd.Timestamp("2024-03-31 01:30", tz="UTC"),
    ]


def test_mixture_of_naive_and_offset_timestamps_rejected(tmp_path: Path):
    csv = "timestamp,airTemperature,humidity\n2024-01-15T00:00:00Z,5,80\n2024-01-15 01:00:00,4,82\n"
    provider = FileWeatherProvider(_cfg(_write(tmp_path, csv), timezone="UTC"), TimePolicyConfig())
    with pytest.raises(ProviderError, match="mix"):
        provider.get_history(MADRID, _window("2024-01-14", "2024-01-16"), ["temp_c"])


def test_unmapped_variable_requested(tmp_path: Path):
    provider = FileWeatherProvider(_cfg(_write(tmp_path, UTC_CSV)), TimePolicyConfig())
    with pytest.raises(ProviderError, match="wind_ms"):
        provider.get_history(MADRID, _window("2024-01-15", "2024-01-16"), ["wind_ms"])


def test_implausible_value_is_rejected_not_repaired(tmp_path: Path):
    csv = "timestamp,airTemperature,humidity\n2024-01-15T00:00:00Z,5,150\n"
    provider = FileWeatherProvider(_cfg(_write(tmp_path, csv)), TimePolicyConfig())
    with pytest.raises(ContractError, match="rh_pct"):
        provider.get_history(MADRID, _window("2024-01-15", "2024-01-16"), ["rh_pct"])


def test_decimal_comma_and_semicolon(tmp_path: Path):
    csv = "timestamp;airTemperature;humidity\n2024-01-15T00:00:00Z;5,5;80\n"
    cfg = _cfg(_write(tmp_path, csv), delimiter=";", decimal=",")
    df = FileWeatherProvider(cfg, TimePolicyConfig()).get_history(
        MADRID, _window("2024-01-15", "2024-01-16"), ["temp_c"]
    )
    assert df["temp_c"].iloc[0] == 5.5


MULTI_CSV = (
    "timestamp,site,la,lo,airTemperature,humidity\n"
    "2024-01-15T00:00:00Z,mad,40.4168,-3.7038,5,80\n"
    "2024-01-15T00:00:00Z,bcn,41.3874,2.1686,9,60\n"
)


def _multi_cfg(tmp_path: Path) -> FileWeatherConfig:
    return _cfg(
        _write(tmp_path, MULTI_CSV),
        point_id_col="site",
        lat_col="la",
        lon_col="lo",
        lat=None,
        lon=None,
        point_id="unused",
    )


def test_multi_point_file_returns_nearest_point(tmp_path: Path):
    provider = FileWeatherProvider(_multi_cfg(tmp_path), TimePolicyConfig())
    near_bcn = Location(41.4, 2.2)
    df = provider.get_history(near_bcn, _window("2024-01-15", "2024-01-16"), ["temp_c"])
    assert df["weather_point_id"].tolist() == ["bcn"]
    assert df["temp_c"].tolist() == [9.0]


def test_points_catalogue(tmp_path: Path):
    points = FileWeatherProvider(_multi_cfg(tmp_path), TimePolicyConfig()).points()
    assert set(points["weather_point_id"]) == {"mad", "bcn"}
    assert set(points.columns) == {"weather_point_id", "lat", "lon"}


def test_point_location_required(tmp_path: Path):
    cfg = _cfg(_write(tmp_path, UTC_CSV), lat=None, lon=None)
    with pytest.raises(ProviderConfigError, match="lat"):
        FileWeatherProvider(cfg, TimePolicyConfig())


FORECAST_CSV = (
    "timestamp,issued,airTemperature,humidity\n"
    "2024-01-15T12:00:00Z,2024-01-15T06:00:00Z,5,80\n"
    "2024-01-15T13:00:00Z,2024-01-15T06:00:00Z,4,82\n"
)


def test_forecast_file_requires_issue_time_column(tmp_path: Path):
    cfg = _cfg(_write(tmp_path, FORECAST_CSV), product_type="forecast")
    with pytest.raises(ProviderConfigError, match="issued_at_col"):
        FileWeatherProvider(cfg, TimePolicyConfig())


def test_forecast_file_roundtrip_and_capability(tmp_path: Path):
    cfg = _cfg(_write(tmp_path, FORECAST_CSV), product_type="forecast", issued_at_col="issued")
    provider = FileWeatherProvider(cfg, TimePolicyConfig())
    window = _window("2024-01-15", "2024-01-16")
    df = provider.get_forecast(MADRID, window, ["temp_c"])
    assert df["product_type"].unique().tolist() == ["forecast"]
    assert df["issued_at_utc"].iloc[0] == pd.Timestamp("2024-01-15 06:00", tz="UTC")
    with pytest.raises(ProviderNotSupportedError):
        provider.get_history(MADRID, window, ["temp_c"])


def test_observation_file_has_no_forecast(tmp_path: Path):
    provider = FileWeatherProvider(_cfg(_write(tmp_path, UTC_CSV)), TimePolicyConfig())
    with pytest.raises(ProviderNotSupportedError):
        provider.get_forecast(MADRID, _window("2024-01-15", "2024-01-16"), ["temp_c"])


def test_missing_file_gives_clear_error(tmp_path: Path):
    provider = FileWeatherProvider(_cfg(tmp_path / "nope.csv"), TimePolicyConfig())
    with pytest.raises(ProviderError, match="not found"):
        provider.get_history(MADRID, _window("2024-01-15", "2024-01-16"), ["temp_c"])


def test_relative_path_resolved_against_base_dir(tmp_path: Path):
    _write(tmp_path, UTC_CSV, "weather.csv")
    cfg = _cfg(Path("weather.csv"))
    provider = FileWeatherProvider(cfg, TimePolicyConfig(), base_dir=tmp_path)
    df = provider.get_history(MADRID, _window("2024-01-15", "2024-01-16"), ["temp_c"])
    assert len(df) == 4
