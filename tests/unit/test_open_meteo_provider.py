from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from jardik.core.config import OpenMeteoConfig
from jardik.core.interfaces import Location, TimeWindow, WeatherProvider
from jardik.core.units import UnknownUnitError
from jardik.providers.base import ProviderError
from jardik.providers.cache import ResponseCache
from jardik.providers.weather.open_meteo import OPEN_METEO_VARIABLES, OpenMeteoProvider

MADRID = Location(40.4168, -3.7038)


def _window(start: str, end: str) -> TimeWindow:
    return TimeWindow(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"))


class FakeApi:
    """Records calls and returns an Open-Meteo shaped payload."""

    def __init__(self, wind_unit: str = "m/s", temp_unit: str = "°C") -> None:
        self.calls: list[tuple[str, dict]] = []
        self.wind_unit, self.temp_unit = wind_unit, temp_unit

    def __call__(self, url: str, params: dict) -> dict:
        self.calls.append((url, params))
        hours = pd.date_range(
            f"{params['start_date']} 00:00", f"{params['end_date']} 23:00", freq="h"
        )
        names = params["hourly"].split(",")
        units = {
            "temperature_2m": self.temp_unit,
            "relative_humidity_2m": "%",
            "wind_speed_10m": self.wind_unit,
            "shortwave_radiation": "W/m²",
            "wind_direction_10m": "°",
            "surface_pressure": "hPa",
            "precipitation": "mm",
            "cloud_cover": "%",
        }
        hourly = {"time": [t.strftime("%Y-%m-%dT%H:%M") for t in hours]}
        for name in names:
            hourly[name] = [float(i % 24) for i in range(len(hours))]  # stays physically plausible
        return {
            "latitude": 40.4,
            "longitude": -3.7,
            "hourly_units": {"time": "iso8601", **{n: units.get(n, "%") for n in names}},
            "hourly": hourly,
        }


def _provider(tmp_path: Path, api: FakeApi, now: datetime | None = None) -> OpenMeteoProvider:
    clock = (lambda: now) if now else None
    cache = (
        ResponseCache(tmp_path / "cache", clock=clock)
        if clock
        else ResponseCache(tmp_path / "cache")
    )
    return OpenMeteoProvider(OpenMeteoConfig(), cache, fetch_json=api)


def test_implements_protocol(tmp_path: Path):
    assert isinstance(_provider(tmp_path, FakeApi()), WeatherProvider)


def test_history_request_parameters_and_frame(tmp_path: Path):
    api = FakeApi()
    df = _provider(tmp_path, api).get_history(
        MADRID, _window("2024-01-15 00:00", "2024-01-16 00:00"), ["temp_c", "rh_pct"]
    )
    url, params = api.calls[0]
    assert url == "https://archive-api.open-meteo.com/v1/archive"
    assert params["hourly"] == "temperature_2m,relative_humidity_2m"
    assert params["timezone"] == "UTC"
    assert params["wind_speed_unit"] == "ms"
    assert (params["start_date"], params["end_date"]) == ("2024-01-15", "2024-01-15")
    assert len(df) == 24
    assert df["product_type"].unique().tolist() == ["reanalysis"]
    assert df["source"].unique().tolist() == ["open_meteo"]
    assert str(df["ts_utc"].dt.tz) == "UTC"
    assert "issued_at_utc" not in df.columns or df["issued_at_utc"].isna().all()
    assert df["weather_point_id"].iloc[0] == "open_meteo:40.4000:-3.7000"


def test_window_is_half_open_and_clipped(tmp_path: Path):
    api = FakeApi()
    df = _provider(tmp_path, api).get_history(
        MADRID, _window("2024-01-15 06:00", "2024-01-15 09:00"), ["temp_c"]
    )
    assert df["ts_utc"].tolist() == list(
        pd.date_range("2024-01-15 06:00", periods=3, freq="h", tz="UTC")
    )


def test_multi_day_window_dates(tmp_path: Path):
    api = FakeApi()
    _provider(tmp_path, api).get_history(
        MADRID, _window("2024-01-15 00:00", "2024-01-18 00:00"), ["temp_c"]
    )
    assert (api.calls[0][1]["start_date"], api.calls[0][1]["end_date"]) == (
        "2024-01-15",
        "2024-01-17",
    )


def test_identical_request_hits_cache_even_with_new_provider_instance(tmp_path: Path):
    api = FakeApi()
    window = _window("2024-01-15", "2024-01-16")
    _provider(tmp_path, api).get_history(MADRID, window, ["temp_c"])
    _provider(tmp_path, api).get_history(MADRID, window, ["temp_c"])
    assert len(api.calls) == 1


def test_response_units_are_converted_not_assumed(tmp_path: Path):
    # The API answers in km/h and degF even though we asked for m/s: values must be converted.
    api = FakeApi(wind_unit="km/h", temp_unit="°F")
    df = _provider(tmp_path, api).get_history(
        MADRID, _window("2024-01-15 00:00", "2024-01-15 03:00"), ["wind_ms", "temp_c"]
    )
    assert df["wind_ms"].tolist() == pytest.approx([0.0, 1 / 3.6, 2 / 3.6])
    assert df["temp_c"].tolist() == pytest.approx([(v - 32) * 5 / 9 for v in (0.0, 1.0, 2.0)])


def test_unknown_response_unit_raises(tmp_path: Path):
    api = FakeApi(wind_unit="furlongs/fortnight")
    with pytest.raises(UnknownUnitError):
        _provider(tmp_path, api).get_history(
            MADRID, _window("2024-01-15", "2024-01-16"), ["wind_ms"]
        )


def test_unsupported_variable(tmp_path: Path):
    with pytest.raises(ProviderError, match="not_a_var"):
        _provider(tmp_path, FakeApi()).get_history(
            MADRID, _window("2024-01-15", "2024-01-16"), ["not_a_var"]
        )


def test_null_values_become_nan(tmp_path: Path):
    def api(url: str, params: dict) -> dict:
        return {
            "latitude": 40.4,
            "longitude": -3.7,
            "hourly_units": {"temperature_2m": "°C"},
            "hourly": {
                "time": ["2024-01-15T00:00", "2024-01-15T01:00"],
                "temperature_2m": [1.0, None],
            },
        }

    cache = ResponseCache(tmp_path / "cache")
    df = OpenMeteoProvider(OpenMeteoConfig(), cache, fetch_json=api).get_history(
        MADRID, _window("2024-01-15", "2024-01-16"), ["temp_c"]
    )
    assert np.isnan(df["temp_c"].iloc[1])


def test_malformed_response_raises(tmp_path: Path):
    cache = ResponseCache(tmp_path / "cache")
    provider = OpenMeteoProvider(OpenMeteoConfig(), cache, fetch_json=lambda u, p: {"error": True})
    with pytest.raises(ProviderError, match="hourly"):
        provider.get_history(MADRID, _window("2024-01-15", "2024-01-16"), ["temp_c"])


def test_forecast_uses_forecast_endpoint_and_stamps_issue_time(tmp_path: Path):
    api = FakeApi()
    now = datetime(2024, 1, 15, 9, 41, tzinfo=UTC)
    df = _provider(tmp_path, api, now=now).get_forecast(
        MADRID, _window("2024-01-15 12:00", "2024-01-16 00:00"), ["temp_c"]
    )
    assert api.calls[0][0] == "https://api.open-meteo.com/v1/forecast"
    assert df["product_type"].unique().tolist() == ["forecast"]
    # Issue time is the request time floored to the hour (the API does not expose the model run).
    assert df["issued_at_utc"].unique().tolist() == [pd.Timestamp("2024-01-15 09:00", tz="UTC")]


def test_forecast_cache_expires_history_does_not(tmp_path: Path):
    api = FakeApi()
    clock_now = [datetime(2024, 1, 15, 9, 0, tzinfo=UTC)]
    cache = ResponseCache(tmp_path / "cache", clock=lambda: clock_now[0])
    provider = OpenMeteoProvider(OpenMeteoConfig(forecast_ttl_hours=3), cache, fetch_json=api)
    window = _window("2024-01-15 12:00", "2024-01-16 00:00")
    provider.get_forecast(MADRID, window, ["temp_c"])
    clock_now[0] = datetime(2024, 1, 15, 11, 0, tzinfo=UTC)
    provider.get_forecast(MADRID, window, ["temp_c"])
    assert len(api.calls) == 1
    clock_now[0] = datetime(2024, 1, 15, 13, 0, tzinfo=UTC)
    provider.get_forecast(MADRID, window, ["temp_c"])
    assert len(api.calls) == 2


def test_every_mapped_variable_is_a_canonical_weather_variable():
    from jardik.core.schemas import WEATHER_VARIABLES

    assert set(OPEN_METEO_VARIABLES) <= set(WEATHER_VARIABLES)
