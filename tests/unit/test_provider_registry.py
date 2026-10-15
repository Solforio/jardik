from pathlib import Path

import pandas as pd
import pytest

from jardik.core.config import AppConfig, ProvidersConfig
from jardik.core.interfaces import Location, TimeWindow
from jardik.core.paths import JardikPaths
from jardik.providers.base import ProviderConfigError
from jardik.providers.registry import available_weather_providers, build_weather_provider
from jardik.providers.weather.file import FileWeatherProvider
from jardik.providers.weather.open_meteo import OpenMeteoProvider

CSV = "timestamp,t\n2024-01-15T00:00:00Z,5.0\n2024-01-15T01:00:00Z,4.0\n"
WINDOW = TimeWindow(pd.Timestamp("2024-01-15", tz="UTC"), pd.Timestamp("2024-01-16", tz="UTC"))
MADRID = Location(40.4168, -3.7038)


def _fake_api(url: str, params: dict) -> dict:
    return {
        "latitude": 40.4,
        "longitude": -3.7,
        "hourly_units": {"temperature_2m": "°C"},
        "hourly": {
            "time": ["2024-01-15T00:00", "2024-01-15T01:00"],
            "temperature_2m": [6.0, 7.0],
        },
    }


def _config(tmp_path: Path, weather: str) -> AppConfig:
    (tmp_path / "w.csv").write_text(CSV, encoding="utf-8")
    return AppConfig.model_validate(
        {
            "providers": {
                "weather": weather,
                "file_weather": {
                    "path": "w.csv",
                    "columns": {"temp_c": "t"},
                    "units": {"temp_c": "degC"},
                    "lat": 40.4,
                    "lon": -3.7,
                },
            }
        }
    )


def test_available_names():
    assert available_weather_providers() == ["file", "open_meteo"]


def test_provider_is_swapped_by_config_only(tmp_path: Path):
    paths = JardikPaths.from_root(tmp_path)
    from_file = build_weather_provider(_config(tmp_path, "file"), paths, fetch_json=_fake_api)
    from_api = build_weather_provider(_config(tmp_path, "open_meteo"), paths, fetch_json=_fake_api)
    assert isinstance(from_file, FileWeatherProvider)
    assert isinstance(from_api, OpenMeteoProvider)
    # Same calling code, same frame shape, different source.
    a = from_file.get_history(MADRID, WINDOW, ["temp_c"])
    b = from_api.get_history(MADRID, WINDOW, ["temp_c"])
    assert list(a.columns) == list(b.columns) or set(a.columns) >= {"ts_utc", "temp_c"}
    assert a["temp_c"].tolist() == [5.0, 4.0]
    assert b["temp_c"].tolist() == [6.0, 7.0]


def test_cache_directory_comes_from_central_paths(tmp_path: Path):
    paths = JardikPaths.from_root(tmp_path)
    provider = build_weather_provider(_config(tmp_path, "open_meteo"), paths, fetch_json=_fake_api)
    provider.get_history(MADRID, WINDOW, ["temp_c"])
    assert any(paths.provider_cache_dir.rglob("*.json"))


def test_cache_can_be_disabled_by_config(tmp_path: Path):
    paths = JardikPaths.from_root(tmp_path)
    cfg = _config(tmp_path, "open_meteo")
    cfg = cfg.model_copy(
        update={"providers": cfg.providers.model_copy(update={"cache_enabled": False})}
    )
    build_weather_provider(cfg, paths, fetch_json=_fake_api).get_history(MADRID, WINDOW, ["temp_c"])
    assert not any(paths.provider_cache_dir.rglob("*.json"))


def test_unknown_provider_name(tmp_path: Path):
    cfg = AppConfig(providers=ProvidersConfig(weather="nope"))
    with pytest.raises(ProviderConfigError, match="file, open_meteo"):
        build_weather_provider(cfg, JardikPaths.from_root(tmp_path))


def test_file_provider_requires_its_section(tmp_path: Path):
    with pytest.raises(ProviderConfigError, match="file_weather"):
        build_weather_provider(AppConfig(), JardikPaths.from_root(tmp_path))
