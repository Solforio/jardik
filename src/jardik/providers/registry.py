"""Chooses the provider implementation from configuration (never hard-coded in models)."""

from __future__ import annotations

from collections.abc import Callable

from jardik.core.config import AppConfig
from jardik.core.interfaces import WeatherProvider
from jardik.core.paths import JardikPaths
from jardik.providers.base import ProviderConfigError
from jardik.providers.cache import ResponseCache
from jardik.providers.http import FetchJson
from jardik.providers.weather.file import FileWeatherProvider
from jardik.providers.weather.open_meteo import OpenMeteoProvider

WeatherFactory = Callable[
    [AppConfig, JardikPaths, ResponseCache, FetchJson | None], WeatherProvider
]


def _build_file(
    cfg: AppConfig, paths: JardikPaths, cache: ResponseCache, fetch: FetchJson | None
) -> WeatherProvider:
    section = cfg.providers.file_weather
    if section is None:
        raise ProviderConfigError(
            "providers.weather is 'file' but providers.file_weather is missing"
        )
    return FileWeatherProvider(section, cfg.time_policy, base_dir=paths.root)


def _build_open_meteo(
    cfg: AppConfig, paths: JardikPaths, cache: ResponseCache, fetch: FetchJson | None
) -> WeatherProvider:
    return OpenMeteoProvider(cfg.providers.open_meteo, cache, fetch_json=fetch)


_WEATHER_FACTORIES: dict[str, WeatherFactory] = {
    "file": _build_file,
    "open_meteo": _build_open_meteo,
}


def available_weather_providers() -> list[str]:
    return sorted(_WEATHER_FACTORIES)


def build_weather_provider(
    cfg: AppConfig, paths: JardikPaths, fetch_json: FetchJson | None = None
) -> WeatherProvider:
    """Create the weather provider named in `cfg.providers.weather`.

    `fetch_json` lets tests (or offline runs) replace the network call.
    """
    name = cfg.providers.weather
    factory = _WEATHER_FACTORIES.get(name)
    if factory is None:
        available = ", ".join(available_weather_providers())
        raise ProviderConfigError(f"Unknown weather provider {name!r}. Available: {available}")
    cache = ResponseCache(paths.provider_cache_dir, enabled=cfg.providers.cache_enabled)
    return factory(cfg, paths, cache, fetch_json)
