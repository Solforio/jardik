"""Open-Meteo weather adapter (history from the archive API, forecasts from the forecast API).

LICENCE: the free API is for NON-COMMERCIAL use only; data is CC BY 4.0 (see LICENSES.md). A paid
plan is needed before any commercial use.

The adapter asks for UTC timestamps and SI units, then converts whatever units the response
declares into canonical ones. A unit that is missing or unknown stops the run.
"""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from jardik.core.config import OpenMeteoConfig
from jardik.core.interfaces import Location, TimeWindow
from jardik.core.logging import get_logger
from jardik.core.schemas import WEATHER_CANONICAL_UNITS
from jardik.core.units import convert
from jardik.providers.base import ProviderError
from jardik.providers.cache import ResponseCache
from jardik.providers.http import FetchJson, make_fetch_json
from jardik.providers.weather.common import build_weather_frame, check_requested, finalize

log = get_logger("providers.weather.open_meteo")

OPEN_METEO_VARIABLES: dict[str, str] = {
    "temp_c": "temperature_2m",
    "dewpoint_c": "dew_point_2m",
    "rh_pct": "relative_humidity_2m",
    "wind_ms": "wind_speed_10m",
    "wind_dir_deg": "wind_direction_10m",
    "pressure_hpa": "surface_pressure",
    "pressure_msl_hpa": "pressure_msl",
    "precip_mm": "precipitation",
    "cloud_pct": "cloud_cover",
    "ghi_wm2": "shortwave_radiation",
    "dni_wm2": "direct_normal_irradiance",
    "dhi_wm2": "diffuse_radiation",
}


class OpenMeteoProvider:
    name = "open_meteo"

    def __init__(
        self,
        config: OpenMeteoConfig,
        cache: ResponseCache,
        fetch_json: FetchJson | None = None,
    ) -> None:
        self._cfg = config
        self._cache = cache
        self._fetch_json = fetch_json or make_fetch_json(config.timeout_s)

    def _request(
        self,
        endpoint: str,
        url: str,
        location: Location,
        window: TimeWindow,
        variables: list[str],
        max_age: timedelta | None,
    ) -> tuple[dict[str, Any], pd.Timestamp]:
        wanted = check_requested(variables, list(OPEN_METEO_VARIABLES))
        start = window.start.tz_convert("UTC")
        last_instant = window.end.tz_convert("UTC") - pd.Timedelta(microseconds=1)
        params: dict[str, Any] = {
            "latitude": round(location.lat, 4),
            "longitude": round(location.lon, 4),
            "start_date": start.date().isoformat(),
            "end_date": last_instant.date().isoformat(),
            "hourly": ",".join(OPEN_METEO_VARIABLES[v] for v in wanted),
            "timezone": "UTC",
            "temperature_unit": "celsius",
            "wind_speed_unit": "ms",
            "precipitation_unit": "mm",
        }
        response = self._cache.get_or_fetch(
            self.name, endpoint, params, lambda: self._fetch_json(url, params), max_age=max_age
        )
        return response.payload, pd.Timestamp(response.fetched_at)

    def _to_frame(
        self,
        payload: dict[str, Any],
        variables: list[str],
        location: Location,
        window: TimeWindow,
        product_type: str,
        issued_at: pd.Timestamp | None,
    ) -> pd.DataFrame:
        hourly = payload.get("hourly") if isinstance(payload, dict) else None
        if not isinstance(hourly, dict) or "time" not in hourly:
            raise ProviderError("Open-Meteo response has no 'hourly' block with 'time'")
        if payload.get("utc_offset_seconds", 0) != 0:
            raise ProviderError("Open-Meteo answered in a non-UTC timezone")
        units = payload.get("hourly_units", {})

        times = pd.Series(pd.to_datetime(hourly["time"], utc=True))
        series: dict[str, pd.Series] = {}
        for variable in variables:
            api_name = OPEN_METEO_VARIABLES[variable]
            if api_name not in hourly:
                raise ProviderError(f"Open-Meteo response is missing '{api_name}'")
            unit = units.get(api_name)
            if not unit:
                raise ProviderError(f"Open-Meteo gave no unit for '{api_name}'; refusing to guess")
            values = pd.Series(np.asarray(hourly[api_name], dtype=float))
            series[variable] = convert(values, unit, WEATHER_CANONICAL_UNITS[variable])

        lat = float(payload.get("latitude", location.lat))
        lon = float(payload.get("longitude", location.lon))
        frame = build_weather_frame(
            point_id=f"open_meteo:{lat:.4f}:{lon:.4f}",
            ts_utc=times,
            source=self.name,
            product_type=product_type,
            issued_at=issued_at,
            series=series,
        )
        return finalize(frame, window)

    def get_history(
        self, location: Location, window: TimeWindow, variables: list[str]
    ) -> pd.DataFrame:
        wanted = check_requested(variables, list(OPEN_METEO_VARIABLES))
        payload, _ = self._request(
            "archive", self._cfg.archive_url, location, window, wanted, max_age=None
        )
        return self._to_frame(payload, wanted, location, window, "reanalysis", None)

    def get_forecast(
        self, location: Location, window: TimeWindow, variables: list[str]
    ) -> pd.DataFrame:
        wanted = check_requested(variables, list(OPEN_METEO_VARIABLES))
        ttl = timedelta(hours=self._cfg.forecast_ttl_hours)
        payload, fetched_at = self._request(
            "forecast", self._cfg.forecast_url, location, window, wanted, max_age=ttl
        )
        # The API does not expose the model run time: the request time, floored to the hour,
        # is the best honest approximation of "what was known when".
        issued = fetched_at.tz_convert("UTC").floor("h")
        return self._to_frame(payload, wanted, location, window, "forecast", issued)
