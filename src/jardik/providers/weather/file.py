"""Weather adapter that reads a local CSV file.

Used for datasets that ship their own weather (for example Building Data Genome 2) and for
client-supplied weather. Everything the file does not say explicitly is refused: each mapped
column must declare its unit, naive timestamps need a declared timezone and DST policy.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from jardik.core.config import FileWeatherConfig, TimePolicyConfig
from jardik.core.geo import nearest_point
from jardik.core.interfaces import Location, TimeWindow
from jardik.core.logging import get_logger
from jardik.core.schemas import WEATHER_CANONICAL_UNITS, WEATHER_VARIABLES
from jardik.core.timeutils import localize_naive
from jardik.core.units import convert
from jardik.providers.base import ProviderConfigError, ProviderError, ProviderNotSupportedError
from jardik.providers.weather.common import build_weather_frame, check_requested, finalize

log = get_logger("providers.weather.file")

_OFFSET_SUFFIX = r"(?:Z|[+-]\d{2}:?\d{2})$"


class FileWeatherProvider:
    name = "file"

    def __init__(
        self, config: FileWeatherConfig, time_policy: TimePolicyConfig, base_dir: Path | None = None
    ) -> None:
        self._cfg = config
        self._policy = time_policy
        path = config.path
        self._path = path if path.is_absolute() or base_dir is None else base_dir / path
        self._data: pd.DataFrame | None = None
        self._validate_config()

    # --- configuration checks ------------------------------------------------------------
    def _validate_config(self) -> None:
        cfg = self._cfg
        unknown = [v for v in cfg.columns if v not in WEATHER_VARIABLES]
        if unknown:
            raise ProviderConfigError(
                f"columns maps unknown canonical variable(s) {unknown}. "
                f"Valid names: {sorted(WEATHER_VARIABLES)}"
            )
        no_unit = [v for v in cfg.columns if v not in cfg.units]
        if no_unit:
            raise ProviderConfigError(
                f"No unit declared for {no_unit}. Declare units for every mapped column; "
                "units are never guessed."
            )
        if cfg.point_id_col is None:
            if cfg.lat is None or cfg.lon is None:
                raise ProviderConfigError(
                    "Set lat and lon, or point_id_col with lat_col and lon_col"
                )
        elif cfg.lat_col is None or cfg.lon_col is None:
            raise ProviderConfigError("point_id_col requires lat_col and lon_col")
        if cfg.product_type == "forecast" and cfg.issued_at_col is None:
            raise ProviderConfigError(
                "Forecast files need issued_at_col (when each forecast was issued)"
            )

    # --- reading --------------------------------------------------------------------------
    def _parse_time(self, raw: pd.Series, what: str) -> pd.Series:
        text = raw.astype(str).str.strip()
        has_offset = text.str.contains(_OFFSET_SUFFIX, regex=True)
        if has_offset.all():
            return pd.to_datetime(text, format="ISO8601", utc=True)
        if has_offset.any():
            raise ProviderError(f"{what}: file mixes timestamps with and without a UTC offset")
        if self._cfg.timezone is None:
            raise ProviderConfigError(
                f"{what} are naive: set 'timezone' (IANA name) in the file weather config"
            )
        naive = pd.DatetimeIndex(pd.to_datetime(text, format="ISO8601"))
        local = localize_naive(
            naive, self._cfg.timezone, self._policy.ambiguous, self._policy.nonexistent
        )
        return pd.Series(local.tz_convert("UTC"), index=raw.index)

    def _load(self) -> pd.DataFrame:
        if self._data is not None:
            return self._data
        cfg = self._cfg
        if not self._path.is_file():
            raise ProviderError(f"Weather file not found: {self._path}")

        wanted = {cfg.timestamp_col, *cfg.columns.values()}
        wanted |= {c for c in (cfg.issued_at_col, cfg.point_id_col, cfg.lat_col, cfg.lon_col) if c}
        raw = pd.read_csv(
            self._path,
            sep=cfg.delimiter,
            decimal=cfg.decimal,
            usecols=lambda col: col in wanted,
            dtype={
                cfg.timestamp_col: str,
                **({cfg.issued_at_col: str} if cfg.issued_at_col else {}),
            },
        )
        absent = wanted - set(raw.columns)
        if absent:
            raise ProviderError(f"Column(s) {sorted(absent)} not found in {self._path.name}")

        out = pd.DataFrame(index=raw.index)
        out["ts_utc"] = self._parse_time(raw[cfg.timestamp_col], "timestamps")
        if cfg.issued_at_col:
            out["issued_at_utc"] = self._parse_time(raw[cfg.issued_at_col], "issue times")
        if cfg.point_id_col:
            out["weather_point_id"] = raw[cfg.point_id_col].astype(str)
            out["lat"] = pd.to_numeric(raw[cfg.lat_col])
            out["lon"] = pd.to_numeric(raw[cfg.lon_col])
        else:
            out["weather_point_id"] = cfg.point_id
            out["lat"], out["lon"] = cfg.lat, cfg.lon

        for variable, column in cfg.columns.items():
            try:
                values = pd.to_numeric(raw[column], errors="raise")
            except (ValueError, TypeError) as exc:
                raise ProviderError(
                    f"Column '{column}' contains non-numeric values: {exc}"
                ) from exc
            out[variable] = convert(values, cfg.units[variable], WEATHER_CANONICAL_UNITS[variable])

        out = out.dropna(subset=["ts_utc"])
        self._data = out
        return out

    # --- public API -----------------------------------------------------------------------
    def points(self) -> pd.DataFrame:
        """Distinct weather points in the file: weather_point_id, lat, lon."""
        data = self._load()
        points = data[["weather_point_id", "lat", "lon"]].drop_duplicates()
        if points["weather_point_id"].duplicated().any():
            raise ProviderError("A weather point has more than one coordinate pair in the file")
        return points.reset_index(drop=True)

    def _read(
        self, location: Location, window: TimeWindow, variables: list[str], product: str
    ) -> pd.DataFrame:
        wanted = check_requested(variables, list(self._cfg.columns))
        data = self._load()
        point, distance = nearest_point(location.lat, location.lon, self.points())
        log.debug("nearest point %s at %.1f km", point["weather_point_id"], distance)
        subset = data[data["weather_point_id"] == point["weather_point_id"]]
        frame = build_weather_frame(
            point_id=subset["weather_point_id"],
            ts_utc=subset["ts_utc"],
            source=self._cfg.source,
            product_type=product,
            issued_at=subset["issued_at_utc"] if "issued_at_utc" in subset else None,
            series={v: subset[v] for v in wanted},
        )
        return finalize(frame, window)

    def get_history(
        self, location: Location, window: TimeWindow, variables: list[str]
    ) -> pd.DataFrame:
        if self._cfg.product_type == "forecast":
            raise ProviderNotSupportedError("This file holds forecasts: use get_forecast")
        return self._read(location, window, variables, self._cfg.product_type)

    def get_forecast(
        self, location: Location, window: TimeWindow, variables: list[str]
    ) -> pd.DataFrame:
        if self._cfg.product_type != "forecast":
            raise ProviderNotSupportedError("This file does not hold forecasts")
        return self._read(location, window, variables, "forecast")
