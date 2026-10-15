"""Building Data Genome Project 2 (BDG2) adapter.

Facts about the dataset that shape this code (verified against the data, see docs/DECISIONS.md):
  * timestamps are LOCAL STANDARD TIME with daylight saving ignored: one continuous hourly
    clock of 17,544 rows (2016-2017), so it never repeats or skips an hour;
  * coordinates are per site and city-level; four sites have none; one site has a longitude with
    the wrong sign (corrected only through an explicit, recorded override);
  * weather is per site, in the same clock, with ISD conventions (-1 = trace precipitation,
    cloud cover in oktas with 9 = sky obscured).
Licence: CC BY-SA 4.0. Citation: Miller et al., Scientific Data 7, 368 (2020).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from jardik.core.config import TimePolicyConfig
from jardik.core.interfaces import TimeWindow
from jardik.core.logging import get_logger
from jardik.core.schemas import (
    WEATHER_CANONICAL_UNITS,
    WEATHER_VARIABLES,
    UsageCategory,
    derive_location_quality,
)
from jardik.core.timeutils import localize_naive, standard_time_to_utc, utc_grid, validate_timezone
from jardik.core.units import convert
from jardik.dataset_manager.adapters.base import DatasetError, wide_to_load_frame
from jardik.providers.weather.common import build_weather_frame, finalize

log = get_logger("dataset_manager.bdg2")

DATASET_ID = "bdg2"
PREFIX = f"{DATASET_ID}:"
CITATION = (
    "Miller, C., Kathirgamanathan, A., Picchetti, B. et al. The Building Data Genome Project 2, "
    "energy meter data from the ASHRAE Great Energy Predictor III competition. "
    "Scientific Data 7, 368 (2020). https://doi.org/10.1038/s41597-020-00712-x"
)
SOURCE_URL = "https://github.com/buds-lab/building-data-genome-project-2"

# canonical variable -> (BDG2 column, unit used in the file)
_WEATHER_COLUMNS: dict[str, tuple[str, str]] = {
    "temp_c": ("airTemperature", "degC"),
    "dewpoint_c": ("dewTemperature", "degC"),
    "wind_ms": ("windSpeed", "m/s"),
    "wind_dir_deg": ("windDirection", "deg"),
    "pressure_msl_hpa": ("seaLvlPressure", "hPa"),
    "precip_mm": ("precipDepth1HR", "mm"),
    "cloud_pct": ("cloudCoverage", "oktas"),
}
_OKTA_TO_PCT = 100.0 / 8.0
_OBSCURED_SKY_OKTA = 9.0


class SiteOverride(BaseModel):
    """A declared correction of a site's coordinates, always recorded in the manifest."""

    model_config = ConfigDict(extra="forbid")

    lat: float | None = None
    lng: float | None = None
    reason: str = Field(min_length=10)


class Bdg2Config(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data_dir: Path
    metadata_file: str = "metadata.csv"
    weather_file: str = "weather.csv"
    electricity_file: str = "electricity_cleaned.csv"
    timestamp_clock: Literal["local_standard", "local_civil"] = "local_standard"
    time_policy: TimePolicyConfig = Field(default_factory=TimePolicyConfig)
    usage_map: dict[str, str] = Field(default_factory=dict)
    usage_sub_map: dict[str, str] = Field(default_factory=dict)
    site_overrides: dict[str, SiteOverride] = Field(default_factory=dict)
    bbox_by_timezone: dict[str, dict[str, float]] = Field(default_factory=dict)

    @field_validator("usage_map", "usage_sub_map")
    @classmethod
    def _check_categories(cls, value: dict[str, str]) -> dict[str, str]:
        valid = {u.value for u in UsageCategory}
        bad = {v for v in value.values() if v not in valid}
        if bad:
            raise ValueError(f"unknown usage categories {sorted(bad)}; valid: {sorted(valid)}")
        return value


class Bdg2Adapter:
    dataset_id = DATASET_ID

    def __init__(self, config: Bdg2Config) -> None:
        self.config = config
        self.corrections: list[dict[str, Any]] = []
        self.weather_screening: dict[str, dict[str, int]] = {}
        self._meta_raw: pd.DataFrame | None = None
        self._catalogue: pd.DataFrame | None = None
        self._electricity: pd.DataFrame | None = None
        self._weather: pd.DataFrame | None = None
        self._unmapped: set[str] = set()
        self._hashes: dict[str, str] = {}

    # --- file access ---------------------------------------------------------------------
    def _path(self, name: str) -> Path:
        path = self.config.data_dir / name
        if not path.is_file():
            raise DatasetError(
                f"BDG2 file not found: {path}. Run `jardik dataset fetch bdg2` to download it."
            )
        return path

    def _read_metadata(self) -> pd.DataFrame:
        if self._meta_raw is None:
            self._meta_raw = pd.read_csv(self._path(self.config.metadata_file))
        return self._meta_raw

    def _read_electricity(self) -> pd.DataFrame:
        if self._electricity is None:
            path = self._path(self.config.electricity_file)
            frame = pd.read_csv(path, index_col="timestamp", parse_dates=["timestamp"])
            frame.index = pd.DatetimeIndex(frame.index)
            self._electricity = frame
        return self._electricity

    def _read_weather(self) -> pd.DataFrame:
        if self._weather is None:
            frame = pd.read_csv(
                self._path(self.config.weather_file),
                parse_dates=["timestamp"],
                dtype={"site_id": str},
            )
            self._weather = frame
        return self._weather

    # --- catalogue -----------------------------------------------------------------------
    def _site_overrides_applied(self, meta: pd.DataFrame) -> pd.DataFrame:
        meta = meta.copy()
        self.corrections = []
        for site, override in self.config.site_overrides.items():
            rows = meta["site_id"] == site
            if not rows.any():
                continue
            for field, column in (("lat", "lat"), ("lng", "lng")):
                new = getattr(override, field)
                if new is None:
                    continue
                old = meta.loc[rows, column].iloc[0]
                meta.loc[rows, column] = new
                self.corrections.append(
                    {
                        "site_id": site,
                        "field": field,
                        "old": None if pd.isna(old) else float(old),
                        "new": float(new),
                        "reason": override.reason,
                        "buildings_affected": int(rows.sum()),
                    }
                )
        return meta

    def _map_usage(self, primary: str, sub: str | None) -> str:
        if sub is not None and f"{primary}|{sub}" in self.config.usage_sub_map:
            return self.config.usage_sub_map[f"{primary}|{sub}"]
        if primary in self.config.usage_map:
            return self.config.usage_map[primary]
        self._unmapped.add(primary)
        return UsageCategory.OTHER.value

    def discover(self) -> pd.DataFrame:
        if self._catalogue is not None:
            return self._catalogue.copy()

        meta = self._site_overrides_applied(self._read_metadata())
        electricity_columns = set(self._read_electricity().columns)
        meta = meta[meta["building_id"].isin(electricity_columns)].reset_index(drop=True)

        self._unmapped = set()
        primary = meta["primaryspaceusage"].fillna("(missing)").astype(str).str.strip()
        sub = meta["sub_primaryspaceusage"].astype("string").str.strip()
        usage = [
            self._map_usage(p, None if pd.isna(s) else str(s))
            for p, s in zip(primary, sub, strict=True)
        ]
        year = pd.to_numeric(meta["yearbuilt"], errors="coerce")
        year = year.where(year.between(1800, 2100)).astype("Int64")
        lat = pd.to_numeric(meta["lat"], errors="coerce")
        lon = pd.to_numeric(meta["lng"], errors="coerce")

        catalogue = pd.DataFrame(
            {
                "building_id": PREFIX + meta["building_id"].astype(str),
                "dataset_id": DATASET_ID,
                "site_id": meta["site_id"].astype(str),
                "usage_category": usage,
                "usage_raw": primary,
                "usage_sub_raw": sub.astype(object).where(sub.notna(), None),
                "country": None,
                "region": None,
                "province": None,
                "municipality": None,
                "postal_code": None,
                "lat": lat,
                "lon": lon,
                "timezone": meta["timezone"].astype(str),
                "floor_area_m2": pd.to_numeric(meta["sqm"], errors="coerce"),
                "contracted_power_kw": np.nan,
                "year_built": year,
                "has_pv": pd.Series(
                    [True if v == "Yes" else None for v in meta["solar"]], dtype="object"
                ),
                "has_battery": None,
                "has_ac": None,
                "has_heating": None,
                "heating_fuel": None,
                "licence_id": DATASET_ID,
                "weather_link": meta["site_id"].astype(str),
            }
        )
        for zone in catalogue["timezone"].unique():
            validate_timezone(zone)
        catalogue["location_quality"] = [
            derive_location_quality(
                None if pd.isna(r.lat) else r.lat,
                None if pd.isna(r.lon) else r.lon,
                None,
                None,
                None,
                None,
            ).value
            for r in catalogue.itertuples(index=False)
        ]
        self._catalogue = catalogue
        return catalogue.copy()

    def load_metadata(self, building_ids: list[str] | None = None) -> pd.DataFrame:
        catalogue = self.discover()
        if building_ids is None:
            return catalogue
        return catalogue[catalogue["building_id"].isin(building_ids)].reset_index(drop=True)

    def unmapped_usage_labels(self) -> set[str]:
        """Usage labels in the data that the configured usage_map does not cover."""
        self.discover()
        return set(self._unmapped)

    # --- time series ---------------------------------------------------------------------
    def _to_utc(self, naive: pd.DatetimeIndex, zone: str) -> pd.DatetimeIndex:
        if self.config.timestamp_clock == "local_standard":
            return standard_time_to_utc(naive, zone)
        policy = self.config.time_policy
        local = localize_naive(naive, zone, policy.ambiguous, policy.nonexistent)
        return local.tz_convert("UTC")

    def load_wide(self, building_ids: list[str], window: TimeWindow) -> pd.DataFrame:
        catalogue = self.discover().set_index("building_id")
        unknown = [b for b in building_ids if b not in catalogue.index]
        if unknown:
            raise DatasetError(f"Unknown building(s) for dataset {DATASET_ID}: {unknown[:5]}")

        grid = utc_grid(window.start, window.end, "h", inclusive="left")
        electricity = self._read_electricity()
        blocks: list[pd.DataFrame] = []
        zones = catalogue.loc[building_ids, "timezone"]
        for zone, members in zones.groupby(zones):
            ids = list(members.index)
            raw_columns = [b.removeprefix(PREFIX) for b in ids]
            block = electricity[raw_columns].copy()
            block.index = self._to_utc(block.index, str(zone))
            block = block[block.index.notna() & ~block.index.duplicated(keep="first")]
            block.columns = ids
            blocks.append(block.reindex(grid))
        if not blocks:
            return pd.DataFrame(index=grid.rename("ts_utc"))
        wide = pd.concat(blocks, axis=1)[list(building_ids)]
        wide.index.name = "ts_utc"
        return wide

    def load_timeseries(self, building_ids: list[str], window: TimeWindow) -> pd.DataFrame:
        wide = self.load_wide(building_ids, window)
        zones = self.discover().set_index("building_id")["timezone"].to_dict()
        return wide_to_load_frame(wide, zones)

    # --- weather -------------------------------------------------------------------------
    def _site_timezones(self) -> dict[str, str]:
        meta = self._read_metadata()
        return meta.groupby("site_id")["timezone"].first().astype(str).to_dict()

    def weather_points(self) -> pd.DataFrame:
        meta = self._site_overrides_applied(self._read_metadata())
        sites = sorted(self._read_weather()["site_id"].unique())
        located = (
            meta.assign(
                lat=pd.to_numeric(meta["lat"], errors="coerce"),
                lng=pd.to_numeric(meta["lng"], errors="coerce"),
            )
            .groupby("site_id")[["lat", "lng"]]
            .median()
        )
        return pd.DataFrame(
            {
                "weather_point_id": sites,
                "lat": [located["lat"].get(s, np.nan) for s in sites],
                "lon": [located["lng"].get(s, np.nan) for s in sites],
            }
        )

    def load_weather(
        self, point_ids: list[str], window: TimeWindow, variables: list[str]
    ) -> pd.DataFrame:
        missing_vars = [v for v in variables if v not in _WEATHER_COLUMNS]
        if missing_vars:
            raise DatasetError(
                f"BDG2 weather does not provide {missing_vars}. "
                f"Available: {sorted(_WEATHER_COLUMNS)}"
            )
        weather = self._read_weather()
        known_sites = set(weather["site_id"].unique())
        unknown = [p for p in point_ids if p not in known_sites]
        if unknown:
            raise DatasetError(f"Unknown weather point(s) {unknown}; sites: {sorted(known_sites)}")
        zones = self._site_timezones()

        self.weather_screening = {v: {} for v in variables}
        frames: list[pd.DataFrame] = []
        for site in point_ids:
            rows = weather[weather["site_id"] == site]
            ts_utc = pd.Series(
                self._to_utc(pd.DatetimeIndex(rows["timestamp"]), zones[site]), index=rows.index
            )
            series = {v: self._convert_variable(v, rows[_WEATHER_COLUMNS[v][0]]) for v in variables}
            frame = build_weather_frame(
                point_id=site,
                ts_utc=ts_utc,
                source=DATASET_ID,
                product_type="observation",
                issued_at=None,
                series=series,
            )
            inside = (frame["ts_utc"] >= window.start) & (frame["ts_utc"] < window.end)
            frames.append(self._screen_ranges(frame[inside], variables))
        combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        return finalize(combined, window)

    def _convert_variable(self, variable: str, values: pd.Series) -> pd.Series:
        column, unit = _WEATHER_COLUMNS[variable]
        numeric = pd.to_numeric(values, errors="coerce").astype(float)
        if variable == "cloud_pct":
            return numeric  # oktas are converted after counting, in _screen_ranges
        if variable == "precip_mm":
            return numeric  # trace marker (-1) is handled in _screen_ranges
        return convert(numeric, unit, WEATHER_CANONICAL_UNITS[variable])

    def _screen_ranges(self, frame: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
        """Apply the declared value rules and count every value they change."""
        out = frame.copy()

        def count(variable: str, rule: str, n: int) -> None:
            if n:
                stats = self.weather_screening.setdefault(variable, {})
                stats[rule] = stats.get(rule, 0) + n

        if "precip_mm" in variables:
            trace = out["precip_mm"] == -1.0
            out.loc[trace, "precip_mm"] = 0.0
            count("precip_mm", "trace_set_to_zero", int(trace.sum()))
        if "cloud_pct" in variables:
            obscured = out["cloud_pct"] == _OBSCURED_SKY_OKTA
            out["cloud_pct"] = out["cloud_pct"] * _OKTA_TO_PCT
            out.loc[obscured, "cloud_pct"] = 100.0
            count("cloud_pct", "obscured_sky_set_to_100", int(obscured.sum()))
        for variable in variables:
            low, high = WEATHER_VARIABLES[variable]
            bad = out[variable].notna() & ((out[variable] < low) | (out[variable] > high))
            out.loc[bad, variable] = np.nan
            count(variable, "out_of_range_set_to_nan", int(bad.sum()))
        return out

    # --- dataset facts -------------------------------------------------------------------
    def _sha256(self, name: str) -> str:
        if name not in self._hashes:
            digest = hashlib.sha256()
            with self._path(name).open("rb") as handle:
                for chunk in iter(lambda: handle.read(1 << 20), b""):
                    digest.update(chunk)
            self._hashes[name] = digest.hexdigest()
        return self._hashes[name]

    def info(self) -> dict[str, Any]:
        names = [self.config.metadata_file, self.config.electricity_file, self.config.weather_file]
        files = [
            {
                "name": n,
                "path": str(self._path(n)),
                "sha256": self._sha256(n),
                "size_bytes": self._path(n).stat().st_size,
            }
            for n in names
        ]
        return {
            "dataset_id": DATASET_ID,
            "name": "Building Data Genome Project 2 (BDG2)",
            "version": "github master, files verified by SHA-256",
            "source_url": SOURCE_URL,
            "licence_id": DATASET_ID,
            "licence": "CC BY-SA 4.0 (Attribution-ShareAlike): credit the authors; adaptations "
            "must be shared under the same licence",
            "citation": CITATION,
            "native_granularity_s": 3600,
            "time_coverage_local": [
                self._read_electricity().index.min().isoformat(),
                self._read_electricity().index.max().isoformat(),
            ],
            "timestamp_clock": self.config.timestamp_clock,
            "supported_tasks": ["forecasting"],
            "meter_types": ["electricity_total"],
            "coordinate_precision": "site level (city); four sites have no coordinates",
            "bbox_by_timezone": self.config.bbox_by_timezone,
            "files": files,
            "notes": [
                "Timestamps are local standard time with DST ignored "
                "(one continuous hourly clock).",
                "Residential-looking categories are mostly student housing, not homes.",
                "All buildings of a site share one coordinate pair and one weather series.",
            ],
        }
