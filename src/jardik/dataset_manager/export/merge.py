"""The ML-ready merged training dataset: load + matched weather + static + calendar columns."""

from __future__ import annotations

import pandas as pd

# column -> (unit, description). Single source for the manifest and the PDF column schema.
COLUMN_DOCS: dict[str, tuple[str, str]] = {
    "building_id": ("", "Stable building key '<dataset>:<source id>'"),
    "ts_utc": ("UTC", "Start of the hourly slot, timezone-aware UTC (the key)"),
    "ts_local": (
        "local",
        "Local wall-clock time of the building, without timezone (see utc_offset_min)",
    ),
    "utc_offset_min": ("min", "Civil UTC offset of ts_local, DST included"),
    "target_kwh": (
        "kWh per hour",
        "Electrical energy in the slot (forecast target); empty if missing",
    ),
    "is_missing": ("bool", "No usable measurement in this slot"),
    "is_imputed": ("bool", "Value filled in by the declared imputation rule"),
    "quality_flag": ("", "ok, imputed or missing"),
    "temp_c": ("degC", "Air temperature"),
    "dewpoint_c": ("degC", "Dew point temperature"),
    "rh_pct": ("%", "Relative humidity"),
    "wind_ms": ("m/s", "Wind speed"),
    "wind_dir_deg": ("deg", "Wind direction (from, clockwise from north)"),
    "pressure_hpa": ("hPa", "Station pressure"),
    "pressure_msl_hpa": ("hPa", "Pressure reduced to mean sea level"),
    "precip_mm": ("mm", "Precipitation in the hour"),
    "cloud_pct": ("%", "Cloud cover"),
    "ghi_wm2": ("W/m2", "Global horizontal irradiance"),
    "dni_wm2": ("W/m2", "Direct normal irradiance"),
    "dhi_wm2": ("W/m2", "Diffuse horizontal irradiance"),
    "weather_time_offset_min": ("min", "Weather record time minus slot time (0 = same instant)"),
    "distance_km": ("km", "Distance between building and weather point; empty if unknown"),
    "spatial_link": ("", "nearest_point or dataset_link"),
    "usage_category": ("", "home, condominium, office, shop or other"),
    "site_id": ("", "Site identifier in the source dataset"),
    "floor_area_m2": ("m2", "Floor area"),
    "year_built": ("year", "Year of construction; empty if unknown"),
    "timezone": ("IANA", "Timezone of the building"),
    "lat": ("deg", "Latitude (site level for BDG2)"),
    "lon": ("deg", "Longitude (site level for BDG2)"),
    "location_quality": ("", "Best location level available for the building"),
    "hour_local": ("h", "Local hour of day 0-23 (civil time, DST aware)"),
    "dayofweek_local": ("", "Local day of week, 0 = Monday"),
    "month_local": ("", "Local month 1-12"),
    "is_weekend_local": ("bool", "Saturday or Sunday in local time"),
}

_STATIC = [
    "usage_category",
    "site_id",
    "floor_area_m2",
    "year_built",
    "timezone",
    "lat",
    "lon",
    "location_quality",
]


def build_merged_dataset(
    load: pd.DataFrame,
    values: pd.DataFrame,
    match: pd.DataFrame,
    metadata: pd.DataFrame,
    weather_variables: list[str],
) -> pd.DataFrame:
    """One row per load slot. Weather columns are empty where no weather was matched."""
    keys = ["building_id", "ts_utc"]
    base = load[
        [*keys, "ts_local", "utc_offset_min", "value", "is_missing", "is_imputed", "quality_flag"]
    ]
    base = base.rename(columns={"value": "target_kwh"})
    merged = base.merge(values[[*keys, *weather_variables]], on=keys, how="left")
    merged = merged.merge(
        match[[*keys, "weather_time_offset_min", "distance_km", "spatial_link"]],
        on=keys,
        how="left",
    )
    merged["weather_time_offset_min"] = merged["weather_time_offset_min"].astype("Int64")
    merged = merged.merge(metadata[["building_id", *_STATIC]], on="building_id", how="left")

    local = merged["ts_local"]
    merged["hour_local"] = local.dt.hour.astype("int64")
    merged["dayofweek_local"] = local.dt.dayofweek.astype("int64")
    merged["month_local"] = local.dt.month.astype("int64")
    merged["is_weekend_local"] = merged["dayofweek_local"] >= 5

    order = [c for c in COLUMN_DOCS if c in merged.columns]
    return merged.sort_values(keys, kind="stable").reset_index(drop=True)[order]
