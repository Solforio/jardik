"""Frame contracts for the canonical tables: load, weather and weather match."""

from __future__ import annotations

import pandas as pd

from jardik.core.schemas.enums import (
    LocationQuality,
    MatchingMethod,
    MeterType,
    ProductType,
    QualityFlag,
    SpatialLink,
)
from jardik.core.schemas.frames import (
    Column,
    FrameContract,
    collect_violations,
    validate_frame,
)


def _values(enum_cls: type) -> frozenset[str]:
    return frozenset(member.value for member in enum_cls)  # type: ignore[attr-defined]


# Canonical weather variables and their physical plausibility range (min, max).
WEATHER_VARIABLES: dict[str, tuple[float, float]] = {
    "temp_c": (-90.0, 60.0),
    "dewpoint_c": (-90.0, 40.0),
    "rh_pct": (0.0, 100.0),
    "wind_ms": (0.0, 120.0),
    "wind_dir_deg": (0.0, 360.0),
    "pressure_hpa": (300.0, 1100.0),
    "pressure_msl_hpa": (850.0, 1100.0),
    "precip_mm": (0.0, 500.0),
    "cloud_pct": (0.0, 100.0),
    "ghi_wm2": (0.0, 1500.0),
    "dni_wm2": (0.0, 1500.0),
    "dhi_wm2": (0.0, 1200.0),
}

# Canonical unit of each weather variable (names follow the column suffix convention).
WEATHER_CANONICAL_UNITS: dict[str, str] = {
    "temp_c": "degC",
    "dewpoint_c": "degC",
    "rh_pct": "%",
    "wind_ms": "m/s",
    "wind_dir_deg": "deg",
    "pressure_hpa": "hPa",
    "pressure_msl_hpa": "hPa",
    "precip_mm": "mm",
    "cloud_pct": "%",
    "ghi_wm2": "W/m2",
    "dni_wm2": "W/m2",
    "dhi_wm2": "W/m2",
}

# Unit that `value` must carry for each meter type.
LOAD_UNIT_BY_METER: dict[str, str] = {
    MeterType.ELECTRICITY_TOTAL.value: "kWh",
    MeterType.ELECTRICITY_APPLIANCE.value: "W",
    MeterType.ELECTRICITY_PV_GENERATION.value: "kWh",
}


def _load_semantics(df: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    if (df["granularity_s"] <= 0).any():
        problems.append("granularity_s must be positive")

    missing_with_value = df["is_missing"] & df["value"].notna()
    if missing_with_value.any():
        problems.append(
            f"{int(missing_with_value.sum())} row(s) flagged is_missing but have a value"
        )
    value_null_unflagged = df["value"].isna() & ~df["is_missing"]
    if value_null_unflagged.any():
        problems.append(f"{int(value_null_unflagged.sum())} null value(s) not flagged is_missing")
    both = df["is_missing"] & df["is_imputed"]
    if both.any():
        problems.append(f"{int(both.sum())} row(s) are both is_missing and is_imputed")

    for meter, expected in LOAD_UNIT_BY_METER.items():
        subset = df[df["meter_type"] == meter]
        wrong = subset[subset["unit"] != expected]
        if len(wrong):
            problems.append(
                f"meter_type '{meter}' must use unit '{expected}' ({len(wrong)} row(s) differ)"
            )

    wall_minus_utc = (df["ts_local"] - df["ts_utc"].dt.tz_localize(None)).dt.total_seconds() / 60
    if not (wall_minus_utc == df["utc_offset_min"]).all():
        problems.append("ts_local does not describe the same instant as ts_utc (offset mismatch)")
    return problems


def _weather_semantics(df: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    forecast = df["product_type"] == ProductType.FORECAST.value
    if "issued_at_utc" not in df.columns:
        if forecast.any():
            problems.append("forecast rows require the issued_at_utc column")
    elif (forecast & df["issued_at_utc"].isna()).any():
        problems.append("forecast rows must carry issued_at_utc")

    for name, (low, high) in WEATHER_VARIABLES.items():
        if name in df.columns:
            out = df[name].dropna()
            bad = int(((out < low) | (out > high)).sum())
            if bad:
                problems.append(
                    f"{bad} value(s) of '{name}' outside plausible range [{low}, {high}]"
                )
    return problems


def _match_semantics(df: pd.DataFrame) -> list[str]:
    problems: list[str] = []
    if (df["distance_km"].dropna() < 0).any():
        problems.append("distance_km must be >= 0")
    unknown_geometry = df[["matched_lat", "matched_lon", "distance_km"]].isna().any(axis=1)
    bad = unknown_geometry & (df["spatial_link"] != SpatialLink.DATASET_LINK.value)
    if bad.any():
        problems.append(
            f"{int(bad.sum())} row(s) lack matched coordinates or distance_km but spatial_link "
            "is not 'dataset_link'; a nearest_point match must know its geometry"
        )
    return problems


LOAD_CONTRACT = FrameContract(
    name="load_timeseries",
    columns=(
        Column("building_id", "str"),
        Column("meter_type", "str", allowed=_values(MeterType)),
        Column("appliance", "str", nullable=True),
        Column("ts_utc", "datetime_utc"),
        Column("ts_local", "datetime_naive"),
        Column("utc_offset_min", "int"),
        Column("value", "float", nullable=True),
        Column("unit", "str"),
        Column("granularity_s", "int"),
        Column("is_missing", "bool"),
        Column("is_imputed", "bool"),
        Column("quality_flag", "str", allowed=_values(QualityFlag)),
    ),
    unique_key=("building_id", "meter_type", "appliance", "ts_utc"),
    extra_checks=(_load_semantics,),
)

WEATHER_CONTRACT = FrameContract(
    name="weather_timeseries",
    columns=(
        Column("weather_point_id", "str"),
        Column("ts_utc", "datetime_utc"),
        Column("source", "str"),
        Column("product_type", "str", allowed=_values(ProductType)),
        Column("issued_at_utc", "datetime_utc", nullable=True, required=False),
        *(Column(name, "float", nullable=True, required=False) for name in WEATHER_VARIABLES),
    ),
    unique_key=("weather_point_id", "ts_utc", "product_type", "issued_at_utc"),
    extra_checks=(_weather_semantics,),
)

WEATHER_MATCH_CONTRACT = FrameContract(
    name="weather_match",
    columns=(
        Column("building_id", "str"),
        Column("ts_utc", "datetime_utc"),
        Column("weather_source", "str"),
        Column("matched_lat", "float", nullable=True),
        Column("matched_lon", "float", nullable=True),
        Column("distance_km", "float", nullable=True),
        Column("weather_time_offset_min", "int"),
        Column("timezone_used", "str"),
        Column("matching_method", "str", allowed=_values(MatchingMethod)),
        Column("spatial_link", "str", allowed=_values(SpatialLink)),
        Column("location_quality", "str", allowed=_values(LocationQuality)),
    ),
    unique_key=("building_id", "ts_utc"),
    extra_checks=(_match_semantics,),
)


def validate_load(df: pd.DataFrame) -> None:
    validate_frame(df, LOAD_CONTRACT)


def validate_weather(df: pd.DataFrame) -> None:
    validate_frame(df, WEATHER_CONTRACT)


def validate_weather_match(df: pd.DataFrame) -> None:
    validate_frame(df, WEATHER_MATCH_CONTRACT)


__all__ = [
    "LOAD_CONTRACT",
    "WEATHER_CANONICAL_UNITS",
    "WEATHER_CONTRACT",
    "WEATHER_MATCH_CONTRACT",
    "WEATHER_VARIABLES",
    "collect_violations",
    "validate_load",
    "validate_weather",
    "validate_weather_match",
]
