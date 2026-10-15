"""Canonical data schemas: enums, building metadata model and DataFrame contracts."""

from jardik.core.schemas.building import BuildingMetadata, derive_location_quality
from jardik.core.schemas.contracts import (
    LOAD_CONTRACT,
    WEATHER_CANONICAL_UNITS,
    WEATHER_CONTRACT,
    WEATHER_MATCH_CONTRACT,
    WEATHER_VARIABLES,
    validate_load,
    validate_weather,
    validate_weather_match,
)
from jardik.core.schemas.enums import (
    LocationQuality,
    MatchingMethod,
    MeterType,
    ProductType,
    QualityFlag,
    SpatialLink,
    UsageCategory,
)
from jardik.core.schemas.frames import (
    Column,
    ContractError,
    FrameContract,
    collect_violations,
    validate_frame,
)

__all__ = [
    "LOAD_CONTRACT",
    "WEATHER_CANONICAL_UNITS",
    "WEATHER_CONTRACT",
    "WEATHER_MATCH_CONTRACT",
    "WEATHER_VARIABLES",
    "BuildingMetadata",
    "Column",
    "ContractError",
    "FrameContract",
    "LocationQuality",
    "MatchingMethod",
    "MeterType",
    "ProductType",
    "QualityFlag",
    "SpatialLink",
    "UsageCategory",
    "collect_violations",
    "derive_location_quality",
    "validate_frame",
    "validate_load",
    "validate_weather",
    "validate_weather_match",
]
