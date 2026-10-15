"""Controlled vocabularies shared by all tables."""

from __future__ import annotations

from enum import StrEnum


class UsageCategory(StrEnum):
    HOME = "home"
    CONDOMINIUM = "condominium"
    OFFICE = "office"
    SHOP = "shop"
    OTHER = "other"


class LocationQuality(StrEnum):
    """Best available location, from most to least precise."""

    COORDINATES = "coordinates"
    POSTAL_CENTROID = "postal_centroid"
    MUNICIPALITY = "municipality"
    PROVINCE = "province"
    REGION = "region"
    UNKNOWN = "unknown"


# Order used when comparing precision (lower index is better).
LOCATION_QUALITY_ORDER: tuple[LocationQuality, ...] = tuple(LocationQuality)


class MeterType(StrEnum):
    ELECTRICITY_TOTAL = "electricity_total"
    ELECTRICITY_APPLIANCE = "electricity_appliance"
    ELECTRICITY_PV_GENERATION = "electricity_pv_generation"
    OTHER = "other"


class ProductType(StrEnum):
    OBSERVATION = "observation"
    REANALYSIS = "reanalysis"
    FORECAST = "forecast"


class QualityFlag(StrEnum):
    OK = "ok"
    SUSPECT = "suspect"
    IMPUTED = "imputed"
    MISSING = "missing"


class MatchingMethod(StrEnum):
    EXACT = "exact"
    NEAREST_PREVIOUS = "nearest_previous"
    NEAREST = "nearest"
    INTERPOLATE = "interpolate"


class SpatialLink(StrEnum):
    """How a building was tied to a weather point."""

    NEAREST_POINT = "nearest_point"  # nearest by distance from known coordinates
    DATASET_LINK = "dataset_link"  # the dataset itself says which weather series belongs to it
