"""Building metadata contract (one row per building)."""

from __future__ import annotations

import re

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)

from jardik.core.schemas.enums import LocationQuality, UsageCategory
from jardik.core.timeutils import TimeError, validate_timezone

_BUILDING_ID = re.compile(r"^[a-z0-9_\-]+:[A-Za-z0-9_\-\.]+$")


def derive_location_quality(
    lat: float | None,
    lon: float | None,
    postal_code: str | None,
    municipality: str | None,
    province: str | None,
    region: str | None,
) -> LocationQuality:
    """Best location level actually available. Derived from data, never typed by the user."""
    if lat is not None and lon is not None:
        return LocationQuality.COORDINATES
    if postal_code:
        return LocationQuality.POSTAL_CENTROID
    if municipality:
        return LocationQuality.MUNICIPALITY
    if province:
        return LocationQuality.PROVINCE
    if region:
        return LocationQuality.REGION
    return LocationQuality.UNKNOWN


class BuildingMetadata(BaseModel):
    """Static description of a building. Unknown values are None, never guessed."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    building_id: str = Field(description="'<dataset_id>:<source_id>'")
    dataset_id: str
    site_id: str | None = None

    usage_category: UsageCategory
    usage_raw: str | None = None

    country: str | None = Field(default=None, description="ISO 3166-1 alpha-2, upper case")
    region: str | None = None
    province: str | None = None
    municipality: str | None = None
    postal_code: str | None = None
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    timezone: str = Field(description="IANA timezone name")

    floor_area_m2: float | None = Field(default=None, gt=0)
    contracted_power_kw: float | None = Field(default=None, gt=0)
    year_built: int | None = Field(default=None, ge=1800, le=2100)

    has_pv: bool | None = None
    has_battery: bool | None = None
    has_ac: bool | None = None
    has_heating: bool | None = None
    heating_fuel: str | None = None

    licence_id: str | None = Field(default=None, description="Entry key in LICENSES.md")

    @field_validator("building_id")
    @classmethod
    def _check_building_id(cls, value: str) -> str:
        if not _BUILDING_ID.match(value):
            raise ValueError("building_id must look like '<dataset_id>:<source_id>'")
        return value

    @field_validator("country")
    @classmethod
    def _check_country(cls, value: str | None) -> str | None:
        if value is not None and not re.fullmatch(r"[A-Z]{2}", value):
            raise ValueError("country must be an upper-case ISO alpha-2 code")
        return value

    @field_validator("timezone")
    @classmethod
    def _check_timezone(cls, value: str) -> str:
        try:
            validate_timezone(value)
        except TimeError as exc:
            raise ValueError(str(exc)) from exc
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> BuildingMetadata:
        if (self.lat is None) != (self.lon is None):
            raise ValueError("lat and lon must be given together")
        if self.building_id.split(":", 1)[0] != self.dataset_id:
            raise ValueError("building_id prefix must equal dataset_id")
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def location_quality(self) -> LocationQuality:
        return derive_location_quality(
            self.lat, self.lon, self.postal_code, self.municipality, self.province, self.region
        )
