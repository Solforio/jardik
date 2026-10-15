"""The extraction request: every choice that shapes an extraction, validated and hashable.

The same model drives the interactive prompt and a YAML file, so an interactive session can be
saved and replayed exactly.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Literal, Self

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from jardik.core.interfaces import TimeWindow
from jardik.core.schemas import WEATHER_VARIABLES, LocationQuality, UsageCategory
from jardik.dataset_manager.matching.policy import MatchPolicy

DEFAULT_WEATHER_VARIABLES = ["temp_c", "dewpoint_c", "wind_ms", "wind_dir_deg", "pressure_msl_hpa"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AreaFilter(_Strict):
    """Geographic selection. Empty lists mean no restriction."""

    countries: list[str] = Field(default_factory=list, description="ISO alpha-2 codes")
    regions: list[str] = Field(default_factory=list)
    provinces: list[str] = Field(default_factory=list)
    municipalities: list[str] = Field(default_factory=list)
    postal_codes: list[str] = Field(default_factory=list)
    site_ids: list[str] = Field(default_factory=list)
    bbox: tuple[float, float, float, float] | None = Field(
        default=None, description="(min_lat, min_lon, max_lat, max_lon)"
    )

    @field_validator("bbox")
    @classmethod
    def _check_bbox(
        cls, value: tuple[float, float, float, float] | None
    ) -> tuple[float, float, float, float] | None:
        if value is None:
            return value
        min_lat, min_lon, max_lat, max_lon = value
        if not (-90 <= min_lat < max_lat <= 90 and -180 <= min_lon < max_lon <= 180):
            raise ValueError("bbox must be (min_lat, min_lon, max_lat, max_lon) with min < max")
        return value


class QualityRequest(_Strict):
    max_missing_ratio: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Max share of slots still without data after imputation.",
    )
    max_imputed_ratio: float = Field(default=0.05, ge=0, le=1)
    max_near_zero_ratio: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="Max share of readings at or below 1 % of the building's median (dead meter).",
    )
    imputation: Literal["none", "linear_short_gaps"] = "none"
    max_gap_hours: int = Field(default=3, ge=1)


class WeatherRequest(_Strict):
    # "dataset": weather shipped with the dataset. Provider-sourced weather arrives with M2b.
    source: Literal["dataset"] = "dataset"
    variables: list[str] = Field(default_factory=lambda: list(DEFAULT_WEATHER_VARIABLES))
    matching: MatchPolicy = Field(default_factory=MatchPolicy)

    @field_validator("variables")
    @classmethod
    def _check_variables(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("at least one weather variable is required")
        unknown = [v for v in value if v not in WEATHER_VARIABLES]
        if unknown:
            raise ValueError(
                f"unknown weather variable(s) {unknown}; valid: {sorted(WEATHER_VARIABLES)}"
            )
        return list(dict.fromkeys(value))


class ExportRequest(_Strict):
    format: Literal["parquet", "csv"] = "parquet"
    # CSV convention: spanish = ';' separator and decimal comma (Excel in Spanish locale).
    csv_style: Literal["spanish", "plain"] = "spanish"
    output_dir: Path | None = None
    report_pdf: bool = True


class ExtractionRequest(_Strict):
    name: str = Field(pattern=r"^[A-Za-z0-9_\-]+$")
    task: Literal["forecasting", "nilm", "both"] = "forecasting"
    dataset_id: str = "bdg2"

    area: AreaFilter = Field(default_factory=AreaFilter)
    usage_categories: list[UsageCategory] = Field(default_factory=list)
    min_location_quality: LocationQuality | None = None
    contracted_power_kw: tuple[float, float] | None = Field(
        default=None, description="(min, max) in kW; buildings with unknown power are excluded"
    )

    n_buildings: int | None = Field(default=None, gt=0)
    sampling: Literal["random", "stratified_site", "first"] = "random"
    seed: int = 42

    start: date
    end: date
    granularity_s: int = Field(default=3600, gt=0)

    quality: QualityRequest = Field(default_factory=QualityRequest)
    weather: WeatherRequest = Field(default_factory=WeatherRequest)
    export: ExportRequest = Field(default_factory=ExportRequest)

    @field_validator("contracted_power_kw")
    @classmethod
    def _check_power(cls, value: tuple[float, float] | None) -> tuple[float, float] | None:
        if value is not None and not (0 <= value[0] < value[1]):
            raise ValueError("contracted_power_kw must be (min, max) with 0 <= min < max")
        return value

    @model_validator(mode="after")
    def _check_period(self) -> Self:
        if self.end <= self.start:
            raise ValueError("end must be after start")
        return self

    @property
    def window(self) -> TimeWindow:
        """Period as a UTC half-open window [start 00:00, end 00:00)."""
        return TimeWindow(pd.Timestamp(self.start, tz="UTC"), pd.Timestamp(self.end, tz="UTC"))

    def request_hash(self) -> str:
        """Short hash of everything that influences the result (not where it is written)."""
        payload = self.model_dump(mode="json", exclude={"export": {"output_dir"}})
        text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]

    @property
    def extraction_id(self) -> str:
        return f"{self.name}-{self.request_hash()}"

    @classmethod
    def from_yaml(cls, path: Path) -> ExtractionRequest:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.model_validate(raw)

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
