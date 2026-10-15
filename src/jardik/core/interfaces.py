"""Public interfaces between modules (typing.Protocol).

Modules depend on these contracts, never on each other's internals. Adapters (datasets,
weather, solar, market) and models (baselines, gradient boosting, deep learning, NILM) all
implement one of these protocols, so they can be swapped by configuration and compared by the
same evaluation code.

Frames exchanged through these interfaces must satisfy the contracts in `jardik.core.schemas`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import pandas as pd

from jardik.core.schemas.enums import LocationQuality


@dataclass(frozen=True)
class Location:
    """A point on Earth plus how well it is known."""

    lat: float
    lon: float
    quality: LocationQuality = LocationQuality.COORDINATES


@dataclass(frozen=True)
class TimeWindow:
    """Half-open interval [start, end). Both bounds must be timezone-aware."""

    start: pd.Timestamp
    end: pd.Timestamp

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.end.tzinfo is None:
            raise ValueError("TimeWindow bounds must be timezone-aware")
        if self.end <= self.start:
            raise ValueError("TimeWindow end must be after start")


# --- external data providers -------------------------------------------------------------


@runtime_checkable
class WeatherProvider(Protocol):
    """Weather source. Returns frames that satisfy WEATHER_CONTRACT."""

    name: str

    def get_history(
        self, location: Location, window: TimeWindow, variables: list[str]
    ) -> pd.DataFrame: ...

    def get_forecast(
        self, location: Location, window: TimeWindow, variables: list[str]
    ) -> pd.DataFrame: ...


@runtime_checkable
class SolarProvider(Protocol):
    """Solar radiation / PV production source."""

    name: str

    def get_irradiance(self, location: Location, window: TimeWindow) -> pd.DataFrame: ...


@runtime_checkable
class MarketProvider(Protocol):
    """Electricity price and market data source."""

    name: str

    def get_prices(self, zone: str, window: TimeWindow) -> pd.DataFrame: ...


# --- datasets ----------------------------------------------------------------------------


@runtime_checkable
class DatasetAdapter(Protocol):
    """One adapter per dataset family. Output frames follow the canonical contracts."""

    dataset_id: str
    # Declared corrections applied to the source data (always reported, never silent) and the
    # counts of weather values changed by the adapter's value rules during the last load.
    corrections: list[dict[str, Any]]
    weather_screening: dict[str, dict[str, int]]

    def info(self) -> dict[str, Any]:
        """Dataset-level facts: licence, citation, native granularity, clock convention,
        supported tasks, source files with checksums, known limitations."""
        ...

    def discover(self) -> pd.DataFrame:
        """One row per building available for extraction (BuildingMetadata fields plus
        usage_sub_raw, weather_link and location_quality)."""
        ...

    def load_metadata(self, building_ids: list[str] | None = None) -> pd.DataFrame:
        """Same shape as `discover`, optionally restricted to some buildings."""
        ...

    def load_wide(self, building_ids: list[str], window: TimeWindow) -> pd.DataFrame:
        """Hourly energy on a common UTC grid: index = every slot of the window, one column per
        building, NaN where there is no usable measurement. Cheap enough for quality screening."""
        ...

    def load_timeseries(self, building_ids: list[str], window: TimeWindow) -> pd.DataFrame:
        """The same data as `load_wide` in long form. Rows that satisfy LOAD_CONTRACT."""
        ...

    def weather_points(self) -> pd.DataFrame:
        """Weather points shipped with the dataset: weather_point_id, lat, lon."""
        ...

    def load_weather(
        self, point_ids: list[str], window: TimeWindow, variables: list[str]
    ) -> pd.DataFrame:
        """Rows that satisfy WEATHER_CONTRACT for the given points."""
        ...


# --- models ------------------------------------------------------------------------------


@runtime_checkable
class Forecaster(Protocol):
    """Every rung of the model ladder implements this, so backtests are model-agnostic."""

    name: str

    def fit(self, train: pd.DataFrame) -> None: ...

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Point forecast in kWh per hour, indexed like `features`."""
        ...

    def predict_quantiles(self, features: pd.DataFrame, quantiles: list[float]) -> pd.DataFrame:
        """One column per quantile. Models without probabilistic output may raise."""
        ...


@runtime_checkable
class Disaggregator(Protocol):
    """NILM model: split an aggregate signal into appliance-level signals."""

    name: str

    def fit(self, aggregate: pd.DataFrame, appliances: pd.DataFrame) -> None: ...

    def disaggregate(self, aggregate: pd.DataFrame) -> pd.DataFrame:
        """One column per appliance, in watts."""
        ...
