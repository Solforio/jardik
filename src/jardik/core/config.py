"""Configuration models (Pydantic) loaded from YAML, plus secret access via environment."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field

from jardik.core.paths import JardikPaths, find_project_root
from jardik.core.timeutils import AmbiguousPolicy, NonexistentPolicy


class ConfigError(ValueError):
    """The configuration file is missing or invalid."""


class MissingSecretError(RuntimeError):
    """A required secret is not present in the environment."""


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PathsConfig(_Strict):
    data_dir: Path = Path("data")
    outputs_dir: Path = Path("outputs")


class LoggingConfig(_Strict):
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_to_file: bool = False


class TimePolicyConfig(_Strict):
    ambiguous: AmbiguousPolicy = "raise"
    nonexistent: NonexistentPolicy = "raise"


class FileWeatherConfig(_Strict):
    """Weather read from a local CSV. Every mapped column needs an explicit unit."""

    path: Path
    source: str = "file"
    timestamp_col: str = "timestamp"
    timezone: str | None = Field(
        default=None, description="IANA zone of naive timestamps. Not needed if they carry offsets."
    )
    delimiter: str = ","
    decimal: str = "."
    columns: dict[str, str] = Field(description="canonical variable -> column name in the file")
    units: dict[str, str] = Field(description="canonical variable -> unit used in the file")
    product_type: Literal["observation", "reanalysis", "forecast"] = "observation"
    issued_at_col: str | None = None
    # One fixed point ...
    point_id: str = "file_point"
    lat: float | None = Field(default=None, ge=-90, le=90)
    lon: float | None = Field(default=None, ge=-180, le=180)
    # ... or one point per row, identified by columns.
    point_id_col: str | None = None
    lat_col: str | None = None
    lon_col: str | None = None


class OpenMeteoConfig(_Strict):
    """Open-Meteo: free for NON-COMMERCIAL use only (see LICENSES.md)."""

    archive_url: str = "https://archive-api.open-meteo.com/v1/archive"
    forecast_url: str = "https://api.open-meteo.com/v1/forecast"
    forecast_ttl_hours: float = Field(default=3.0, gt=0)
    timeout_s: float = Field(default=30.0, gt=0)


class ProvidersConfig(_Strict):
    weather: str = "file"
    solar: str | None = None
    market: str | None = None
    cache_enabled: bool = True
    file_weather: FileWeatherConfig | None = None
    open_meteo: OpenMeteoConfig = OpenMeteoConfig()


class ForecastConfig(_Strict):
    horizon_h: int = Field(default=24, gt=0)
    granularity_s: int = Field(default=3600, gt=0)
    target_unit: Literal["kWh"] = "kWh"


class AppConfig(_Strict):
    paths: PathsConfig = PathsConfig()
    logging: LoggingConfig = LoggingConfig()
    time_policy: TimePolicyConfig = TimePolicyConfig()
    providers: ProvidersConfig = ProvidersConfig()
    forecast: ForecastConfig = ForecastConfig()

    def build_paths(self, root: Path | None = None) -> JardikPaths:
        return JardikPaths.from_root(root, self.paths.data_dir, self.paths.outputs_dir)


def load_config(path: Path | None = None) -> AppConfig:
    """Load and validate a YAML config. With no path, use configs/default.yaml if it exists."""
    if path is None:
        default = JardikPaths.from_root().default_config
        if not default.is_file():
            return AppConfig()
        path = default
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"Config root must be a mapping: {path}")
    return AppConfig.model_validate(raw)


def config_hash(config: BaseModel) -> str:
    """Stable short hash of a config, recorded in every run and extraction manifest."""
    payload = json.dumps(config.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_env(root: Path | None = None) -> None:
    """Load the project's .env file, if any. Existing environment variables win."""
    env_file = (root or find_project_root()) / ".env"
    if env_file.is_file():
        load_dotenv(env_file, override=False)


def get_secret(name: str) -> str:
    """Return a secret from the environment. Never read from config files or code."""
    value = os.environ.get(name, "").strip()
    if not value:
        raise MissingSecretError(
            f"Environment variable {name} is not set. See .env.example for the expected names."
        )
    return value
