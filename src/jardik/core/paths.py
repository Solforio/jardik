"""Single place that defines every directory and file path used by Jardik.

Other modules must ask `JardikPaths` for a location instead of building paths themselves.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def find_project_root(start: Path | None = None) -> Path:
    """Return the project root.

    Order: the JARDIK_HOME environment variable, then the closest parent of `start`
    (default: current directory) that contains a pyproject.toml, then `start` itself.
    """
    env_home = os.environ.get("JARDIK_HOME", "").strip()
    if env_home:
        return Path(env_home).expanduser().resolve()
    here = (start or Path.cwd()).resolve()
    for candidate in (here, *here.parents):
        if (candidate / "pyproject.toml").is_file():
            return candidate
    return here


@dataclass(frozen=True)
class JardikPaths:
    """All filesystem locations, derived from a root plus two configurable folders."""

    root: Path
    data_dir: Path
    outputs_dir: Path

    @classmethod
    def from_root(
        cls,
        root: Path | None = None,
        data_dir: Path | str = "data",
        outputs_dir: Path | str = "outputs",
    ) -> JardikPaths:
        base = (root or find_project_root()).resolve()
        return cls(
            root=base,
            data_dir=_resolve(base, data_dir),
            outputs_dir=_resolve(base, outputs_dir),
        )

    # --- configuration -------------------------------------------------
    @property
    def configs_dir(self) -> Path:
        return self.root / "configs"

    @property
    def default_config(self) -> Path:
        return self.configs_dir / "default.yaml"

    @property
    def env_file(self) -> Path:
        return self.root / ".env"

    # --- data layers ---------------------------------------------------
    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def provider_cache_dir(self) -> Path:
        """API response cache shared by all providers (one sub-folder per provider)."""
        return self.cache_dir / "providers"

    @property
    def catalogue_path(self) -> Path:
        return self.data_dir / "catalogue.duckdb"

    # --- outputs -------------------------------------------------------
    @property
    def extractions_dir(self) -> Path:
        return self.outputs_dir / "extractions"

    @property
    def runs_dir(self) -> Path:
        return self.outputs_dir / "runs"

    @property
    def reports_dir(self) -> Path:
        return self.outputs_dir / "reports"

    @property
    def requests_dir(self) -> Path:
        """Extraction requests saved from interactive sessions, so they can be replayed."""
        return self.outputs_dir / "requests"

    @property
    def review_dir(self) -> Path:
        """Human-readable copies of extractions (Excel workbook, PDF, sample CSV)."""
        return self.root / "review"

    def extraction_dir(self, extraction_id: str) -> Path:
        return self.extractions_dir / _safe_name(extraction_id)

    def run_dir(self, run_id: str) -> Path:
        return self.runs_dir / _safe_name(run_id)

    def log_file(self) -> Path:
        return self.outputs_dir / "logs" / "jardik.log"

    def ensure(self) -> None:
        """Create the writable directory tree (idempotent)."""
        for directory in (
            self.raw_dir,
            self.interim_dir,
            self.processed_dir,
            self.cache_dir,
            self.extractions_dir,
            self.runs_dir,
            self.reports_dir,
            self.requests_dir,
            self.log_file().parent,
        ):
            directory.mkdir(parents=True, exist_ok=True)


def _resolve(base: Path, value: Path | str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (base / path).resolve()


def _safe_name(name: str) -> str:
    """Reject identifiers that could escape the output folders."""
    if not name or any(sep in name for sep in ("/", "\\", "..")) or ":" in name:
        raise ValueError(f"Unsafe identifier for a path component: {name!r}")
    return name
