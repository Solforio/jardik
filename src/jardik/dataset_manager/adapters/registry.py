"""Chooses the dataset adapter by id; per-dataset settings live in configs/datasets/<id>.yaml."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from jardik.core.config import AppConfig
from jardik.core.interfaces import DatasetAdapter
from jardik.core.paths import JardikPaths
from jardik.dataset_manager.adapters.base import DatasetError
from jardik.dataset_manager.adapters.bdg2 import Bdg2Adapter, Bdg2Config


def dataset_config_path(paths: JardikPaths, dataset_id: str) -> Path:
    return paths.configs_dir / "datasets" / f"{dataset_id}.yaml"


def raw_data_dir(paths: JardikPaths, dataset_id: str) -> Path:
    return paths.raw_dir / dataset_id


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _build_bdg2(cfg: AppConfig, paths: JardikPaths) -> DatasetAdapter:
    settings = _load_yaml(dataset_config_path(paths, "bdg2"))
    settings.setdefault("data_dir", str(raw_data_dir(paths, "bdg2")))
    settings.setdefault("time_policy", cfg.time_policy.model_dump())
    return Bdg2Adapter(Bdg2Config.model_validate(settings))


_ADAPTERS: dict[str, Callable[[AppConfig, JardikPaths], DatasetAdapter]] = {"bdg2": _build_bdg2}


def available_datasets() -> list[str]:
    return sorted(_ADAPTERS)


def build_adapter(dataset_id: str, cfg: AppConfig, paths: JardikPaths) -> DatasetAdapter:
    factory = _ADAPTERS.get(dataset_id)
    if factory is None:
        raise DatasetError(
            f"Unknown dataset {dataset_id!r}. Available: {', '.join(available_datasets())}"
        )
    return factory(cfg, paths)
