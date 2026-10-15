from pathlib import Path

import pytest
from pydantic import ValidationError

from jardik.core.config import (
    AppConfig,
    ConfigError,
    MissingSecretError,
    config_hash,
    get_secret,
    load_config,
)
from jardik.core.formatting import format_number_es, format_percent_es
from jardik.core.paths import JardikPaths, find_project_root


def test_defaults_match_the_plan():
    cfg = AppConfig()
    assert cfg.forecast.horizon_h == 24
    assert cfg.forecast.target_unit == "kWh"
    assert cfg.time_policy.ambiguous == "raise"
    assert cfg.providers.weather == "file"


def test_load_yaml_and_hash_is_stable(tmp_path: Path):
    file = tmp_path / "cfg.yaml"
    file.write_text("forecast:\n  horizon_h: 48\n", encoding="utf-8")
    cfg = load_config(file)
    assert cfg.forecast.horizon_h == 48
    assert config_hash(cfg) == config_hash(load_config(file))
    assert config_hash(cfg) != config_hash(AppConfig())


def test_empty_yaml_gives_defaults(tmp_path: Path):
    file = tmp_path / "empty.yaml"
    file.write_text("", encoding="utf-8")
    assert load_config(file) == AppConfig()


def test_unknown_key_is_rejected(tmp_path: Path):
    file = tmp_path / "bad.yaml"
    file.write_text("forecast:\n  horizon: 24\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(file)


def test_invalid_values_rejected(tmp_path: Path):
    file = tmp_path / "bad.yaml"
    file.write_text("forecast:\n  horizon_h: 0\n", encoding="utf-8")
    with pytest.raises(ValidationError):
        load_config(file)


def test_missing_file_and_bad_root(tmp_path: Path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "nope.yaml")
    file = tmp_path / "list.yaml"
    file.write_text("- a\n- b\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(file)


def test_repository_default_config_is_valid():
    root = find_project_root(Path(__file__))
    assert load_config(root / "configs" / "default.yaml") == AppConfig()


def test_get_secret(monkeypatch):
    monkeypatch.setenv("JARDIK_TEST_TOKEN", " abc ")
    assert get_secret("JARDIK_TEST_TOKEN") == "abc"
    monkeypatch.delenv("JARDIK_TEST_TOKEN")
    with pytest.raises(MissingSecretError):
        get_secret("JARDIK_TEST_TOKEN")


def test_paths_are_centralised(tmp_path: Path):
    paths = JardikPaths.from_root(tmp_path)
    assert paths.data_dir == (tmp_path / "data").resolve()
    assert paths.catalogue_path.parent == paths.data_dir
    assert paths.extraction_dir("ext_001").parent == paths.extractions_dir
    paths.ensure()
    assert paths.cache_dir.is_dir() and paths.reports_dir.is_dir()


def test_paths_reject_unsafe_ids(tmp_path: Path):
    paths = JardikPaths.from_root(tmp_path)
    for bad in ("..", "a/b", "a\\b", "C:evil", ""):
        with pytest.raises(ValueError):
            paths.run_dir(bad)


def test_jardik_home_overrides_root(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("JARDIK_HOME", str(tmp_path))
    assert find_project_root() == tmp_path.resolve()


def test_spanish_number_format():
    assert format_number_es(1234567.891) == "1.234.567,89"
    assert format_number_es(0.5, 1) == "0,5"
    assert format_number_es(-1234.56, 1) == "-1.234,6"
    assert format_number_es(None) == "n/a"
    assert format_number_es(float("nan")) == "n/a"
    assert format_percent_es(0.153) == "15,3 %"
