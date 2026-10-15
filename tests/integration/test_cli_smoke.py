from pathlib import Path

from typer.testing import CliRunner

from jardik import __version__
from jardik.cli import app

runner = CliRunner()


def test_version():
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout


def test_info_shows_paths_and_provider():
    result = runner.invoke(app, ["info"])
    assert result.exit_code == 0
    assert "config hash" in result.stdout
    assert "weather provider: file" in result.stdout


def test_check_config_ok_and_invalid(tmp_path: Path):
    good = tmp_path / "good.yaml"
    good.write_text("forecast:\n  horizon_h: 48\n", encoding="utf-8")
    assert runner.invoke(app, ["check-config", str(good)]).exit_code == 0

    bad = tmp_path / "bad.yaml"
    bad.write_text("forecast:\n  horizon_h: -1\n", encoding="utf-8")
    result = runner.invoke(app, ["check-config", str(bad)])
    assert result.exit_code == 1


def _weather_config(tmp_path: Path, csv_name: str = "w.csv") -> Path:
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "providers:\n"
        "  weather: file\n"
        "  file_weather:\n"
        f"    path: '{(tmp_path / csv_name).as_posix()}'\n"
        "    columns: {temp_c: t}\n"
        "    units: {temp_c: degC}\n"
        "    lat: 40.4\n"
        "    lon: -3.7\n",
        encoding="utf-8",
    )
    return cfg


def test_weather_fetch_from_file_provider(tmp_path: Path):
    (tmp_path / "w.csv").write_text(
        "timestamp,t\n2024-01-15T00:00:00Z,5.0\n2024-01-15T01:00:00Z,4.0\n", encoding="utf-8"
    )
    out = tmp_path / "out" / "result.csv"
    result = runner.invoke(
        app,
        [
            "--config",
            str(_weather_config(tmp_path)),
            "weather",
            "fetch",
            "--lat",
            "40.4",
            "--lon",
            "-3.7",
            "--start",
            "2024-01-15",
            "--end",
            "2024-01-16",
            "--variables",
            "temp_c",
            "--out",
            str(out),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "provider file: 2 rows" in result.stdout
    assert out.is_file()


def test_weather_fetch_rejects_unknown_variable_and_reports_provider_errors(tmp_path: Path):
    base = [
        "--config",
        str(_weather_config(tmp_path)),
        "weather",
        "fetch",
        "--lat",
        "40.4",
        "--lon",
        "-3.7",
        "--start",
        "2024-01-15",
        "--end",
        "2024-01-16",
    ]
    assert runner.invoke(app, [*base, "--variables", "nope"]).exit_code == 2
    # w.csv does not exist: provider error, clean exit code 1 (no traceback)
    assert runner.invoke(app, [*base, "--variables", "temp_c"]).exit_code == 1


def test_bad_config_flag_exits_with_code_2(tmp_path: Path):
    result = runner.invoke(app, ["--config", str(tmp_path / "missing.yaml"), "version"])
    assert result.exit_code == 2
