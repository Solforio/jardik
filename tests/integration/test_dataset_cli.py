"""`jardik dataset ...` against a miniature BDG2 placed where the real one would live."""

import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from jardik.cli import app

runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parents[2]

REQUEST = """\
name: cli_offices
start: 2016-03-08
end: 2016-03-15
usage_categories: [office]
n_buildings: 3
seed: 1
"""


@pytest.fixture
def project(tmp_path, mini_bdg2, monkeypatch) -> Path:
    """A project root with the real bdg2.yaml and the mini dataset in data/raw/bdg2."""
    root = tmp_path / "project"
    (root / "configs" / "datasets").mkdir(parents=True)
    # The shipped file corrects site 'Wolf'; the mini dataset has the equivalent site 'Dub'.
    shipped = (REPO_ROOT / "configs" / "datasets" / "bdg2.yaml").read_text(encoding="utf-8")
    assert "  Wolf:" in shipped
    (root / "configs" / "datasets" / "bdg2.yaml").write_text(
        shipped.replace("  Wolf:", "  Dub:"), encoding="utf-8"
    )
    shutil.copytree(mini_bdg2, root / "data" / "raw" / "bdg2")
    (root / "pyproject.toml").write_text("[project]\nname='x'\n")
    monkeypatch.setenv("JARDIK_HOME", str(root))
    return root


def test_dataset_group_is_listed():
    result = runner.invoke(app, ["dataset", "--help"])
    assert result.exit_code == 0
    for command in ("fetch", "describe", "extract", "extractions", "validate-request", "list"):
        assert command in result.stdout


def test_describe_shows_licence_clock_and_the_declared_correction(project):
    result = runner.invoke(app, ["dataset", "describe", "bdg2"])
    assert result.exit_code == 0, result.output
    assert "CC BY-SA 4.0" in result.stdout
    assert "local_standard" in result.stdout
    assert "buildings with electricity: 12" in result.stdout
    assert "correction: Dub lng 6.26 -> -6.26" in result.stdout


def test_describe_without_data_explains_how_to_fetch(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text("[project]\nname='x'\n")
    monkeypatch.setenv("JARDIK_HOME", str(tmp_path))
    result = runner.invoke(app, ["dataset", "describe", "bdg2"])
    assert result.exit_code == 1
    assert "dataset fetch" in result.output


def test_extract_from_a_request_file_then_list_it(project):
    request = project / "req.yaml"
    request.write_text(REQUEST, encoding="utf-8")
    assert runner.invoke(app, ["dataset", "validate-request", str(request)]).exit_code == 0

    result = runner.invoke(app, ["dataset", "extract", "--request", str(request)])
    assert result.exit_code == 0, result.output
    assert "buildings 3" in result.stdout
    assert (project / "outputs" / "extractions").is_dir()

    listing = runner.invoke(app, ["dataset", "extractions"])
    assert "cli_offices-" in listing.stdout and "3 buildings" in listing.stdout

    again = runner.invoke(app, ["dataset", "extract", "--request", str(request)])
    assert again.exit_code == 1 and "force" in again.output
    assert (
        runner.invoke(app, ["dataset", "extract", "--request", str(request), "--force"]).exit_code
        == 0
    )


def test_invalid_request_exits_with_code_2(project):
    bad = project / "bad.yaml"
    bad.write_text("name: x\nstart: 2016-03-10\nend: 2016-03-01\n", encoding="utf-8")
    assert runner.invoke(app, ["dataset", "validate-request", str(bad)]).exit_code == 2
    assert runner.invoke(app, ["dataset", "extract", "--request", str(bad)]).exit_code == 2


def test_extract_needs_exactly_one_input_mode(project):
    assert runner.invoke(app, ["dataset", "extract"]).exit_code == 2


def test_interactive_extraction(project):
    answers = "\n".join(
        [
            "bdg2",  # Dataset
            "inter",  # Extraction name
            "office",  # Usage categories
            "",  # Sites
            "",  # Minimum location quality
            "2",  # Number of buildings
            "",  # Sampling
            "2016-03-08",  # Start date
            "2016-03-12",  # End date
            "",  # Max missing ratio
            "",  # Imputation
            "",  # Weather variables
            "",  # Matching method
            "",  # Export format
            "y",  # Run now?
        ]
    )
    result = runner.invoke(app, ["dataset", "extract", "--interactive"], input=answers + "\n")
    assert result.exit_code == 0, result.output
    assert "buildings 2" in result.stdout
    assert (project / "outputs" / "requests" / "inter.yaml").is_file()


def test_review_command_builds_the_readable_folder(project):
    request = project / "req.yaml"
    request.write_text(REQUEST, encoding="utf-8")
    assert runner.invoke(app, ["dataset", "extract", "--request", str(request)]).exit_code == 0
    result = runner.invoke(app, ["dataset", "review"])  # newest extraction by default
    assert result.exit_code == 0, result.output
    assert "extraction_tables.xlsx" in result.stdout and "extraction_report.pdf" in result.stdout
    folders = list((project / "review").iterdir())
    assert len(folders) == 1 and folders[0].name.startswith("cli_offices-")


def test_review_without_extractions_or_with_unknown_id(project):
    assert runner.invoke(app, ["dataset", "review"]).exit_code == 1
    request = project / "req.yaml"
    request.write_text(REQUEST, encoding="utf-8")
    runner.invoke(app, ["dataset", "extract", "--request", str(request)])
    assert runner.invoke(app, ["dataset", "review", "nope-123"]).exit_code == 1
