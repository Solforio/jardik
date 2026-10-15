from datetime import date

import pandas as pd
import pytest

from jardik.dataset_manager.interactive import build_request_interactively

INFO = {
    "dataset_id": "bdg2",
    "name": "Building Data Genome Project 2 (BDG2)",
    "supported_tasks": ["forecasting"],
    "native_granularity_s": 3600,
    "time_coverage_local": ["2016-01-01T00:00:00", "2017-12-31T23:00:00"],
}
CATALOGUE = pd.DataFrame(
    {
        "building_id": ["bdg2:a", "bdg2:b", "bdg2:c"],
        "site_id": ["S1", "S1", "S2"],
        "usage_category": ["office", "office", "shop"],
        "country": [None, None, None],
    }
)


def scripted(answers: dict[str, str]):
    """A prompt function that answers by label and falls back to the proposed default."""
    asked: list[str] = []

    def prompt(text: str, default=None, **_kwargs):
        label = text.split(" [")[0].split(" (")[0]
        asked.append(label)
        return answers.get(label, "" if default is None else default)

    prompt.asked = asked  # type: ignore[attr-defined]
    return prompt


def test_defaults_alone_need_only_a_name_and_a_period():
    request = build_request_interactively(
        INFO,
        CATALOGUE,
        scripted(
            {"Extraction name": "my_run", "Start date": "2016-01-01", "End date": "2016-07-01"}
        ),
    )
    assert request.name == "my_run" and request.dataset_id == "bdg2"
    assert request.task == "forecasting" and request.granularity_s == 3600
    assert request.start == date(2016, 1, 1) and request.end == date(2016, 7, 1)
    assert request.usage_categories == [] and request.n_buildings is None


def test_answers_are_parsed_into_the_request():
    answers = {
        "Extraction name": "offices",
        "Usage categories": "office, shop",
        "Sites": "S1",
        "Minimum location quality": "coordinates",
        "Number of buildings": "25",
        "Sampling": "stratified_site",
        "Start date": "2016-03-01",
        "End date": "2016-04-01",
        "Max missing ratio": "0.05",
        "Imputation": "linear_short_gaps",
        "Max gap hours": "6",
        "Weather variables": "temp_c, wind_ms",
        "Matching method": "nearest_previous",
        "Max time offset minutes": "45",
        "Export format": "csv",
        "CSV style": "plain",
    }
    request = build_request_interactively(INFO, CATALOGUE, scripted(answers))
    assert request.usage_categories == ["office", "shop"]
    assert request.area.site_ids == ["S1"]
    assert request.min_location_quality == "coordinates"
    assert request.n_buildings == 25 and request.sampling == "stratified_site"
    assert request.quality.max_missing_ratio == 0.05
    assert request.quality.imputation == "linear_short_gaps" and request.quality.max_gap_hours == 6
    assert request.weather.variables == ["temp_c", "wind_ms"]
    assert request.weather.matching.method == "nearest_previous"
    assert request.weather.matching.max_time_offset_min == 45
    assert request.export.format == "csv" and request.export.csv_style == "plain"


def test_follow_up_questions_are_only_asked_when_relevant():
    prompt = scripted(
        {"Extraction name": "x", "Start date": "2016-01-01", "End date": "2016-02-01"}
    )
    build_request_interactively(INFO, CATALOGUE, prompt)
    assert "Max gap hours" not in prompt.asked  # imputation stayed 'none'
    assert "Max time offset minutes" not in prompt.asked  # matching stayed 'exact'
    assert "CSV style" not in prompt.asked  # parquet export
    assert "Task" not in prompt.asked  # the dataset supports a single task


def test_invalid_answers_are_asked_again_not_crashed():
    calls = {"n": 0}

    def prompt(text, default=None, **_kwargs):
        label = text.split(" [")[0].split(" (")[0]
        if label == "Extraction name":
            calls["n"] += 1
            return "bad name!" if calls["n"] == 1 else "good_name"
        return {"Start date": "2016-01-01", "End date": "2016-02-01"}.get(
            label, "" if default is None else default
        )

    request = build_request_interactively(INFO, CATALOGUE, prompt)
    assert request.name == "good_name" and calls["n"] == 2


def test_gives_up_after_repeated_invalid_answers():
    def prompt(text, default=None, **_kwargs):
        return "bad name!" if text.startswith("Extraction name") else (default or "")

    with pytest.raises(ValueError, match="Extraction name"):
        build_request_interactively(INFO, CATALOGUE, prompt)
