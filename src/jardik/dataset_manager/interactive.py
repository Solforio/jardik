"""Interactive construction of an ExtractionRequest (answers become the same model as a YAML file).

The prompt function is injected, so the dialogue is testable without a terminal; the CLI passes
`typer.prompt`. An invalid answer is explained and asked again, up to three times.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date, timedelta
from typing import Any

import pandas as pd

from jardik.core.schemas import WEATHER_VARIABLES, LocationQuality, UsageCategory
from jardik.dataset_manager.request import DEFAULT_WEATHER_VARIABLES, ExtractionRequest

Prompt = Callable[..., str]
Echo = Callable[[str], Any]
MAX_TRIES = 3


def _choice(options: list[str]) -> Callable[[str], str]:
    def convert(raw: str) -> str:
        if raw not in options:
            raise ValueError(f"choose one of: {', '.join(options)}")
        return raw

    return convert


def _csv(raw: str) -> list[str]:
    return [part.strip() for part in raw.split(",") if part.strip()]


def _ratio(raw: str) -> float:
    value = float(raw.replace(",", "."))
    if not 0 <= value <= 1:
        raise ValueError("enter a number between 0 and 1")
    return value


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise ValueError("enter a positive whole number")
    return value


def build_request_interactively(
    info: dict[str, Any],
    catalogue: pd.DataFrame,
    prompt: Prompt,
    echo: Echo = print,
) -> ExtractionRequest:
    """Ask the questions of the selection dialogue and return a validated request."""

    def ask(
        label: str,
        default: str = "",
        hint: str = "",
        convert: Callable[[str], Any] = lambda raw: raw,
    ) -> Any:
        text = f"{label} ({hint})" if hint else label
        for _ in range(MAX_TRIES):
            raw = str(prompt(text, default=default)).strip()
            try:
                return convert(raw)
            except ValueError as exc:
                echo(f"  Invalid answer: {exc}")
        raise ValueError(f"Too many invalid answers for '{label}'")

    def name_ok(raw: str) -> str:
        if not re.fullmatch(r"[A-Za-z0-9_\-]+", raw):
            raise ValueError("use letters, digits, '_' and '-' only")
        return raw

    fields: dict[str, Any] = {"dataset_id": info["dataset_id"]}
    fields["name"] = ask("Extraction name", hint="e.g. offices_2016", convert=name_ok)

    tasks = list(info["supported_tasks"])
    if len(tasks) > 1:
        fields["task"] = ask("Task", default=tasks[0], hint="/".join(tasks), convert=_choice(tasks))
    else:
        fields["task"] = tasks[0]

    usage_counts = catalogue["usage_category"].value_counts().to_dict()
    usage_hint = ", ".join(f"{k}: {v}" for k, v in sorted(usage_counts.items()))
    valid_usage = [u.value for u in UsageCategory]

    def usages(raw: str) -> list[str]:
        values = _csv(raw)
        bad = [v for v in values if v not in valid_usage]
        if bad:
            raise ValueError(f"unknown {bad}; valid: {', '.join(valid_usage)}")
        return values

    fields["usage_categories"] = ask(
        "Usage categories", hint=f"blank = all; available {usage_hint}", convert=usages
    )

    sites = sorted(catalogue["site_id"].dropna().unique())

    def site_list(raw: str) -> list[str]:
        values = _csv(raw)
        bad = [v for v in values if v not in sites]
        if bad:
            raise ValueError(f"unknown site(s) {bad}; available: {', '.join(sites[:15])}")
        return values

    area: dict[str, Any] = {
        "site_ids": ask("Sites", hint=f"blank = all of {len(sites)}", convert=site_list)
    }
    if catalogue["country"].notna().any():
        countries = sorted(catalogue["country"].dropna().unique())
        area["countries"] = ask(
            "Countries", hint=f"blank = all; {', '.join(countries)}", convert=_csv
        )
    fields["area"] = area

    qualities = [q.value for q in LocationQuality]
    fields["min_location_quality"] = ask(
        "Minimum location quality",
        hint=f"blank = any; {', '.join(qualities[:-1])}",
        convert=lambda raw: None if not raw else _choice(qualities)(raw),
    )
    fields["n_buildings"] = ask(
        "Number of buildings",
        hint="blank = all that qualify",
        convert=lambda raw: None if not raw else _positive_int(raw),
    )
    if fields["n_buildings"] is not None:
        samplings = ["random", "stratified_site", "first"]
        fields["sampling"] = ask(
            "Sampling", default="random", hint="/".join(samplings), convert=_choice(samplings)
        )

    coverage = info.get("time_coverage_local")
    first = date.fromisoformat(coverage[0][:10]) if coverage else date(2016, 1, 1)
    last = (
        date.fromisoformat(coverage[1][:10]) + timedelta(days=1) if coverage else date(2018, 1, 1)
    )
    start = ask(
        "Start date", default=first.isoformat(), hint="YYYY-MM-DD, UTC", convert=date.fromisoformat
    )

    def end_after(raw: str) -> date:
        value = date.fromisoformat(raw)
        if value <= start:
            raise ValueError("the end date must be after the start date")
        return value

    fields["start"] = start
    fields["end"] = ask(
        "End date", default=last.isoformat(), hint="YYYY-MM-DD, excluded", convert=end_after
    )

    quality: dict[str, Any] = {
        "max_missing_ratio": ask(
            "Max missing ratio", default="0.1", hint="0-1, after imputation", convert=_ratio
        )
    }
    imputations = ["none", "linear_short_gaps"]
    quality["imputation"] = ask(
        "Imputation", default="none", hint="/".join(imputations), convert=_choice(imputations)
    )
    if quality["imputation"] != "none":
        quality["max_gap_hours"] = ask("Max gap hours", default="3", convert=_positive_int)
        quality["max_imputed_ratio"] = ask(
            "Max imputed ratio", default="0.05", hint="0-1", convert=_ratio
        )
    fields["quality"] = quality

    def weather_vars(raw: str) -> list[str]:
        values = _csv(raw)
        bad = [v for v in values if v not in WEATHER_VARIABLES]
        if bad or not values:
            raise ValueError(f"unknown {bad}; valid: {', '.join(sorted(WEATHER_VARIABLES))}")
        return values

    methods = ["exact", "nearest_previous", "nearest", "interpolate"]
    matching: dict[str, Any] = {
        "method": ask(
            "Matching method", default="exact", hint="/".join(methods), convert=_choice(methods)
        )
    }
    if matching["method"] != "exact":
        matching["max_time_offset_min"] = ask(
            "Max time offset minutes", default="60", convert=_positive_int
        )
    fields["weather"] = {
        "variables": ask(
            "Weather variables", default=", ".join(DEFAULT_WEATHER_VARIABLES), convert=weather_vars
        ),
        "matching": matching,
    }

    export: dict[str, Any] = {
        "format": ask(
            "Export format",
            default="parquet",
            hint="parquet/csv",
            convert=_choice(["parquet", "csv"]),
        )
    }
    if export["format"] == "csv":
        export["csv_style"] = ask(
            "CSV style",
            default="spanish",
            hint="spanish: ';' and decimal comma / plain",
            convert=_choice(["spanish", "plain"]),
        )
    fields["export"] = export
    return ExtractionRequest(**fields)
