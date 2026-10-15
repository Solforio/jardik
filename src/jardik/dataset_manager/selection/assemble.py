"""Assemble the final selection: load data, weather assignment and matching, with replacement.

Candidates are taken in their deterministic order. A candidate that cannot be tied to weather,
or whose weather coverage is too low, is recorded as excluded and replaced by the next one, so
the extraction reaches the requested number of buildings whenever enough candidates exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import pandas as pd

from jardik.core.interfaces import DatasetAdapter, TimeWindow
from jardik.core.logging import get_logger
from jardik.dataset_manager.adapters.base import wide_to_load_frame
from jardik.dataset_manager.matching import (
    assign_weather_points,
    match_weather,
    validate_join,
)
from jardik.dataset_manager.request import ExtractionRequest
from jardik.dataset_manager.selection.quality_screen import gap_limit_rows, impute_wide

log = get_logger("dataset_manager.assemble")


@dataclass
class Assembled:
    accepted: list[str] = field(default_factory=list)  # in rank order
    load: pd.DataFrame = field(default_factory=pd.DataFrame)
    weather: pd.DataFrame = field(default_factory=pd.DataFrame)
    assignment: pd.DataFrame = field(default_factory=pd.DataFrame)
    excluded: list[dict[str, Any]] = field(default_factory=list)
    evaluated: int = 0
    rejected_assignment: int = 0
    rejected_coverage: int = 0
    reject_reasons: dict[str, int] = field(default_factory=dict)
    weather_screening: dict[str, dict[str, int]] = field(default_factory=dict)


def padded_window(request: ExtractionRequest) -> TimeWindow:
    pad = timedelta(minutes=request.weather.matching.tolerance_min, hours=1)
    window = request.window
    return TimeWindow(window.start - pad, window.end + pad)


def _merge_screening(total: dict[str, dict[str, int]], new: dict[str, dict[str, int]]) -> None:
    for variable, rules in new.items():
        bucket = total.setdefault(variable, {})
        for rule, count in rules.items():
            bucket[rule] = bucket.get(rule, 0) + count


def assemble_selection(
    adapter: DatasetAdapter,
    request: ExtractionRequest,
    pool: pd.DataFrame,
    order: list[str],
    target: int,
) -> Assembled:
    """Walk through `order` until `target` buildings are fully usable (or candidates run out)."""
    policy = request.weather.matching
    variables = request.weather.variables
    points = adapter.weather_points()
    window = request.window
    weather_window = padded_window(request)
    gap_rows = gap_limit_rows(request)
    meta = pool.set_index("building_id", drop=False)
    timezone_of = meta["timezone"].to_dict()

    result = Assembled()
    weather_cache: dict[str, pd.DataFrame] = {}
    load_parts: list[pd.DataFrame] = []
    assignment_parts: list[pd.DataFrame] = []
    cursor = 0

    while len(result.accepted) < target and cursor < len(order):
        chunk = order[cursor : cursor + (target - len(result.accepted))]
        cursor += len(chunk)
        result.evaluated += len(chunk)
        chunk_meta = meta.loc[chunk].reset_index(drop=True)

        assignment = assign_weather_points(
            chunk_meta[["building_id", "lat", "lon", "weather_link"]],
            points,
            policy.max_distance_km,
        )
        rejected = assignment[assignment["weather_point_id"].isna()]
        for row in rejected.itertuples(index=False):
            result.rejected_assignment += 1
            result.reject_reasons[row.reject_reason] = (
                result.reject_reasons.get(row.reject_reason, 0) + 1
            )
            result.excluded.append(
                {
                    "building_id": row.building_id,
                    "step": "weather_assignment",
                    "detail": row.reject_reason,
                }
            )
        accepted = assignment[assignment["weather_point_id"].notna()]
        if accepted.empty:
            continue

        needed = sorted(set(accepted["weather_point_id"]) - set(weather_cache))
        if needed:
            loaded = adapter.load_weather(needed, weather_window, variables)
            for point, frame in loaded.groupby("weather_point_id"):
                weather_cache[str(point)] = frame
            _merge_screening(result.weather_screening, adapter.weather_screening)
            for point in needed:  # a point with no rows must still be known to be empty
                weather_cache.setdefault(point, loaded.iloc[0:0])

        ids = accepted["building_id"].tolist()
        wide = adapter.load_wide(ids, window)
        if gap_rows:
            wide, imputed = impute_wide(wide, gap_rows)
        else:
            imputed = pd.DataFrame(False, index=wide.index, columns=wide.columns)
        load = wide_to_load_frame(
            wide, timezone_of, imputed=imputed, granularity_s=request.granularity_s
        )

        weather = pd.concat([weather_cache[p] for p in accepted["weather_point_id"].unique()])
        chunk_buildings = chunk_meta[["building_id", "timezone", "location_quality"]]
        keys = load[["building_id", "ts_utc"]]
        matched = match_weather(
            load_keys=keys,
            weather=weather,
            assignment=accepted,
            buildings=chunk_buildings,
            policy=policy,
            variables=variables,
        )
        report = validate_join(matched, keys, policy, chunk_buildings)
        coverage = report["coverage_by_building"]

        too_low = set(coverage[coverage < policy.min_coverage].index)
        for building in ids:
            if building in too_low:
                result.rejected_coverage += 1
                result.excluded.append(
                    {
                        "building_id": building,
                        "step": "weather_coverage",
                        "detail": f"weather coverage {coverage[building]:.3f} "
                        f"< {policy.min_coverage}",
                    }
                )
        keep = [b for b in ids if b not in too_low]
        result.accepted += keep
        load_parts.append(load[load["building_id"].isin(keep)])
        assignment_parts.append(accepted[accepted["building_id"].isin(keep)])

    if load_parts:
        result.load = pd.concat(load_parts, ignore_index=True)
        result.assignment = pd.concat(assignment_parts, ignore_index=True)
        used = result.assignment["weather_point_id"].unique()
        result.weather = pd.concat([weather_cache[p] for p in used], ignore_index=True)
    return result
