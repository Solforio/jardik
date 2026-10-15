"""Final validation of the temporal join (required by the brief)."""

from __future__ import annotations

from datetime import timedelta
from typing import Any

import pandas as pd

from jardik.core.schemas import ContractError, validate_weather_match
from jardik.core.timeutils import local_day_hours
from jardik.dataset_manager.matching.policy import MatchPolicy
from jardik.dataset_manager.matching.temporal import MatchResult


class MatchingError(RuntimeError):
    """The join is structurally wrong. Low coverage is reported, not raised."""


def validate_join(
    result: MatchResult,
    load_keys: pd.DataFrame,
    policy: MatchPolicy,
    buildings: pd.DataFrame,
) -> dict[str, Any]:
    """Check the join and return a report.

    Raises MatchingError for structural problems: duplicated matches, matches that do not belong
    to a load record, offsets larger than the policy allows, or contract violations.
    Returns coverage per building and the DST-day audit; deciding what to do with low-coverage
    buildings is left to the caller.
    """
    match = result.match
    key = ["building_id", "ts_utc"]

    duplicated = int(match.duplicated(subset=key).sum())
    if duplicated:
        raise MatchingError(f"{duplicated} duplicate match row(s) on {key}")

    orphan = match.merge(load_keys[key].drop_duplicates(), on=key, how="left", indicator=True)
    n_orphan = int((orphan["_merge"] == "left_only").sum())
    if n_orphan:
        raise MatchingError(f"{n_orphan} match row(s) have no corresponding load record")

    too_far = match["weather_time_offset_min"].abs() > policy.tolerance_min
    if too_far.any():
        worst = int(match["weather_time_offset_min"].abs().max())
        raise MatchingError(
            f"{int(too_far.sum())} match row(s) have a time offset beyond the policy "
            f"({policy.tolerance_min} min); worst offset {worst} min"
        )

    try:
        validate_weather_match(match)
    except ContractError as exc:
        raise MatchingError(str(exc)) from exc

    loads = load_keys.groupby("building_id").size()
    matched = match.groupby("building_id").size().reindex(loads.index, fill_value=0)
    coverage = (matched / loads).rename("coverage")

    return {
        "rows": int(len(match)),
        "coverage_by_building": coverage,
        "dst_days": _dst_day_audit(load_keys, match, buildings),
    }


def _dst_day_audit(
    load_keys: pd.DataFrame, match: pd.DataFrame, buildings: pd.DataFrame
) -> pd.DataFrame:
    """For every local day whose length is not 24 h, compare load and matched record counts."""
    columns = ["building_id", "local_date", "expected_hours", "load_rows", "matched_rows"]
    tz_by_building = buildings.set_index("building_id")["timezone"]
    day_cache: dict[tuple[str, object, object], pd.Series] = {}
    out: list[dict[str, Any]] = []

    for building, group in load_keys.groupby("building_id"):
        tz = tz_by_building.get(building)
        if tz is None:
            continue
        local = group["ts_utc"].dt.tz_convert(tz)
        first, last = local.min().date(), local.max().date() + timedelta(days=1)
        cache_key = (tz, first, last)
        if cache_key not in day_cache:
            hours = local_day_hours(tz, first, last)
            day_cache[cache_key] = hours[hours != 24]
        odd_days = day_cache[cache_key]
        if odd_days.empty:
            continue
        load_days = local.dt.date.value_counts()
        matched_ts = match.loc[match["building_id"] == building, "ts_utc"].dt.tz_convert(tz)
        matched_days = matched_ts.dt.date.value_counts()
        for day, expected in odd_days.items():
            if day in load_days.index:
                out.append(
                    {
                        "building_id": building,
                        "local_date": day,
                        "expected_hours": int(expected),
                        "load_rows": int(load_days.get(day, 0)),
                        "matched_rows": int(matched_days.get(day, 0)),
                    }
                )
    return pd.DataFrame(out, columns=columns)
