"""Temporal step: join weather records to load records by timestamp, under a declared policy.

Everything is done on UTC timestamps, so DST can never duplicate or drop a record: a 23 or 25
hour local day is simply 23 or 25 consecutive UTC hours.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from jardik.dataset_manager.matching.policy import MatchPolicy

_NS = "datetime64[ns, UTC]"
_MATCH_COLUMNS = [
    "building_id",
    "ts_utc",
    "weather_source",
    "matched_lat",
    "matched_lon",
    "distance_km",
    "weather_time_offset_min",
    "timezone_used",
    "matching_method",
    "spatial_link",
    "location_quality",
]


class MatchInputError(ValueError):
    """The inputs to the join are unusable (for example forecast weather)."""


@dataclass
class MatchResult:
    """`match` follows WEATHER_MATCH_CONTRACT. `values` holds the matched weather variables,
    row for row in the same order as `match`."""

    match: pd.DataFrame
    values: pd.DataFrame
    stats: dict[str, Any]


def _asof(
    left: pd.DataFrame, right: pd.DataFrame, direction: str, tolerance_min: int
) -> pd.DataFrame:
    return pd.merge_asof(
        left,
        right,
        on="ts_utc",
        by="weather_point_id",
        direction=direction,  # type: ignore[arg-type]
        tolerance=pd.Timedelta(minutes=tolerance_min),
    )


def _join(
    left: pd.DataFrame, right: pd.DataFrame, policy: MatchPolicy, variables: list[str]
) -> pd.DataFrame:
    """Return `left` plus columns w_ts, source and the weather variables (NaT/NaN if unmatched)."""
    method = policy.method.value
    right = right.assign(w_ts=right["ts_utc"])
    if method in ("exact", "nearest_previous"):
        direction = "backward"
    elif method == "nearest":
        direction = "nearest"
    else:
        direction = ""

    if method == "interpolate":
        prev = _asof(
            left, right.rename(columns={"w_ts": "w_prev"}), "backward", policy.tolerance_min
        )
        nxt = _asof(left, right.rename(columns={"w_ts": "w_next"}), "forward", policy.tolerance_min)
        out = prev.copy()
        out["w_next"] = nxt["w_next"]
        both = out["w_prev"].notna() & out["w_next"].notna()
        span = (out["w_next"] - out["w_prev"]).dt.total_seconds()
        pos = (out["ts_utc"] - out["w_prev"]).dt.total_seconds()
        weight = np.where(span > 0, pos / span.where(span > 0, 1.0), 0.0)
        for variable in variables:
            lo = out[variable].astype(float)
            hi = nxt[variable].astype(float).to_numpy()
            out[variable] = np.where(both, lo + (hi - lo) * weight, np.nan)
        out["w_ts"] = out["ts_utc"].where(both)  # interpolated value is valid at the load time
        return out

    out = _asof(left, right, direction, policy.tolerance_min)
    return out


def match_weather(
    *,
    load_keys: pd.DataFrame,
    weather: pd.DataFrame,
    assignment: pd.DataFrame,
    buildings: pd.DataFrame,
    policy: MatchPolicy,
    variables: list[str],
) -> MatchResult:
    """Attach weather to load records.

    load_keys:  building_id, ts_utc (one row per load record to enrich)
    weather:    WEATHER_CONTRACT frame (observation or reanalysis)
    assignment: output of assign_weather_points
    buildings:  building_id, timezone, location_quality
    Records without weather inside the policy are left out of `match` and counted in `stats`.
    """
    if (weather["product_type"] == "forecast").any():
        raise MatchInputError(
            "Forecast weather is not supported by the matching step yet: "
            "it needs issue-time-aware joins"
        )

    assigned = assignment[assignment["weather_point_id"].notna()]
    link_cols = [
        "building_id",
        "weather_point_id",
        "matched_lat",
        "matched_lon",
        "distance_km",
        "spatial_link",
    ]
    left_all = load_keys[["building_id", "ts_utc"]].copy()
    left_all["ts_utc"] = left_all["ts_utc"].astype(_NS)
    left = left_all.merge(assigned[link_cols], on="building_id", how="inner")
    rows_without_point = len(left_all) - len(left)

    right = weather[["weather_point_id", "ts_utc", "source", *variables]].copy()
    right["ts_utc"] = right["ts_utc"].astype(_NS)
    # merge_asof needs identical key dtypes, also when one side is empty.
    left["weather_point_id"] = left["weather_point_id"].astype(str)
    right["weather_point_id"] = right["weather_point_id"].astype(str)
    right = right.sort_values("ts_utc", kind="stable")
    left = left.sort_values("ts_utc", kind="stable")

    joined = _join(left, right, policy, variables)
    matched = joined[joined["w_ts"].notna()].copy()

    offset = (matched["w_ts"] - matched["ts_utc"]).dt.total_seconds() / 60.0
    if policy.method.value == "interpolate":
        offset = pd.Series(0.0, index=matched.index)
    tz_map = buildings.set_index("building_id")["timezone"]
    quality_map = buildings.set_index("building_id")["location_quality"]

    match = pd.DataFrame(
        {
            "building_id": matched["building_id"],
            "ts_utc": matched["ts_utc"],
            "weather_source": matched["source"],
            "matched_lat": matched["matched_lat"].astype(float),
            "matched_lon": matched["matched_lon"].astype(float),
            "distance_km": matched["distance_km"].astype(float),
            "weather_time_offset_min": offset.round().astype("int64"),
            "timezone_used": matched["building_id"].map(tz_map),
            "matching_method": policy.method.value,
            "spatial_link": matched["spatial_link"],
            "location_quality": matched["building_id"].map(quality_map),
        }
    )[_MATCH_COLUMNS]
    values = matched[["building_id", "ts_utc", *variables]].astype({v: float for v in variables})

    order = match.sort_values(["building_id", "ts_utc"], kind="stable").index
    match = match.loc[order].reset_index(drop=True)
    values = values.loc[order].reset_index(drop=True)

    stats: dict[str, Any] = {
        "load_rows": int(len(left_all)),
        "rows_without_weather_point": int(rows_without_point),
        "matched_rows": int(len(match)),
        "unmatched_rows": int(len(left) - len(match)),
        "method": policy.method.value,
        "tolerance_min": policy.tolerance_min,
        "value_missing_ratio": {
            v: float(values[v].isna().mean()) if len(values) else float("nan") for v in variables
        },
        "offset_min": _describe(match["weather_time_offset_min"]),
        "distance_km": _describe(match["distance_km"].dropna()),
    }
    return MatchResult(match=match, values=values, stats=stats)


def _describe(series: pd.Series) -> dict[str, float | None]:
    if series.empty:
        return {"min": None, "max": None, "mean": None}
    return {
        "min": float(series.min()),
        "max": float(series.max()),
        "mean": float(series.mean()),
    }
