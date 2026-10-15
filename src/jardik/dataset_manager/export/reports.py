"""Quality, distribution and funnel reports (plain data structures, written as JSON/CSV)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from jardik.dataset_manager.matching.policy import MatchPolicy


def funnel_frame(funnel: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(funnel, columns=["step", "remaining", "removed", "detail"])


def quantiles(series: pd.Series) -> dict[str, float] | None:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return None
    return {
        "min": float(values.min()),
        "p25": float(values.quantile(0.25)),
        "median": float(values.median()),
        "p75": float(values.quantile(0.75)),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def build_distribution_report(
    *,
    available: int,
    eligible: int,
    selected: pd.DataFrame,
    merged: pd.DataFrame,
) -> dict[str, Any]:
    """How the selected buildings are distributed (sites, usage, size, load level, profile)."""
    mean_load = merged.groupby("building_id")["target_kwh"].mean()
    profile = merged.groupby("hour_local")["target_kwh"].mean()
    return {
        "buildings": {
            "available": available,
            "eligible_after_filters": eligible,
            "selected": len(selected),
        },
        "by_site": selected["site_id"].value_counts().sort_index().to_dict(),
        "by_usage_category": selected["usage_category"].value_counts().sort_index().to_dict(),
        "by_location_quality": selected["location_quality"].value_counts().sort_index().to_dict(),
        "floor_area_m2": quantiles(selected["floor_area_m2"]),
        "mean_load_kwh_per_hour": quantiles(mean_load),
        "residual_missing_ratio": quantiles(selected["residual_missing_ratio"]),
        "hourly_profile_kwh": {int(h): float(v) for h, v in profile.items()},
    }


def build_quality_report(
    *,
    extraction_id: str,
    request_window: tuple[str, str],
    granularity_s: int,
    selected_stats: pd.DataFrame,
    contracts: dict[str, str],
    duplicates: dict[str, int],
    geo_issues: pd.DataFrame,
    corrections: list[dict[str, Any]],
    policy: MatchPolicy,
    match_stats: dict[str, Any],
    join_report: dict[str, Any],
    weather_screening: dict[str, dict[str, int]],
    excluded_count: int,
) -> dict[str, Any]:
    stats_cols = [
        "building_id",
        "expected_rows",
        "present_rows",
        "missing_rows",
        "imputed_rows",
        "residual_missing_rows",
        "missing_ratio",
        "imputed_ratio",
        "residual_missing_ratio",
        "longest_gap_h",
        "zero_ratio",
        "near_zero_ratio",
        "first_value_ts",
        "last_value_ts",
    ]
    present_first = selected_stats["first_value_ts"].dropna()
    present_last = selected_stats["last_value_ts"].dropna()
    return {
        "extraction_id": extraction_id,
        "contracts": contracts,
        "duplicates": duplicates,
        "temporal_coverage": {
            "window_start": request_window[0],
            "window_end": request_window[1],
            "granularity_s": granularity_s,
            "slots_per_building": int(selected_stats["expected_rows"].iloc[0])
            if len(selected_stats)
            else 0,
            "first_value": present_first.min() if len(present_first) else None,
            "last_value": present_last.max() if len(present_last) else None,
        },
        "summary": {
            "missing_ratio": quantiles(selected_stats["missing_ratio"]),
            "imputed_ratio": quantiles(selected_stats["imputed_ratio"]),
            "residual_missing_ratio": quantiles(selected_stats["residual_missing_ratio"]),
            "longest_gap_h": quantiles(selected_stats["longest_gap_h"]),
        },
        "buildings": selected_stats[stats_cols].to_dict("records"),
        "geographic_consistency": {
            "issues": geo_issues.to_dict("records"),
            "corrections_applied": corrections,
        },
        "weather_matching": {
            "policy": policy.model_dump(mode="json"),
            "stats": match_stats,
            "coverage_by_building": join_report["coverage_by_building"].to_dict(),
            "dst_days": join_report["dst_days"]
            .assign(local_date=lambda d: d["local_date"].astype(str))
            .to_dict("records"),
        },
        "weather_value_rules": weather_screening,
        "excluded_buildings": excluded_count,
    }
