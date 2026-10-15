"""Quality screening: per-building gap statistics and exclusion of unusable buildings."""

from __future__ import annotations

from typing import Any

import pandas as pd

from jardik.core.interfaces import DatasetAdapter
from jardik.core.logging import get_logger
from jardik.dataset_manager.quality.imputation import impute_short_gaps
from jardik.dataset_manager.quality.stats import summarise_series
from jardik.dataset_manager.request import ExtractionRequest

log = get_logger("dataset_manager.quality_screen")

BATCH = 200


def gap_limit_rows(request: ExtractionRequest) -> int:
    """Maximum gap length in time slots, or 0 when imputation is off."""
    if request.quality.imputation == "none":
        return 0
    return max(1, request.quality.max_gap_hours * 3600 // request.granularity_s)


def impute_wide(wide: pd.DataFrame, max_gap_rows: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply short-gap imputation to every column. Returns (filled values, imputed mask)."""
    filled, mask = {}, {}
    for column in wide.columns:
        filled[column], mask[column] = impute_short_gaps(wide[column], max_gap_rows)
    return pd.DataFrame(filled, index=wide.index), pd.DataFrame(mask, index=wide.index)


def screen_quality(
    adapter: DatasetAdapter, building_ids: list[str], request: ExtractionRequest
) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute gap statistics for every candidate and drop the ones that are not usable.

    Returns (statistics of the buildings that pass, funnel steps, excluded-building records).
    A building passes when its residual missing ratio (gaps left after the optional imputation)
    and its imputed ratio are both within the request limits.
    """
    max_gap = gap_limit_rows(request)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(building_ids), BATCH):
        batch = building_ids[start : start + BATCH]
        wide = adapter.load_wide(batch, request.window)
        for building in batch:
            raw = wide[building]
            filled, imputed = impute_short_gaps(raw, max_gap)
            stats = summarise_series(raw, filled, imputed, request.granularity_s)
            present = raw.dropna().index
            stats.update(
                building_id=building,
                first_value_ts=present.min().isoformat() if len(present) else None,
                last_value_ts=present.max().isoformat() if len(present) else None,
            )
            rows.append(stats)
        log.debug(
            "screened %d/%d buildings", min(start + BATCH, len(building_ids)), len(building_ids)
        )

    stats_df = pd.DataFrame(rows)
    funnel: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    current = stats_df

    for step, column, limit in (
        ("max_missing_ratio", "residual_missing_ratio", request.quality.max_missing_ratio),
        ("max_imputed_ratio", "imputed_ratio", request.quality.max_imputed_ratio),
    ):
        ok = current[column] <= limit
        for row in current[~ok].itertuples(index=False):
            excluded.append(
                {
                    "building_id": row.building_id,
                    "step": step,
                    "detail": f"{column}={getattr(row, column):.3f} > {limit}",
                }
            )
        funnel.append(
            {
                "step": step,
                "remaining": int(ok.sum()),
                "removed": int((~ok).sum()),
                "detail": f"{column} <= {limit}",
            }
        )
        current = current[ok]
    return current.reset_index(drop=True), funnel, excluded
