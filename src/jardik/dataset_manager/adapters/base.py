"""Pieces shared by every dataset adapter."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from jardik.core.timeutils import add_local_columns


class DatasetError(RuntimeError):
    """A dataset is missing, unreadable or inconsistent."""


LOAD_COLUMNS = [
    "building_id",
    "meter_type",
    "appliance",
    "ts_utc",
    "ts_local",
    "utc_offset_min",
    "value",
    "unit",
    "granularity_s",
    "is_missing",
    "is_imputed",
    "quality_flag",
]


def wide_to_load_frame(
    values: pd.DataFrame,
    timezones: Mapping[str, str],
    *,
    imputed: pd.DataFrame | None = None,
    meter_type: str = "electricity_total",
    unit: str = "kWh",
    granularity_s: int = 3600,
) -> pd.DataFrame:
    """Turn a wide frame (UTC index, one column per building) into LOAD_CONTRACT rows.

    A NaN is a missing slot. Cells flagged in `imputed` carry a value that was filled in, which
    is recorded with is_imputed and quality_flag = 'imputed'. Rows are never dropped: a missing
    slot stays visible so that gaps can be counted and reported.
    """
    index_name = values.index.name or "ts_utc"
    frames: list[pd.DataFrame] = []
    by_zone: dict[str, list[str]] = {}
    for building in values.columns:
        by_zone.setdefault(timezones[building], []).append(building)

    for zone, buildings in by_zone.items():
        long = (
            values[buildings]
            .rename_axis(index="ts_utc")
            .reset_index()
            .melt(id_vars="ts_utc", var_name="building_id", value_name="value")
        )
        if imputed is None:
            was_imputed = pd.Series(False, index=long.index)
        else:
            flags = (
                imputed[buildings]
                .rename_axis(index="ts_utc")
                .reset_index()
                .melt(id_vars="ts_utc", var_name="building_id", value_name="imputed")
            )
            was_imputed = flags["imputed"].astype(bool).reset_index(drop=True)
        missing = long["value"].isna()
        long["meter_type"] = meter_type
        long["appliance"] = pd.Series([None] * len(long), dtype="object")
        long["unit"] = unit
        long["granularity_s"] = granularity_s
        long["is_missing"] = missing
        long["is_imputed"] = was_imputed.to_numpy()
        long["quality_flag"] = "ok"
        long.loc[long["is_imputed"], "quality_flag"] = "imputed"
        long.loc[missing, "quality_flag"] = "missing"
        frames.append(add_local_columns(long, zone))

    if not frames:
        return pd.DataFrame({c: pd.Series(dtype="object") for c in LOAD_COLUMNS})
    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["building_id", "ts_utc"], kind="stable").reset_index(drop=True)
    del index_name
    return out[LOAD_COLUMNS]
