"""Timezone and DST handling.

Policy: UTC is the key, local time is derived. Naive timestamps are never accepted silently:
they must be localised with an explicit DST policy. Hourly grids are built in UTC so that
every local day has 23, 24 or 25 hours as the calendar dictates.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Literal
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import numpy as np
import pandas as pd

AmbiguousPolicy = Literal["raise", "dst", "standard", "nat"]
NonexistentPolicy = Literal["raise", "shift_forward", "nat"]


class TimeError(ValueError):
    """Invalid or unsafe timestamp handling."""


def validate_timezone(name: str) -> ZoneInfo:
    try:
        return ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        raise TimeError(f"Unknown IANA timezone: {name!r}") from None


def ensure_aware(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the index unchanged if timezone-aware, otherwise raise."""
    if index.tz is None:
        raise TimeError("Naive timestamps are not accepted: localise them with an explicit policy")
    return index


def to_utc(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return ensure_aware(index).tz_convert("UTC")


def localize_naive(
    index: pd.DatetimeIndex,
    timezone: str,
    ambiguous: AmbiguousPolicy = "raise",
    nonexistent: NonexistentPolicy = "raise",
) -> pd.DatetimeIndex:
    """Attach a timezone to naive local timestamps with an explicit DST policy.

    ambiguous (hour repeated in autumn): raise | dst (first occurrence) |
        standard (second occurrence) | nat (drop to NaT).
    nonexistent (hour skipped in spring): raise | shift_forward | nat.
    """
    if index.tz is not None:
        raise TimeError("Index is already timezone-aware")
    validate_timezone(timezone)
    amb_map: dict[str, object] = {
        "raise": "raise",
        "nat": "NaT",
        "dst": np.ones(len(index), dtype=bool),
        "standard": np.zeros(len(index), dtype=bool),
    }
    non_map = {"raise": "raise", "shift_forward": "shift_forward", "nat": "NaT"}
    if ambiguous not in amb_map:
        raise TimeError(f"Unknown ambiguous policy: {ambiguous!r}")
    if nonexistent not in non_map:
        raise TimeError(f"Unknown nonexistent policy: {nonexistent!r}")
    try:
        return index.tz_localize(
            timezone, ambiguous=amb_map[ambiguous], nonexistent=non_map[nonexistent]
        )
    except Exception as exc:  # pandas raises AmbiguousTimeError / NonExistentTimeError
        raise TimeError(
            f"DST problem while localising to {timezone}: {exc}. "
            "Set time_policy.ambiguous / time_policy.nonexistent explicitly."
        ) from exc


def add_local_columns(df: pd.DataFrame, timezone: str, utc_col: str = "ts_utc") -> pd.DataFrame:
    """Return a copy with `ts_local` and `utc_offset_min` derived from `ts_utc`.

    `ts_local` is the local wall-clock time WITHOUT a timezone (a column can hold only one
    timezone, but a dataset spans many); `utc_offset_min` makes it unambiguous, also during the
    repeated hour of the autumn DST change. The IANA zone lives in the building metadata.
    """
    validate_timezone(timezone)
    series = df[utc_col]
    if not isinstance(series.dtype, pd.DatetimeTZDtype) or str(series.dt.tz) != "UTC":
        raise TimeError(f"Column {utc_col!r} must be timezone-aware UTC")
    out = df.copy()
    wall = series.dt.tz_convert(timezone).dt.tz_localize(None)
    utc_wall = series.dt.tz_localize(None)
    out["ts_local"] = wall
    out["utc_offset_min"] = ((wall - utc_wall).dt.total_seconds() // 60).astype("int64")
    return out


def utc_grid(
    start: pd.Timestamp | str,
    end: pd.Timestamp | str,
    freq: str = "h",
    inclusive: Literal["both", "neither", "left", "right"] = "left",
) -> pd.DatetimeIndex:
    """Regular UTC grid. `start` and `end` must be timezone-aware (or strings with an offset)."""
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if start_ts.tzinfo is None or end_ts.tzinfo is None:
        raise TimeError("Grid bounds must be timezone-aware")
    return pd.date_range(
        start_ts.tz_convert("UTC"), end_ts.tz_convert("UTC"), freq=freq, inclusive=inclusive
    )


def local_day_hours(timezone: str, start: date, end: date) -> pd.Series:
    """Number of real hours in each local calendar day in [start, end). DST days are 23 or 25."""
    validate_timezone(timezone)
    first = pd.Timestamp(start).tz_localize(timezone)
    last = pd.Timestamp(end).tz_localize(timezone)
    grid = pd.date_range(
        first.tz_convert("UTC"), last.tz_convert("UTC"), freq="h", inclusive="left"
    )
    local_dates = grid.tz_convert(timezone).date
    counts = pd.Series(1, index=pd.Index(local_dates, name="local_date")).groupby(level=0).sum()
    return counts.rename("hours")


def dst_transition_days(timezone: str, start: date, end: date) -> list[date]:
    """Local dates in [start, end) whose length is not 24 hours."""
    hours = local_day_hours(timezone, start, end)
    return [d for d, n in hours.items() if n != 24]


def standard_utc_offset(timezone: str, year: int) -> pd.Timedelta:
    """UTC offset of the zone's STANDARD (winter, non-DST) time in a given year.

    Taken as the smaller of the offsets on 1 January and 1 July, which covers both hemispheres.
    This is deliberately not `utcoffset() - dst()`: tzdata models Irish winter time as negative
    DST, so that formula would return UTC+1 for Dublin all year.
    """
    zone = validate_timezone(timezone)
    offsets = [datetime(year, month, 1, 12, tzinfo=zone).utcoffset() for month in (1, 7)]
    offset = min(o for o in offsets if o is not None)
    return pd.Timedelta(offset)


def standard_time_to_utc(index: pd.DatetimeIndex, timezone: str) -> pd.DatetimeIndex:
    """Convert naive local STANDARD time (DST ignored, fixed offset all year) to UTC.

    Used for datasets whose documentation says "local time, daylight saving ignored". Such a
    clock never repeats or skips an hour, so no DST policy is needed, but the offset is the
    zone's winter offset even in summer.
    """
    if index.tz is not None:
        raise TimeError("Index is already timezone-aware")
    validate_timezone(timezone)
    years = index.year
    offsets = pd.Series(years, dtype="int64").map(
        {int(y): standard_utc_offset(timezone, int(y)) for y in pd.unique(years)}
    )
    shifted = index - pd.TimedeltaIndex(offsets.to_numpy())
    return shifted.tz_localize(UTC)
