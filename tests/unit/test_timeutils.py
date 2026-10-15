from datetime import date

import pandas as pd
import pytest

from jardik.core.timeutils import (
    TimeError,
    add_local_columns,
    dst_transition_days,
    ensure_aware,
    local_day_hours,
    localize_naive,
    utc_grid,
    validate_timezone,
)

MADRID = "Europe/Madrid"
# 2024: clocks forward on 31 March (02:00 -> 03:00), back on 27 October (03:00 -> 02:00).
AUTUMN = pd.DatetimeIndex(["2024-10-27 01:30", "2024-10-27 02:30", "2024-10-27 03:30"])
SPRING = pd.DatetimeIndex(["2024-03-31 01:30", "2024-03-31 02:30", "2024-03-31 03:30"])


def test_naive_is_rejected():
    with pytest.raises(TimeError):
        ensure_aware(pd.DatetimeIndex(["2024-01-01"]))


def test_unknown_timezone_raises():
    with pytest.raises(TimeError):
        validate_timezone("Mars/Olympus")


def test_ambiguous_hour_raises_by_default():
    with pytest.raises(TimeError):
        localize_naive(AUTUMN, MADRID)


def test_ambiguous_policies():
    first = localize_naive(AUTUMN, MADRID, ambiguous="dst")
    second = localize_naive(AUTUMN, MADRID, ambiguous="standard")
    # 02:30 local happens twice: +02:00 first, +01:00 second.
    assert first[1].utcoffset().total_seconds() == 7200
    assert second[1].utcoffset().total_seconds() == 3600
    assert first[1] < second[1]
    assert localize_naive(AUTUMN, MADRID, ambiguous="nat").isna().sum() == 1


def test_nonexistent_hour_raises_by_default():
    with pytest.raises(TimeError):
        localize_naive(SPRING, MADRID)


def test_nonexistent_policies():
    shifted = localize_naive(SPRING, MADRID, nonexistent="shift_forward")
    assert shifted[1].hour == 3  # 02:30 does not exist, moved to 03:00
    assert localize_naive(SPRING, MADRID, nonexistent="nat").isna().sum() == 1


def test_already_aware_index_is_rejected():
    with pytest.raises(TimeError):
        localize_naive(pd.DatetimeIndex(["2024-01-01"], tz="UTC"), MADRID)


def test_local_day_lengths_on_dst_days():
    hours = local_day_hours(MADRID, date(2024, 3, 30), date(2024, 4, 2))
    assert hours[date(2024, 3, 30)] == 24
    assert hours[date(2024, 3, 31)] == 23
    assert hours[date(2024, 4, 1)] == 24
    autumn = local_day_hours(MADRID, date(2024, 10, 26), date(2024, 10, 29))
    assert autumn[date(2024, 10, 27)] == 25


def test_dst_transition_days_for_a_year():
    assert dst_transition_days(MADRID, date(2024, 1, 1), date(2025, 1, 1)) == [
        date(2024, 3, 31),
        date(2024, 10, 27),
    ]
    assert dst_transition_days("UTC", date(2024, 1, 1), date(2025, 1, 1)) == []


def test_add_local_columns_offsets():
    df = pd.DataFrame(
        {"ts_utc": pd.to_datetime(["2024-01-15 12:00", "2024-07-15 12:00"], utc=True)}
    )
    out = add_local_columns(df, MADRID)
    assert out["utc_offset_min"].tolist() == [60, 120]
    assert out["ts_local"].dt.hour.tolist() == [13, 14]
    assert "ts_local" not in df.columns  # input untouched


def test_add_local_columns_requires_utc():
    df = pd.DataFrame({"ts_utc": pd.to_datetime(["2024-01-15 12:00"])})
    with pytest.raises(TimeError):
        add_local_columns(df, MADRID)


def test_utc_grid():
    grid = utc_grid("2024-01-01T00:00:00+01:00", "2024-01-01T03:00:00+01:00")
    assert len(grid) == 3
    assert str(grid.tz) == "UTC"
    assert grid[0] == pd.Timestamp("2023-12-31 23:00", tz="UTC")
    with pytest.raises(TimeError):
        utc_grid("2024-01-01", "2024-01-02")


# --- local STANDARD time (used by Building Data Genome 2: "DST is ignored") -----------------


def test_standard_time_ignores_dst_in_summer():
    from jardik.core.timeutils import standard_time_to_utc

    naive = pd.DatetimeIndex(["2016-01-15 12:00", "2016-07-15 12:00"])
    # Eastern standard time is UTC-5 all year, even in July when civil time is UTC-4.
    assert standard_time_to_utc(naive, "US/Eastern").tolist() == [
        pd.Timestamp("2016-01-15 17:00", tz="UTC"),
        pd.Timestamp("2016-07-15 17:00", tz="UTC"),
    ]
    assert standard_time_to_utc(naive, "Europe/London").tolist() == [
        pd.Timestamp("2016-01-15 12:00", tz="UTC"),
        pd.Timestamp("2016-07-15 12:00", tz="UTC"),
    ]


def test_standard_time_dublin_is_gmt_not_ist():
    # tzdata models Irish winter time as "negative DST": subtracting dst() would give +1h all
    # year. Standard time must be GMT (UTC+0).
    from jardik.core.timeutils import standard_time_to_utc

    naive = pd.DatetimeIndex(["2016-01-15 12:00", "2016-07-15 12:00"])
    assert standard_time_to_utc(naive, "Europe/Dublin").tolist() == [
        pd.Timestamp("2016-01-15 12:00", tz="UTC"),
        pd.Timestamp("2016-07-15 12:00", tz="UTC"),
    ]


def test_standard_time_grid_stays_continuous_across_dst_dates():
    from jardik.core.timeutils import standard_time_to_utc

    naive = pd.date_range("2016-01-01", "2016-12-31 23:00", freq="h")
    for tz in ("US/Eastern", "US/Pacific", "Europe/London", "Europe/Dublin"):
        utc = standard_time_to_utc(naive, tz)
        assert str(utc.tz) == "UTC"
        assert utc.is_unique and utc.is_monotonic_increasing
        assert (utc[1:] - utc[:-1] == pd.Timedelta(hours=1)).all()


def test_standard_time_rejects_aware_input_and_unknown_zone():
    from jardik.core.timeutils import standard_time_to_utc

    with pytest.raises(TimeError):
        standard_time_to_utc(pd.DatetimeIndex(["2016-01-01"], tz="UTC"), "US/Eastern")
    with pytest.raises(TimeError):
        standard_time_to_utc(pd.DatetimeIndex(["2016-01-01"]), "Mars/Olympus")


def test_standard_utc_offset():
    from jardik.core.timeutils import standard_utc_offset

    assert standard_utc_offset("US/Pacific", 2016) == pd.Timedelta(hours=-8)
    assert standard_utc_offset("Europe/Dublin", 2016) == pd.Timedelta(0)
    assert standard_utc_offset("Europe/Madrid", 2024) == pd.Timedelta(hours=1)
