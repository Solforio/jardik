import numpy as np
import pandas as pd
import pytest

from jardik.core.schemas import validate_weather_match
from jardik.dataset_manager.matching.policy import MatchPolicy
from jardik.dataset_manager.matching.temporal import MatchResult, match_weather
from jardik.dataset_manager.matching.validate import MatchingError, validate_join


def _ts(start: str, periods: int, freq: str = "h") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=periods, freq=freq, tz="UTC")


def _weather(point: str, index: pd.DatetimeIndex, temp=None) -> pd.DataFrame:
    values = np.arange(len(index), dtype=float) if temp is None else temp
    return pd.DataFrame(
        {
            "weather_point_id": point,
            "ts_utc": index,
            "source": "bdg2_file",
            "product_type": "observation",
            "issued_at_utc": pd.Series(
                pd.NaT, index=range(len(index)), dtype="datetime64[ns, UTC]"
            ),
            "temp_c": values,
        }
    )


def _assignment(rows):
    return pd.DataFrame(
        rows,
        columns=[
            "building_id",
            "weather_point_id",
            "matched_lat",
            "matched_lon",
            "distance_km",
            "spatial_link",
            "reject_reason",
        ],
    )


A_NEAR = ("b1", "p1", 40.4, -3.7, 1.5, "nearest_point", None)
BUILDINGS = pd.DataFrame(
    {
        "building_id": ["b1", "b2"],
        "timezone": ["Europe/Madrid", "Europe/Madrid"],
        "location_quality": ["coordinates", "unknown"],
    }
)


def _load_keys(building: str, index: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({"building_id": building, "ts_utc": index})


def _run(load_index, weather_index, policy, assignment=None, temp=None):
    return match_weather(
        load_keys=_load_keys("b1", load_index),
        weather=_weather("p1", weather_index, temp),
        assignment=_assignment([A_NEAR] if assignment is None else assignment),
        buildings=BUILDINGS,
        policy=policy,
        variables=["temp_c"],
    )


def test_exact_match_on_identical_grids():
    grid = _ts("2024-01-15", 24)
    result = _run(grid, grid, MatchPolicy(method="exact"))
    assert len(result.match) == 24
    assert (result.match["weather_time_offset_min"] == 0).all()
    assert result.values["temp_c"].tolist() == list(range(24))
    validate_weather_match(result.match)


def test_match_columns_follow_the_brief():
    grid = _ts("2024-01-15", 3)
    m = _run(grid, grid, MatchPolicy(method="exact")).match.iloc[0]
    assert m["weather_source"] == "bdg2_file"
    assert (m["matched_lat"], m["matched_lon"], m["distance_km"]) == (40.4, -3.7, 1.5)
    assert m["timezone_used"] == "Europe/Madrid"
    assert m["matching_method"] == "exact"
    assert m["spatial_link"] == "nearest_point"
    assert m["location_quality"] == "coordinates"


def test_exact_does_not_match_off_grid_records():
    result = _run(_ts("2024-01-15 00:30", 5), _ts("2024-01-15", 24), MatchPolicy(method="exact"))
    assert len(result.match) == 0
    assert result.stats["unmatched_rows"] == 5


def test_nearest_previous_within_threshold():
    policy = MatchPolicy(method="nearest_previous", max_time_offset_min=45)
    result = _run(_ts("2024-01-15 00:30", 3), _ts("2024-01-15", 24), policy)
    assert result.match["weather_time_offset_min"].tolist() == [-30, -30, -30]
    assert result.values["temp_c"].tolist() == [0.0, 1.0, 2.0]  # the previous hour, never the next


def test_nearest_previous_beyond_threshold_is_unmatched():
    policy = MatchPolicy(method="nearest_previous", max_time_offset_min=20)
    result = _run(_ts("2024-01-15 00:30", 3), _ts("2024-01-15", 24), policy)
    assert len(result.match) == 0 and result.stats["unmatched_rows"] == 3


def test_nearest_can_pick_a_following_record():
    policy = MatchPolicy(method="nearest", max_time_offset_min=60)
    result = _run(_ts("2024-01-15 00:40", 2), _ts("2024-01-15", 24), policy)
    # 00:40 is 40 min after 00:00 but 20 min before 01:00 -> the following record wins.
    assert result.match["weather_time_offset_min"].tolist() == [20, 20]
    assert result.values["temp_c"].tolist() == [1.0, 2.0]


def test_interpolate_is_declared_and_linear():
    policy = MatchPolicy(method="interpolate", max_time_offset_min=60)
    result = _run(
        _ts("2024-01-15 00:30", 2),
        _ts("2024-01-15", 24),
        policy,
        temp=np.arange(24, dtype=float) * 2,
    )
    assert result.values["temp_c"].tolist() == [1.0, 3.0]
    assert (result.match["matching_method"] == "interpolate").all()
    assert (result.match["weather_time_offset_min"] == 0).all()


def test_interpolate_needs_both_neighbours():
    policy = MatchPolicy(method="interpolate", max_time_offset_min=60)
    # Only one weather sample before the record: no extrapolation.
    result = _run(_ts("2024-01-15 05:30", 1), _ts("2024-01-15", 6), policy)
    assert len(result.match) == 0


def test_interpolate_does_not_bridge_missing_values():
    policy = MatchPolicy(method="interpolate", max_time_offset_min=60)
    temp = np.array([1.0, np.nan, 3.0, 4.0])
    result = _run(_ts("2024-01-15 00:30", 1), _ts("2024-01-15", 4), policy, temp=temp)
    assert len(result.match) == 1 and np.isnan(result.values["temp_c"].iloc[0])


@pytest.mark.parametrize(
    ("day", "hours"), [("2024-03-31", 23), ("2024-10-27", 25), ("2024-06-01", 24)]
)
def test_dst_local_days_keep_every_record(day, hours):
    start = pd.Timestamp(day, tz="Europe/Madrid")
    end = start + pd.DateOffset(days=1)
    load = pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"), freq="h", inclusive="left")
    weather = pd.date_range(
        start.tz_convert("UTC") - pd.Timedelta(days=1),
        end.tz_convert("UTC") + pd.Timedelta(days=1),
        freq="h",
    )
    policy = MatchPolicy(method="exact")
    result = _run(load, weather, policy)
    assert len(result.match) == hours == len(load)
    assert (result.match["weather_time_offset_min"] == 0).all()
    assert result.match["ts_utc"].is_unique
    report = validate_join(result, _load_keys("b1", load), policy, BUILDINGS)
    dst = report["dst_days"]
    if hours != 24:
        assert len(dst) == 1
        assert dst.iloc[0]["expected_hours"] == hours
        assert dst.iloc[0]["load_rows"] == dst.iloc[0]["matched_rows"] == hours
    else:
        assert dst.empty


def test_rejected_buildings_get_no_rows_and_are_counted():
    rejected = ("b1", None, np.nan, np.nan, np.nan, None, "no_location")
    result = _run(_ts("2024-01-15", 5), _ts("2024-01-15", 24), MatchPolicy(), assignment=[rejected])
    assert len(result.match) == 0
    assert result.stats["rows_without_weather_point"] == 5


def test_two_buildings_with_different_points():
    load = pd.concat(
        [_load_keys("b1", _ts("2024-01-15", 3)), _load_keys("b2", _ts("2024-01-15", 3))]
    )
    weather = pd.concat(
        [
            _weather("p1", _ts("2024-01-15", 3), np.array([1.0, 2.0, 3.0])),
            _weather("p2", _ts("2024-01-15", 3), np.array([10.0, 20.0, 30.0])),
        ]
    )
    assignment = _assignment([A_NEAR, ("b2", "p2", np.nan, np.nan, np.nan, "dataset_link", None)])
    result = match_weather(
        load_keys=load,
        weather=weather,
        assignment=assignment,
        buildings=BUILDINGS,
        policy=MatchPolicy(),
        variables=["temp_c"],
    )
    by_building = result.values.groupby("building_id")["temp_c"].apply(list).to_dict()
    assert by_building == {"b1": [1.0, 2.0, 3.0], "b2": [10.0, 20.0, 30.0]}
    b2 = result.match[result.match["building_id"] == "b2"].iloc[0]
    assert b2["spatial_link"] == "dataset_link" and b2["location_quality"] == "unknown"
    validate_weather_match(result.match)


def test_value_missing_ratio_is_reported_separately_from_coverage():
    grid = _ts("2024-01-15", 4)
    result = _run(grid, grid, MatchPolicy(), temp=np.array([1.0, np.nan, 3.0, np.nan]))
    assert result.stats["unmatched_rows"] == 0
    assert result.stats["value_missing_ratio"]["temp_c"] == pytest.approx(0.5)


def test_validate_join_reports_coverage_per_building():
    load = _ts("2024-01-15", 10)
    result = _run(load, _ts("2024-01-15", 5), MatchPolicy())
    report = validate_join(result, _load_keys("b1", load), MatchPolicy(), BUILDINGS)
    assert report["coverage_by_building"]["b1"] == pytest.approx(0.5)


def test_validate_join_rejects_duplicate_matches():
    grid = _ts("2024-01-15", 3)
    result = _run(grid, grid, MatchPolicy())
    broken = MatchResult(
        match=pd.concat([result.match, result.match.iloc[[0]]], ignore_index=True),
        values=result.values,
        stats=result.stats,
    )
    with pytest.raises(MatchingError, match="duplicate"):
        validate_join(broken, _load_keys("b1", grid), MatchPolicy(), BUILDINGS)


def test_validate_join_rejects_offsets_beyond_policy():
    grid = _ts("2024-01-15", 3)
    result = _run(grid, grid, MatchPolicy())
    bad = result.match.assign(weather_time_offset_min=[0, 0, 500])
    broken = MatchResult(match=bad, values=result.values, stats=result.stats)
    with pytest.raises(MatchingError, match="offset"):
        validate_join(
            broken, _load_keys("b1", grid), MatchPolicy(max_time_offset_min=60), BUILDINGS
        )


def test_policy_validation():
    with pytest.raises(ValueError):
        MatchPolicy(method="telepathy")
    with pytest.raises(ValueError):
        MatchPolicy(max_distance_km=0)
    assert MatchPolicy().method == "exact"
