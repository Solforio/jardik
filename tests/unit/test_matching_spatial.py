import numpy as np
import pandas as pd
import pytest

from jardik.dataset_manager.matching.spatial import assign_weather_points

POINTS = pd.DataFrame(
    {
        "weather_point_id": ["mad", "bcn", "nowhere"],
        "lat": [40.4168, 41.3874, np.nan],
        "lon": [-3.7038, 2.1686, np.nan],
    }
)


def _buildings(rows):
    return pd.DataFrame(rows, columns=["building_id", "lat", "lon", "weather_link"])


def test_nearest_point_from_coordinates():
    out = assign_weather_points(_buildings([("b1", 40.5, -3.6, None)]), POINTS, max_distance_km=50)
    row = out.iloc[0]
    assert row["weather_point_id"] == "mad"
    assert row["spatial_link"] == "nearest_point"
    assert row["matched_lat"] == pytest.approx(40.4168)
    assert 0 < row["distance_km"] < 20
    assert pd.isna(row["reject_reason"])


def test_point_without_coordinates_is_never_chosen_by_distance():
    out = assign_weather_points(_buildings([("b1", 0.0, 0.0, None)]), POINTS, max_distance_km=1e6)
    assert out.iloc[0]["weather_point_id"] in {"mad", "bcn"}


def test_too_far_is_rejected_with_reason():
    out = assign_weather_points(
        _buildings([("b1", 48.85, 2.35, None)]), POINTS, max_distance_km=100
    )
    row = out.iloc[0]
    assert pd.isna(row["weather_point_id"]) and row["reject_reason"] == "too_far"
    assert row["distance_km"] > 100  # the distance is still reported for debugging


def test_no_location_and_no_link_is_rejected():
    out = assign_weather_points(
        _buildings([("b1", np.nan, np.nan, None)]), POINTS, max_distance_km=50
    )
    assert out.iloc[0]["reject_reason"] == "no_location"


def test_dataset_link_wins_over_coordinates():
    # Coordinates say Madrid, but the dataset states this building's weather is the BCN series.
    out = assign_weather_points(
        _buildings([("b1", 40.4168, -3.7038, "bcn")]), POINTS, max_distance_km=50
    )
    row = out.iloc[0]
    assert row["weather_point_id"] == "bcn" and row["spatial_link"] == "dataset_link"
    assert row["distance_km"] > 400  # informational: link and coordinates disagree


def test_dataset_link_with_unknown_point_coordinates():
    out = assign_weather_points(
        _buildings([("b1", np.nan, np.nan, "nowhere")]), POINTS, max_distance_km=50
    )
    row = out.iloc[0]
    assert row["weather_point_id"] == "nowhere" and row["spatial_link"] == "dataset_link"
    assert pd.isna(row["matched_lat"]) and pd.isna(row["distance_km"])


def test_broken_link_falls_back_to_coordinates_or_is_rejected():
    out = assign_weather_points(
        _buildings([("b1", 40.5, -3.6, "ghost"), ("b2", np.nan, np.nan, "ghost")]),
        POINTS,
        max_distance_km=50,
    )
    assert out.iloc[0]["weather_point_id"] == "mad"
    assert out.iloc[1]["reject_reason"] == "link_not_found"


def test_one_row_per_building_in_input_order():
    b = _buildings([("b2", 41.4, 2.2, None), ("b1", 40.4, -3.7, None)])
    out = assign_weather_points(b, POINTS, max_distance_km=50)
    assert out["building_id"].tolist() == ["b2", "b1"]
    assert out["weather_point_id"].tolist() == ["bcn", "mad"]
