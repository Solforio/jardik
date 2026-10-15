"""Spatial step: which weather point belongs to which building."""

from __future__ import annotations

import numpy as np
import pandas as pd

from jardik.core.geo import haversine_km, nearest_point

_COLUMNS = [
    "building_id",
    "weather_point_id",
    "matched_lat",
    "matched_lon",
    "distance_km",
    "spatial_link",
    "reject_reason",
]


def assign_weather_points(
    buildings: pd.DataFrame, points: pd.DataFrame, max_distance_km: float
) -> pd.DataFrame:
    """Assign one weather point to each building (one output row per input row, same order).

    Order of preference:
      1. `weather_link`: the dataset itself says which weather series belongs to the building
         (spatial_link = dataset_link). The distance is informational and may be unknown.
      2. Coordinates: the nearest point with known coordinates (spatial_link = nearest_point),
         refused beyond `max_distance_km`.
    Buildings that cannot be assigned get `weather_point_id = None` and a `reject_reason`:
    too_far, no_location or link_not_found.

    buildings: building_id, lat, lon, weather_link (None when the dataset gives no link).
    points: weather_point_id, lat, lon (coordinates may be NaN).
    """
    by_id = points.set_index("weather_point_id")[["lat", "lon"]]
    located = points.dropna(subset=["lat", "lon"])
    rows: list[dict[str, object]] = []

    for b in buildings.itertuples(index=False):
        row: dict[str, object] = dict.fromkeys(_COLUMNS)
        row["building_id"] = b.building_id
        row.update(matched_lat=np.nan, matched_lon=np.nan, distance_km=np.nan)
        has_xy = pd.notna(b.lat) and pd.notna(b.lon)
        link = b.weather_link if pd.notna(b.weather_link) else None

        if link is not None and link in by_id.index:
            p_lat, p_lon = by_id.loc[link, "lat"], by_id.loc[link, "lon"]
            row.update(weather_point_id=link, spatial_link="dataset_link")
            row.update(matched_lat=p_lat, matched_lon=p_lon)
            if has_xy and pd.notna(p_lat) and pd.notna(p_lon):
                row["distance_km"] = float(haversine_km(b.lat, b.lon, p_lat, p_lon))
        elif has_xy and not located.empty:
            point, distance = nearest_point(b.lat, b.lon, located)
            row["distance_km"] = distance
            if distance > max_distance_km:
                row["reject_reason"] = "too_far"
            else:
                row.update(
                    weather_point_id=point["weather_point_id"],
                    matched_lat=float(point["lat"]),
                    matched_lon=float(point["lon"]),
                    spatial_link="nearest_point",
                )
        else:
            row["reject_reason"] = "link_not_found" if link is not None else "no_location"
        rows.append(row)

    out = pd.DataFrame(rows, columns=_COLUMNS)
    for column in ("matched_lat", "matched_lon", "distance_km"):
        out[column] = out[column].astype(float)
    return out
