"""Geographic helpers: great-circle distance and nearest-point lookup."""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import pandas as pd

EARTH_RADIUS_KM = 6371.0088

T = TypeVar("T", float, np.ndarray, pd.Series)


def haversine_km(lat1: float, lon1: float, lat2: T, lon2: T) -> T:
    """Great-circle distance in km between one point and a point or arrays of points."""
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = np.radians(lon2) - np.radians(lon1)
    a = np.sin(d_phi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2.0) ** 2
    central_angle = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return EARTH_RADIUS_KM * central_angle  # type: ignore[return-value]


def nearest_point(lat: float, lon: float, points: pd.DataFrame) -> tuple[pd.Series, float]:
    """Row of `points` (columns lat, lon) closest to (lat, lon), and its distance in km."""
    if points.empty:
        raise ValueError("No points to search")
    distances = haversine_km(lat, lon, points["lat"].to_numpy(float), points["lon"].to_numpy(float))
    position = int(np.argmin(distances))
    return points.iloc[position], float(distances[position])
