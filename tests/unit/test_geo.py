import numpy as np
import pandas as pd
import pytest

from jardik.core.geo import haversine_km, nearest_point
from jardik.core.units import convert

MADRID = (40.4168, -3.7038)
BARCELONA = (41.3874, 2.1686)


def test_haversine_known_distance():
    assert haversine_km(*MADRID, *BARCELONA) == pytest.approx(505, abs=5)


def test_haversine_zero_and_symmetry():
    assert haversine_km(*MADRID, *MADRID) == 0.0
    assert haversine_km(*MADRID, *BARCELONA) == pytest.approx(haversine_km(*BARCELONA, *MADRID))


def test_haversine_vectorised():
    lat = np.array([MADRID[0], BARCELONA[0]])
    lon = np.array([MADRID[1], BARCELONA[1]])
    out = haversine_km(*MADRID, lat, lon)
    assert out[0] == 0.0 and out[1] == pytest.approx(505, abs=5)


def test_nearest_point():
    points = pd.DataFrame(
        {
            "weather_point_id": ["bcn", "mad"],
            "lat": [BARCELONA[0], MADRID[0]],
            "lon": [BARCELONA[1], MADRID[1]],
        }
    )
    row, distance = nearest_point(40.5, -3.6, points)
    assert row["weather_point_id"] == "mad"
    assert distance < 20


def test_nearest_point_empty_raises():
    with pytest.raises(ValueError):
        nearest_point(0.0, 0.0, pd.DataFrame({"lat": [], "lon": []}))


def test_unit_symbols_used_by_open_meteo_are_accepted():
    assert convert(100.0, "W/m²", "W/m2") == 100.0
    assert convert(90.0, "°", "deg") == 90.0
