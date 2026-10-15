"""Synthetic merged_training_dataset frames for the forecasting tests."""

import numpy as np
import pandas as pd

from jardik.core.timeutils import add_local_columns


def make_merged_frame(
    n_buildings: int = 4,
    start: str = "2024-03-01",
    days: int = 60,
    tz: str = "Europe/Madrid",
    seed: int = 0,
    noise: float = 0.05,
    n_sites: int = 3,
) -> pd.DataFrame:
    """Office-like load driven by hour, weekday and temperature, on a contiguous hourly UTC grid
    with civil local time (DST aware). Buildings differ in size; sites share weather."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range(start, periods=days * 24, freq="h", tz="UTC")
    n = len(hours)
    site_temp = []
    for _ in range(n_sites):
        daily = np.repeat(np.cumsum(rng.normal(0, 1.5, days)) * 0.5, 24)
        site_temp.append(12 + daily + 6 * np.sin(2 * np.pi * (hours.hour.to_numpy() - 15) / 24))
    frames = []
    for b in range(n_buildings):
        site = b % n_sites
        frame = pd.DataFrame({"ts_utc": hours})
        frame = add_local_columns(frame, tz)
        hour = frame["ts_local"].dt.hour.to_numpy()
        weekday = frame["ts_local"].dt.dayofweek.to_numpy()
        working = (weekday < 5) & (hour >= 7) & (hour < 19)
        profile = np.where(working, 1.6, np.where(weekday < 5, 0.5, 0.6))
        scale = 20.0 + 15.0 * b
        temp = site_temp[site]
        load = scale * profile * (1 + 0.03 * (temp - 15)) + rng.normal(0, noise * scale, n)
        frame["building_id"] = f"bdg2:b{b:02d}"
        frame["target_kwh"] = np.clip(load, 0.1, None)
        frame["is_missing"] = False
        frame["is_imputed"] = False
        frame["quality_flag"] = "ok"
        frame["temp_c"] = temp
        frame["dewpoint_c"] = temp - 5
        frame["wind_ms"] = 3.0
        frame["pressure_msl_hpa"] = 1013.0
        frame["usage_category"] = "office"
        frame["site_id"] = f"S{site}"
        frame["floor_area_m2"] = 1000.0 + 500 * b
        frame["year_built"] = 1990
        frame["timezone"] = tz
        frame["lat"] = 40.0
        frame["lon"] = -3.0
        frame["location_quality"] = "coordinates"
        frame["hour_local"] = hour
        frame["dayofweek_local"] = weekday
        frame["month_local"] = frame["ts_local"].dt.month.to_numpy()
        frame["is_weekend_local"] = weekday >= 5
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)
