"""Features must use only information available at the forecast origin (local midnight).

The decisive test perturbs every load value at or after the origin and checks that no feature of
the forecast day changes. Weather at the target hour is allowed: it stands for the weather
forecast, which is known in advance.
"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from jardik.forecasting.features import FeatureConfig, build_features, model_feature_columns

TARGET_ONLY = {"target_kwh", "is_measured"}


def _features(frame, **kw) -> pd.DataFrame:
    return build_features(frame, FeatureConfig(**kw))


def _day(features: pd.DataFrame, building: str, day: date) -> pd.DataFrame:
    return features[(features["building_id"] == building) & (features["ts_local"].dt.date == day)]


# --- the forecast origin ---


def test_hours_ahead_counts_slots_since_local_midnight_also_on_dst_days(make_merged):
    spring = _features(make_merged(n_buildings=1, start="2024-03-20", days=20))
    day = _day(spring, "bdg2:b00", date(2024, 3, 31))
    assert day["hours_ahead"].tolist() == list(range(23))  # 23 hour day

    autumn = _features(make_merged(n_buildings=1, start="2024-10-15", days=20))
    day = _day(autumn, "bdg2:b00", date(2024, 10, 27))
    assert day["hours_ahead"].tolist() == list(range(25))  # 25 hour day


def test_lag_features_never_reach_the_forecast_day(make_merged):
    frame = make_merged(n_buildings=1, start="2024-10-15", days=20)
    f = _features(frame)
    day = _day(f, "bdg2:b00", date(2024, 10, 27))  # 25 slots: the last one is 24 h after origin
    assert day["lag_24"].iloc[:24].notna().all()
    assert np.isnan(day["lag_24"].iloc[24])  # lag_24 would be the first slot of the same day
    assert day["lag_48"].notna().all()


def test_lags_equal_the_load_shifted_on_the_utc_grid(make_merged):
    frame = make_merged(n_buildings=1, days=30)
    f = _features(frame).set_index("ts_utc")
    target = frame.set_index("ts_utc")["target_kwh"]
    t = pd.Timestamp("2024-03-25 10:00", tz="UTC")
    assert f.loc[t, "lag_168"] == pytest.approx(target[t - pd.Timedelta(hours=168)])
    assert f.loc[t, "lag_336"] == pytest.approx(target[t - pd.Timedelta(hours=336)])


# --- no look-ahead ---


def test_features_of_a_day_do_not_depend_on_load_at_or_after_the_origin(make_merged):
    frame = make_merged(n_buildings=2, start="2024-03-01", days=40, seed=1)
    base = _features(frame)
    columns = [c for c in base.columns if c not in TARGET_ONLY]

    day = date(2024, 3, 20)
    origin = _day(base, "bdg2:b00", day)["ts_utc"].min()
    tampered = frame.copy()
    hit = (tampered["building_id"] == "bdg2:b00") & (tampered["ts_utc"] >= origin)
    tampered.loc[hit, "target_kwh"] = np.random.default_rng(9).uniform(5_000, 9_000, int(hit.sum()))
    changed = _features(tampered)

    a = _day(base, "bdg2:b00", day)[columns].reset_index(drop=True)
    b = _day(changed, "bdg2:b00", day)[columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)
    # sanity: the perturbation does change the features of LATER days
    later_a = _day(base, "bdg2:b00", date(2024, 3, 25))[columns].reset_index(drop=True)
    later_b = _day(changed, "bdg2:b00", date(2024, 3, 25))[columns].reset_index(drop=True)
    assert not later_a.equals(later_b)


def test_load_of_another_building_never_leaks_in(make_merged):
    frame = make_merged(n_buildings=2, days=30, seed=2)
    base = _features(frame)
    tampered = frame.copy()
    tampered.loc[tampered["building_id"] == "bdg2:b01", "target_kwh"] = 1.0
    changed = _features(tampered)
    columns = [c for c in base.columns if c not in TARGET_ONLY]
    a = base[base["building_id"] == "bdg2:b00"][columns].reset_index(drop=True)
    b = changed[changed["building_id"] == "bdg2:b00"][columns].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


# --- scale and normalisation ---


def test_scale_is_the_mean_load_of_the_week_before_the_origin(make_merged):
    frame = make_merged(n_buildings=1, days=40)
    f = _features(frame)
    day = _day(f, "bdg2:b00", date(2024, 3, 20))
    origin = day["ts_utc"].min()
    target = frame.set_index("ts_utc")["target_kwh"]
    week = target[(target.index >= origin - pd.Timedelta(hours=168)) & (target.index < origin)]
    assert day["scale_kwh"].iloc[0] == pytest.approx(week.mean())
    assert day["scale_kwh"].nunique() == 1  # one value for the whole forecast day


def test_normalised_features_are_divided_by_the_scale(make_merged):
    f = _features(make_merged(n_buildings=1, days=40))
    row = f[f["has_history"]].iloc[100]
    assert row["n_lag_24"] == pytest.approx(row["lag_24"] / row["scale_kwh"], rel=1e-5)


def test_origin_statistics_describe_the_last_hours_before_midnight(make_merged):
    frame = make_merged(n_buildings=1, days=40)
    f = _features(frame)
    day = _day(f, "bdg2:b00", date(2024, 3, 20))
    origin = day["ts_utc"].min()
    target = frame.set_index("ts_utc")["target_kwh"]
    last_24 = target[(target.index >= origin - pd.Timedelta(hours=24)) & (target.index < origin)]
    scale = day["scale_kwh"].iloc[0]
    assert day["n_last"].iloc[0] == pytest.approx(
        target[origin - pd.Timedelta(hours=1)] / scale, rel=1e-5
    )
    assert day["n_mean_24"].iloc[0] == pytest.approx(last_24.mean() / scale, rel=1e-5)
    assert day["n_max_24"].iloc[0] == pytest.approx(last_24.max() / scale, rel=1e-5)


def test_rows_without_enough_history_are_flagged(make_merged):
    f = _features(make_merged(n_buildings=1, days=30), min_history_obs=72)
    early = f[f["ts_utc"] < f["ts_utc"].min() + pd.Timedelta(hours=60)]
    assert not early["has_history"].any() and early["scale_kwh"].isna().all()
    assert f["has_history"].iloc[-1]
    assert f["has_history"].sum() < len(f)


def test_is_measured_excludes_missing_and_imputed_targets(make_merged):
    frame = make_merged(n_buildings=1, days=30)
    frame.loc[500, ["target_kwh", "is_missing"]] = [np.nan, True]
    frame.loc[501, "is_imputed"] = True
    f = _features(frame)
    assert not f["is_measured"].iloc[500] and not f["is_measured"].iloc[501]
    assert f["is_measured"].iloc[502]


def test_gaps_in_the_load_do_not_break_the_features(make_merged):
    frame = make_merged(n_buildings=1, days=30)
    frame.loc[300:340, ["target_kwh", "is_missing"]] = [np.nan, True]
    f = _features(frame)
    assert len(f) == len(frame)
    assert f["scale_kwh"].notna().sum() > 0


# --- weather and calendar ---


def test_weather_features(make_merged):
    frame = make_merged(n_buildings=1, days=20)
    f = _features(frame)
    day = _day(f, "bdg2:b00", date(2024, 3, 15))
    raw = _day(frame, "bdg2:b00", date(2024, 3, 15))
    assert day["temp_day_mean"].iloc[0] == pytest.approx(raw["temp_c"].mean(), rel=1e-5)
    assert np.allclose(day["hdd"], np.maximum(0, 18 - raw["temp_c"].to_numpy()), rtol=1e-5)
    assert np.allclose(day["cdd"], np.maximum(0, raw["temp_c"].to_numpy() - 21), rtol=1e-5)
    row = f.iloc[200]
    assert row["temp_lag_24"] == pytest.approx(frame["temp_c"].iloc[200 - 24], rel=1e-5)
    assert row["temp_delta_24"] == pytest.approx(row["temp_c"] - row["temp_lag_24"], rel=1e-4)


def test_calendar_features(make_merged):
    f = _features(make_merged(n_buildings=1, days=20))
    assert f["doy_sin"].between(-1, 1).all() and f["doy_cos"].between(-1, 1).all()
    assert set(f["is_weekend_local"].unique()) <= {0, 1}
    assert (f["hour_local"] == f["ts_local"].dt.hour).all()


# --- structure ---


def test_one_row_per_input_row_in_building_time_order(make_merged):
    frame = make_merged(n_buildings=3, days=20).sample(frac=1.0, random_state=0)
    f = _features(frame)
    assert len(f) == len(frame)
    assert not f.duplicated(["building_id", "ts_utc"]).any()
    assert f.equals(f.sort_values(["building_id", "ts_utc"]).reset_index(drop=True))


def test_model_feature_columns_exist_and_exclude_target_and_ids(make_merged):
    config = FeatureConfig()
    f = build_features(make_merged(n_buildings=1, days=20), config)
    columns = model_feature_columns(config)
    assert set(columns) <= set(f.columns)
    for forbidden in (
        "target_kwh",
        "building_id",
        "ts_utc",
        "ts_local",
        "site_id",
        "scale_kwh",
        "lag_24",
    ):
        assert forbidden not in columns  # raw (unnormalised) lags are for baselines only
    assert len(columns) == len(set(columns))


def test_missing_input_column_is_reported(make_merged):
    with pytest.raises(KeyError, match="target_kwh"):
        build_features(
            make_merged(n_buildings=1, days=10).drop(columns=["target_kwh"]), FeatureConfig()
        )


def test_missing_weather_columns_become_nan_features_not_errors(make_merged):
    frame = make_merged(n_buildings=1, days=20).drop(columns=["wind_ms", "pressure_msl_hpa"])
    f = build_features(frame, FeatureConfig())
    assert f["wind_ms"].isna().all() and f["pressure_msl_hpa"].isna().all()


# --- the base forecast (what a residual model corrects) -----------------------------------------


def test_base_is_the_mean_of_the_same_slot_in_the_last_weeks(make_merged):
    frame = make_merged(n_buildings=1, days=60)
    f = _features(frame).set_index("ts_utc")
    target = frame.set_index("ts_utc")["target_kwh"]
    t = pd.Timestamp("2024-04-10 10:00", tz="UTC")
    expected = np.mean([target[t - pd.Timedelta(hours=h)] for h in (168, 336, 504, 672)])
    assert f.loc[t, "base_kwh"] == pytest.approx(expected)
    assert f.loc[t, "n_base"] == pytest.approx(expected / f.loc[t, "scale_kwh"], rel=1e-5)


def test_base_falls_back_to_yesterday_then_to_the_scale(make_merged):
    frame = make_merged(n_buildings=1, days=30)
    f = _features(frame)
    early = f[f["has_history"]].iloc[:30]  # first usable day: only lag_24 / scale are available
    assert early["base_kwh"].notna().all()
    row = early[early["lag_168"].isna() & early["lag_24"].notna()].iloc[0]
    assert row["base_kwh"] == pytest.approx(row["lag_24"])
    no_lags = early[early["lag_168"].isna() & early["lag_24"].isna()]
    if len(no_lags):
        assert (no_lags["base_kwh"] == no_lags["scale_kwh"]).all()


def test_base_never_uses_load_from_the_forecast_day(make_merged):
    frame = make_merged(n_buildings=1, start="2024-03-01", days=40, seed=4)
    base = _features(frame)
    day = date(2024, 3, 25)
    origin = _day(base, "bdg2:b00", day)["ts_utc"].min()
    tampered = frame.copy()
    hit = tampered["ts_utc"] >= origin
    tampered.loc[hit, "target_kwh"] = 9_999.0
    changed = _features(tampered)
    a = _day(base, "bdg2:b00", day)[["base_kwh", "n_base"]].reset_index(drop=True)
    b = _day(changed, "bdg2:b00", day)[["base_kwh", "n_base"]].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_n_base_is_a_model_feature_and_base_kwh_is_not(make_merged):
    columns = model_feature_columns(FeatureConfig())
    assert "n_base" in columns and "base_kwh" not in columns
