import numpy as np
import pandas as pd
import pytest

from jardik.core.interfaces import Forecaster
from jardik.forecasting.features import FeatureConfig, build_features, model_feature_columns
from jardik.forecasting.lightgbm_model import LGBMParams, LightGBMForecaster
from jardik.forecasting.metrics import interval_coverage, mae

CONFIG = FeatureConfig()
FEATURES = model_feature_columns(CONFIG)
FAST = LGBMParams(n_estimators=150, num_leaves=15, min_child_samples=20, early_stopping_rounds=15)


@pytest.fixture
def split(make_merged):
    frame = build_features(make_merged(n_buildings=8, start="2024-01-01", days=150, seed=3), CONFIG)
    usable = frame[frame["has_history"] & frame["is_measured"]]
    cut1, cut2 = pd.Timestamp("2024-04-01", tz="UTC"), pd.Timestamp("2024-04-20", tz="UTC")
    return {
        "train": usable[usable["ts_utc"] < cut1],
        "valid": usable[(usable["ts_utc"] >= cut1) & (usable["ts_utc"] < cut2)],
        "test": usable[usable["ts_utc"] >= cut2],
    }


def _model(**kw) -> LightGBMForecaster:
    return LightGBMForecaster(FEATURES, FAST, **kw)


def test_implements_the_forecaster_protocol():
    model = _model()
    assert isinstance(model, Forecaster) and model.supports_quantiles


def test_predictions_are_in_kwh_and_beat_a_constant_level_forecast(split):
    model = _model()
    model.fit(split["train"], split["valid"])
    test = split["test"]
    pred = model.predict(test)
    assert pred.index.equals(test.index) and pred.notna().all()
    assert mae(test["target_kwh"], pred) < 0.5 * mae(test["target_kwh"], test["scale_kwh"])


def test_the_scale_converts_the_ratio_back_to_kwh(split):
    model = LightGBMForecaster(
        FEATURES, LGBMParams(**{**FAST.model_dump(), "target_mode": "ratio"})
    )
    model.fit(split["train"], split["valid"])
    sample = split["test"].head(200)
    doubled = sample.assign(scale_kwh=sample["scale_kwh"] * 2)
    assert np.allclose(model.predict(doubled), 2 * model.predict(sample))


def test_quantiles_are_ordered_and_roughly_calibrated(split):
    model = _model()
    model.fit(split["train"], split["valid"])
    test = split["test"]
    q = model.predict_quantiles(test, [0.1, 0.5, 0.9])
    assert list(q.columns) == [0.1, 0.5, 0.9]
    assert (q[0.1] <= q[0.5]).all() and (q[0.5] <= q[0.9]).all()  # no crossing
    coverage = interval_coverage(test["target_kwh"], q[0.1], q[0.9])
    assert 0.6 < coverage < 0.95  # nominal 80 %


def test_a_quantile_that_was_not_trained_is_refused(split):
    model = _model(quantiles=(0.1, 0.9))
    model.fit(split["train"], split["valid"])
    with pytest.raises(ValueError, match="0.5"):
        model.predict_quantiles(split["test"].head(5), [0.5])


def test_early_stopping_stops_before_the_tree_limit(split):
    model = LightGBMForecaster(
        FEATURES,
        LGBMParams(
            n_estimators=1000,
            num_leaves=15,
            min_child_samples=20,
            early_stopping_rounds=10,
            learning_rate=0.2,
        ),
    )
    model.fit(split["train"], split["valid"])
    assert 0 < model.best_iterations["mean"] < 1000


def test_without_a_validation_set_a_fixed_number_of_trees_is_used(split):
    model = LightGBMForecaster(
        FEATURES,
        LGBMParams(
            n_estimators=900,
            n_estimators_without_validation=40,
            num_leaves=15,
            min_child_samples=20,
        ),
    )
    model.fit(split["train"])
    assert model.best_iterations["mean"] == 40


def test_fitting_twice_gives_identical_predictions(split):
    a, b = _model(seed=5), _model(seed=5)
    a.fit(split["train"], split["valid"])
    b.fit(split["train"], split["valid"])
    assert np.array_equal(a.predict(split["test"]), b.predict(split["test"]))


def test_save_and_load_roundtrip(split, tmp_path):
    model = _model()
    model.fit(split["train"], split["valid"])
    model.save(tmp_path / "m")
    loaded = LightGBMForecaster.load(tmp_path / "m")
    test = split["test"].head(300)
    assert np.allclose(model.predict(test), loaded.predict(test))
    assert np.allclose(
        model.predict_quantiles(test, [0.1, 0.9]), loaded.predict_quantiles(test, [0.1, 0.9])
    )
    assert loaded.features == FEATURES


def test_missing_feature_columns_are_reported(split):
    model = _model()
    model.fit(split["train"], split["valid"])
    with pytest.raises(KeyError, match="n_lag_168"):
        model.predict(split["test"].drop(columns=["n_lag_168"]))
    with pytest.raises(KeyError, match="scale_kwh"):
        model.predict(split["test"].drop(columns=["scale_kwh"]))


def test_missing_feature_values_are_handled_natively(split):
    model = _model()
    model.fit(split["train"], split["valid"])
    sample = split["test"].head(100).copy()
    sample["n_lag_168"] = np.nan
    assert model.predict(sample).notna().all()


def test_rows_with_missing_target_or_scale_are_ignored_in_fit(split):
    train = split["train"].copy()
    train.iloc[:100, train.columns.get_loc("target_kwh")] = np.nan
    train.iloc[100:200, train.columns.get_loc("scale_kwh")] = np.nan
    model = _model()
    model.fit(train, split["valid"])
    assert model.n_train_rows == len(train) - 200


def test_fit_without_usable_rows_is_an_error(split):
    empty = split["train"].iloc[0:0]
    with pytest.raises(ValueError, match="no usable"):
        _model().fit(empty)


def test_feature_importance_lists_every_feature_best_first(split):
    model = _model()
    model.fit(split["train"], split["valid"])
    importance = model.feature_importance()
    assert list(importance.columns) == ["feature", "gain"]
    assert set(importance["feature"]) == set(FEATURES)
    assert importance["gain"].is_monotonic_decreasing
    top = importance["feature"].head(8).tolist()
    assert "hour_local" in top or "n_lag_168" in top or "n_lag_24" in top


def test_predict_before_fit_is_an_error(split):
    with pytest.raises(RuntimeError, match="not fitted"):
        _model().predict(split["test"].head(3))


# --- residual target and loss choice ---


def _fast(**kw) -> LGBMParams:
    return LGBMParams(
        n_estimators=150, num_leaves=15, min_child_samples=20, early_stopping_rounds=15, **kw
    )


def test_defaults_are_the_ones_chosen_on_validation_and_are_validated():
    assert LGBMParams().target_mode == "ratio" and LGBMParams().objective == "huber"
    with pytest.raises(ValueError):
        LGBMParams(target_mode="magic")
    with pytest.raises(ValueError):
        LGBMParams(objective="rmse")


@pytest.mark.parametrize("mode", ["ratio", "residual"])
@pytest.mark.parametrize("objective", ["regression", "regression_l1", "huber"])
def test_every_mode_and_loss_trains_and_beats_a_constant(split, mode, objective):
    model = LightGBMForecaster(FEATURES, _fast(target_mode=mode, objective=objective))
    model.fit(split["train"], split["valid"])
    test = split["test"]
    assert mae(test["target_kwh"], model.predict(test)) < 0.6 * mae(
        test["target_kwh"], test["scale_kwh"]
    )


def test_a_residual_model_starts_from_the_base_forecast(split):
    # Where the target equals the base exactly there is nothing to correct: the residual model
    # must reproduce the base forecast almost perfectly.
    train = split["train"].copy()
    train["target_kwh"] = train["base_kwh"]
    valid = split["valid"].assign(target_kwh=split["valid"]["base_kwh"])
    model = LightGBMForecaster(FEATURES, _fast(target_mode="residual"))
    model.fit(train, valid)
    test = split["test"].head(500)
    assert np.allclose(model.predict(test), test["base_kwh"], atol=1e-6)


def test_residual_predictions_scale_with_base_and_scale_together(split):
    model = LightGBMForecaster(FEATURES, _fast(target_mode="residual"))
    model.fit(split["train"], split["valid"])
    sample = split["test"].head(200)
    doubled = sample.assign(scale_kwh=sample["scale_kwh"] * 2, base_kwh=sample["base_kwh"] * 2)
    assert np.allclose(model.predict(doubled), 2 * model.predict(sample))


def test_residual_quantiles_stay_ordered_and_include_the_base_shift(split):
    model = LightGBMForecaster(FEATURES, _fast(target_mode="residual"))
    model.fit(split["train"], split["valid"])
    q = model.predict_quantiles(split["test"], [0.1, 0.5, 0.9])
    assert (q[0.1] <= q[0.5]).all() and (q[0.5] <= q[0.9]).all()
    assert 0.6 < interval_coverage(split["test"]["target_kwh"], q[0.1], q[0.9]) < 0.95


def test_target_mode_and_objective_survive_save_and_load(split, tmp_path):
    model = LightGBMForecaster(FEATURES, _fast(target_mode="residual", objective="huber"))
    model.fit(split["train"], split["valid"])
    model.save(tmp_path / "m")
    loaded = LightGBMForecaster.load(tmp_path / "m")
    assert loaded.params.target_mode == "residual" and loaded.params.objective == "huber"
    test = split["test"].head(100)
    assert np.allclose(model.predict(test), loaded.predict(test))


def test_residual_mode_needs_the_base_column(split):
    model = LightGBMForecaster(FEATURES, _fast(target_mode="residual"))
    model.fit(split["train"], split["valid"])
    with pytest.raises(KeyError, match="base_kwh"):
        model.predict(split["test"].drop(columns=["base_kwh"]))
