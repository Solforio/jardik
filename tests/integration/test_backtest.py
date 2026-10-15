"""Backtest of the model ladder on synthetic offices (a signal the baselines miss)."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from jardik.forecasting.backtest import BacktestResult, run_backtest
from jardik.forecasting.baselines import default_baselines
from jardik.forecasting.features import FeatureConfig, build_features, model_feature_columns
from jardik.forecasting.lightgbm_model import LGBMParams, LightGBMForecaster
from jardik.forecasting.splits import SplitSpec
from synthetic import make_merged_frame

CONFIG = FeatureConfig()
FEATURES = model_feature_columns(CONFIG)
SPEC = SplitSpec(train_end=date(2024, 3, 15), val_end=date(2024, 4, 10), seed=7)
FAST = LGBMParams(n_estimators=120, num_leaves=15, min_child_samples=20, early_stopping_rounds=15)
BASELINES = list(default_baselines())


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    merged = make_merged_frame(n_buildings=16, start="2024-01-01", days=150, seed=11)
    return build_features(merged, CONFIG)


def _models(extra=None):
    models = default_baselines()
    models["lightgbm"] = LightGBMForecaster(FEATURES, FAST, seed=1)
    models.update(extra or {})
    return models


@pytest.fixture(scope="module")
def result(features) -> BacktestResult:
    return run_backtest(features, SPEC, _models(), challenger="lightgbm", baselines=BASELINES)


# --- structure ---


def test_every_scenario_and_model_is_scored(result):
    metrics = result.metrics
    assert set(metrics["scenario"]) == {"val", "test_time", "test_building", "test_both"}
    assert set(metrics["model"]) == {*BASELINES, "lightgbm"}
    assert len(metrics) == 4 * 4
    assert (metrics["n"] > 0).all()


def test_all_models_are_compared_on_exactly_the_same_rows(result):
    counts = result.metrics.groupby("scenario")["n"].nunique()
    assert (counts == 1).all()


def test_rows_without_a_baseline_forecast_are_excluded_and_reported(result):
    rows = result.metrics.set_index(["scenario", "model"])
    early = rows.loc[("test_building", "lightgbm"), "row_coverage"]
    assert 0 < early < 1  # the first week has no lag_168, so baselines can not forecast it
    assert rows.loc[("test_both", "lightgbm"), "row_coverage"] > 0.99


def test_test_buildings_are_never_in_the_training_data(result, features):
    roles = result.roles
    test_buildings = set(roles[roles == "test"].index)
    train_buildings = set(roles[roles == "train"].index)
    assert test_buildings and train_buildings and not (test_buildings & train_buildings)
    pred = result.predictions
    both = pred[pred["scenario"] == "test_both"]
    assert set(both["building_id"]) <= test_buildings
    assert both["ts_utc"].min() >= pd.Timestamp(SPEC.val_end, tz="UTC")
    n_train = result.models["lightgbm"].n_train_rows
    in_train = features[
        features["building_id"].isin(train_buildings)
        & (features["ts_utc"] < pd.Timestamp(SPEC.train_end, tz="UTC"))
        & features["has_history"]
        & features["is_measured"]
    ]
    assert n_train == len(in_train)


def test_predictions_frame_has_point_and_quantile_columns(result):
    pred = result.predictions
    for col in (
        "scenario",
        "building_id",
        "ts_utc",
        "target_kwh",
        "pred__lightgbm",
        "pred__seasonal_naive_168",
        "q0.1__lightgbm",
        "q0.5__lightgbm",
        "q0.9__lightgbm",
    ):
        assert col in pred.columns, col
    assert not any(c.startswith("q0.1__") and "naive" in c for c in pred.columns)


# --- the gate ---


def test_the_learned_model_beats_every_baseline_on_unseen_buildings_and_period(result):
    rows = result.metrics.set_index(["scenario", "model"])
    gbm_mae = rows.loc[("test_both", "lightgbm"), "mae"]
    for name in BASELINES:
        assert gbm_mae < rows.loc[("test_both", name), "mae"], name


def test_the_gate_reports_the_comparison(result):
    gate = result.gate
    assert gate["scenario"] == "test_both" and gate["challenger"] == "lightgbm"
    assert gate["passed"] is True
    assert gate["best_baseline"] in BASELINES
    assert set(gate["checks"]) == {"mae", "rmse", "median_cv_rmse"}
    for check in gate["checks"].values():
        assert check["improvement"] > 0 and check["challenger"] < check["baseline"]
    assert "beats" in gate["message"]


def test_the_best_baseline_is_the_one_with_the_lowest_mae(result):
    rows = result.metrics[result.metrics["scenario"] == "test_both"].set_index("model")
    best = rows.loc[BASELINES, "mae"].idxmin()
    assert result.gate["best_baseline"] == best


def test_skill_scores_match_the_metrics(result):
    skill = result.skill[
        (result.skill["scenario"] == "test_both") & (result.skill["model"] == "lightgbm")
    ]
    rows = result.metrics[result.metrics["scenario"] == "test_both"].set_index("model")
    best = result.gate["best_baseline"]
    expected = 1 - rows.loc["lightgbm", "mae"] / rows.loc[best, "mae"]
    assert skill["mae_skill"].iloc[0] == pytest.approx(expected)
    assert skill["vs"].iloc[0] == best


def test_a_useless_challenger_fails_the_gate(features):
    class Constant:
        name = "constant"
        supports_quantiles = False

        def fit(self, train, valid=None):
            return None

        def predict(self, frame):
            return pd.Series(1.0, index=frame.index)

        def predict_quantiles(self, frame, quantiles):
            raise NotImplementedError

    models = default_baselines()
    models["constant"] = Constant()
    out = run_backtest(features, SPEC, models, challenger="constant", baselines=BASELINES)
    assert out.gate["passed"] is False
    assert "does not beat" in out.gate["message"]


# --- probabilistic forecast and per-building view ---


def test_probabilistic_metrics_exist_only_for_models_with_quantiles(result):
    rows = result.metrics.set_index(["scenario", "model"])
    both = rows.loc[("test_both", "lightgbm")]
    assert both["pinball"] > 0 and 0.6 < both["coverage_80"] < 0.95 and both["interval_width"] > 0
    assert np.isnan(rows.loc[("test_both", "seasonal_naive_168"), "pinball"])


def test_per_building_and_macro_tables(result):
    per = result.per_building
    both = per[(per["scenario"] == "test_both") & (per["model"] == "lightgbm")]
    n_test_buildings = int((result.roles == "test").sum())
    assert len(both) == n_test_buildings
    macro = result.macro.set_index(["scenario", "model"]).loc[("test_both", "lightgbm")]
    assert macro["buildings"] == n_test_buildings
    assert macro["cv_rmse_mean"] == pytest.approx(both["cv_rmse"].mean())
    assert macro["mae_median"] == pytest.approx(both["mae"].median())


# --- robustness ---


def test_the_backtest_is_deterministic(features, result):
    again = run_backtest(features, SPEC, _models(), challenger="lightgbm", baselines=BASELINES)
    pd.testing.assert_frame_equal(result.metrics, again.metrics)


def test_splitting_by_site_also_works(features):
    spec = SplitSpec(train_end=date(2024, 3, 15), val_end=date(2024, 4, 10), by="site", seed=2)
    out = run_backtest(features, spec, _models(), challenger="lightgbm", baselines=BASELINES)
    assert out.gate["scenario"] == "test_both"
    sites = features[["building_id", "site_id"]].drop_duplicates()
    by_role = (
        sites.assign(role=sites["building_id"].map(out.roles)).groupby("site_id")["role"].nunique()
    )
    assert (by_role == 1).all()


def test_unknown_challenger_or_empty_scenario_is_an_error(features):
    with pytest.raises(ValueError, match="challenger"):
        run_backtest(features, SPEC, default_baselines(), challenger="nope", baselines=BASELINES)
    late = SplitSpec(
        train_end=date(2024, 3, 15), val_end=date(2024, 4, 10), test_end=date(2024, 4, 11)
    )
    short = features[features["ts_utc"] < pd.Timestamp("2024-04-10", tz="UTC")]
    with pytest.raises(ValueError, match="no rows"):
        run_backtest(short, late, _models(), challenger="lightgbm", baselines=BASELINES)
