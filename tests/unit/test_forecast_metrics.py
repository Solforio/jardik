import numpy as np
import pandas as pd
import pytest

from jardik.forecasting.metrics import (
    cv_rmse,
    evaluate_point,
    interval_coverage,
    interval_width,
    mae,
    mean_pinball_loss,
    nmbe,
    pinball_loss,
    rmse,
    skill_score,
)

ACTUAL = np.array([10.0, 20.0, 30.0])
PRED = np.array([12.0, 18.0, 33.0])  # errors (pred - actual): +2, -2, +3


def test_mae_and_rmse():
    assert mae(ACTUAL, PRED) == pytest.approx(7 / 3)
    assert rmse(ACTUAL, PRED) == pytest.approx(np.sqrt(17 / 3))


def test_cv_rmse_is_rmse_over_mean_actual():
    assert cv_rmse(ACTUAL, PRED) == pytest.approx(np.sqrt(17 / 3) / 20.0)


def test_nmbe_follows_ashrae_guideline_14_sign():
    # (actual - pred) summed over n * mean(actual): negative means the model over-forecasts.
    assert nmbe(ACTUAL, PRED) == pytest.approx((-2 + 2 - 3) / (3 * 20.0))
    assert nmbe(ACTUAL, ACTUAL - 5) > 0  # under-forecast gives a positive NMBE


def test_perfect_forecast_has_zero_error():
    for metric in (mae, rmse, cv_rmse, nmbe):
        assert metric(ACTUAL, ACTUAL) == 0.0


def test_nan_pairs_are_ignored_and_empty_gives_nan():
    actual = np.array([10.0, np.nan, 30.0, 40.0])
    pred = np.array([12.0, 5.0, np.nan, 44.0])
    assert mae(actual, pred) == pytest.approx(3.0)  # only pairs 0 and 3
    assert np.isnan(mae(np.array([np.nan]), np.array([1.0])))
    assert np.isnan(cv_rmse(np.array([]), np.array([])))


def test_zero_mean_actual_gives_nan_for_relative_metrics():
    zeros = np.zeros(3)
    assert np.isnan(cv_rmse(zeros, np.ones(3))) and np.isnan(nmbe(zeros, np.ones(3)))
    assert mae(zeros, np.ones(3)) == 1.0


def test_length_mismatch_is_an_error():
    with pytest.raises(ValueError, match="length"):
        mae(np.array([1.0, 2.0]), np.array([1.0]))


def test_series_input_is_accepted():
    assert mae(pd.Series(ACTUAL), pd.Series(PRED)) == pytest.approx(7 / 3)


def test_evaluate_point_returns_all_metrics_and_count():
    out = evaluate_point(ACTUAL, PRED)
    assert set(out) == {"n", "mae", "rmse", "cv_rmse", "nmbe"} and out["n"] == 3


def test_pinball_loss_penalises_the_two_sides_asymmetrically():
    y = np.array([10.0])
    assert pinball_loss(y, np.array([8.0]), 0.9) == pytest.approx(0.9 * 2)  # under-forecast
    assert pinball_loss(y, np.array([12.0]), 0.9) == pytest.approx(0.1 * 2)  # over-forecast
    assert pinball_loss(y, np.array([12.0]), 0.5) == pytest.approx(1.0)
    assert pinball_loss(y, y, 0.3) == 0.0


def test_pinball_loss_rejects_invalid_quantiles():
    with pytest.raises(ValueError):
        pinball_loss(ACTUAL, PRED, 1.0)


def test_mean_pinball_averages_over_quantiles():
    y = np.array([10.0, 10.0])
    q = pd.DataFrame({0.1: [9.0, 9.0], 0.9: [11.0, 11.0]})
    expected = (
        pinball_loss(y, q[0.1].to_numpy(), 0.1) + pinball_loss(y, q[0.9].to_numpy(), 0.9)
    ) / 2
    assert mean_pinball_loss(y, q) == pytest.approx(expected)


def test_interval_coverage_and_width():
    y = np.array([1.0, 5.0, 9.0, 20.0])
    lo, hi = np.array([0.0, 4.0, 10.0, 15.0]), np.array([2.0, 6.0, 12.0, 25.0])
    assert interval_coverage(y, lo, hi) == pytest.approx(0.75)  # 9.0 is below its interval
    assert interval_width(lo, hi) == pytest.approx(np.mean([2, 2, 2, 10]))


def test_skill_score():
    assert skill_score(8.0, 10.0) == pytest.approx(0.2)
    assert skill_score(12.0, 10.0) == pytest.approx(-0.2)
    assert np.isnan(skill_score(1.0, 0.0))
