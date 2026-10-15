import numpy as np
import pandas as pd
import pytest

from jardik.core.interfaces import Forecaster
from jardik.forecasting.baselines import ColumnBaseline, default_baselines


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "lag_24": [10.0, np.nan, 30.0],
            "lag_168": [12.0, 20.0, np.nan],
            "lag_336": [14.0, 22.0, np.nan],
            "lag_504": [16.0, np.nan, np.nan],
            "lag_672": [18.0, 24.0, np.nan],
        }
    )


def test_default_baselines_are_the_three_of_the_plan():
    baselines = default_baselines()
    assert list(baselines) == ["naive_lag_24", "seasonal_naive_168", "hourly_weekday_mean_4w"]
    for model in baselines.values():
        assert isinstance(model, Forecaster)
        assert model.supports_quantiles is False


def test_naive_and_seasonal_naive_copy_the_lag():
    baselines = default_baselines()
    assert baselines["naive_lag_24"].predict(_frame()).tolist()[0::2] == [10.0, 30.0]
    assert baselines["seasonal_naive_168"].predict(_frame()).iloc[1] == 20.0


def test_hourly_weekday_mean_averages_the_available_same_slot_values():
    pred = default_baselines()["hourly_weekday_mean_4w"].predict(_frame())
    assert pred.iloc[0] == pytest.approx(np.mean([12, 14, 16, 18]))
    assert pred.iloc[1] == pytest.approx(np.mean([20, 22, 24]))  # one lag missing is skipped
    assert np.isnan(pred.iloc[2])  # nothing available -> no forecast, not a guess


def test_prediction_keeps_the_index_and_ignores_extra_columns():
    frame = _frame().set_index(pd.Index([7, 8, 9]))
    frame["something_else"] = 1.0
    pred = default_baselines()["naive_lag_24"].predict(frame)
    assert pred.index.tolist() == [7, 8, 9]


def test_fit_is_a_no_op_and_quantiles_are_not_offered():
    model = ColumnBaseline("x", ("lag_24",))
    assert model.fit(_frame()) is None
    with pytest.raises(NotImplementedError, match="point forecast"):
        model.predict_quantiles(_frame(), [0.1, 0.9])


def test_missing_lag_column_is_reported():
    with pytest.raises(KeyError, match="lag_168"):
        default_baselines()["seasonal_naive_168"].predict(pd.DataFrame({"lag_24": [1.0]}))
