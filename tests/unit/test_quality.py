import numpy as np
import pandas as pd
import pytest

from jardik.dataset_manager.quality.geo import check_geographic_consistency
from jardik.dataset_manager.quality.imputation import impute_short_gaps
from jardik.dataset_manager.quality.stats import summarise_series

NAN = np.nan


def _s(values) -> pd.Series:
    index = pd.date_range("2024-01-01", periods=len(values), freq="h", tz="UTC")
    return pd.Series(values, index=index, dtype=float)


# --- imputation -----------------------------------------------------------------------------


def test_short_interior_gap_is_interpolated_linearly():
    filled, mask = impute_short_gaps(_s([1, NAN, NAN, 4, 5]), max_gap=3)
    assert filled.tolist() == [1, 2, 3, 4, 5]
    assert mask.tolist() == [False, True, True, False, False]


def test_gap_longer_than_limit_is_left_untouched_entirely():
    # pandas' own `limit` would fill the first 3 hours of a 4 hour gap: we must not.
    series = _s([1, NAN, NAN, NAN, NAN, 6])
    filled, mask = impute_short_gaps(series, max_gap=3)
    assert filled.isna().sum() == 4 and not mask.any()


def test_leading_and_trailing_gaps_are_never_filled():
    filled, mask = impute_short_gaps(_s([NAN, 2, 3, NAN]), max_gap=3)
    assert np.isnan(filled.iloc[0]) and np.isnan(filled.iloc[-1]) and not mask.any()


def test_zero_max_gap_disables_imputation():
    series = _s([1, NAN, 3])
    filled, mask = impute_short_gaps(series, max_gap=0)
    assert filled.isna().sum() == 1 and not mask.any()


def test_measured_values_are_never_changed_and_input_is_not_mutated():
    series = _s([1, NAN, 3, 10, NAN, 12])
    before = series.copy()
    filled, _ = impute_short_gaps(series, max_gap=2)
    assert filled.iloc[[0, 2, 3, 5]].tolist() == [1, 3, 10, 12]
    pd.testing.assert_series_equal(series, before)


def test_two_gaps_with_different_lengths():
    filled, mask = impute_short_gaps(_s([1, NAN, 3, NAN, NAN, NAN, 7]), max_gap=2)
    assert filled.iloc[1] == 2 and np.isnan(filled.iloc[3:6]).all()
    assert mask.sum() == 1


# --- per-series statistics ---------------------------------------------------------------------


def test_summarise_series_ratios():
    raw = _s([1, NAN, NAN, 4, NAN, NAN, NAN, NAN, 9, 10])
    filled, imputed = impute_short_gaps(raw, max_gap=2)
    stats = summarise_series(raw, filled, imputed)
    assert stats["expected_rows"] == 10
    assert stats["present_rows"] == 4
    assert stats["missing_rows"] == 6  # raw gaps before imputation
    assert stats["imputed_rows"] == 2
    assert stats["residual_missing_rows"] == 4
    assert stats["missing_ratio"] == pytest.approx(0.6)
    assert stats["imputed_ratio"] == pytest.approx(0.2)
    assert stats["residual_missing_ratio"] == pytest.approx(0.4)
    assert stats["longest_gap_h"] == 4  # after imputation


def test_summarise_series_all_present_and_empty():
    full = _s([1.0, 2.0, 3.0])
    stats = summarise_series(full, full, pd.Series(False, index=full.index))
    assert stats["missing_ratio"] == 0 and stats["longest_gap_h"] == 0
    empty = _s([])
    assert summarise_series(empty, empty, pd.Series(dtype=bool))["expected_rows"] == 0


def test_zero_share_is_reported():
    series = _s([0.0, 0.0, 1.0, NAN])
    stats = summarise_series(series, series, pd.Series(False, index=series.index))
    assert stats["zero_ratio"] == pytest.approx(2 / 3)


# --- geographic consistency ---------------------------------------------------------------------

BBOX = {
    "Europe/Dublin": {"min_lat": 51.0, "max_lat": 56.0, "min_lon": -11.0, "max_lon": -5.5},
    "US/Eastern": {"min_lat": 24.0, "max_lat": 50.0, "min_lon": -85.0, "max_lon": -66.0},
}


def test_coordinates_outside_the_timezone_region_are_flagged():
    catalogue = pd.DataFrame(
        {
            "building_id": ["ok", "wrong_sign", "unlocated", "no_bbox"],
            "timezone": ["Europe/Dublin", "Europe/Dublin", "US/Eastern", "Asia/Tokyo"],
            "lat": [53.35, 53.35, NAN, 35.0],
            "lon": [-6.26, 6.26, NAN, 139.0],
        }
    )
    issues = check_geographic_consistency(catalogue, BBOX)
    flagged = dict(zip(issues["building_id"], issues["issue"], strict=True))
    assert "ok" not in flagged
    assert flagged["wrong_sign"] == "outside_timezone_region"
    assert flagged["unlocated"] == "no_coordinates"
    assert flagged["no_bbox"] == "no_reference_region"


def test_no_issues_gives_empty_frame_with_columns():
    catalogue = pd.DataFrame(
        {"building_id": ["a"], "timezone": ["Europe/Dublin"], "lat": [53.35], "lon": [-6.26]}
    )
    out = check_geographic_consistency(catalogue, BBOX)
    assert out.empty and list(out.columns) == ["building_id", "issue", "detail"]
