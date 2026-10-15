from datetime import date

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from jardik.forecasting.splits import SplitSpec, assign_roles, partition


def _buildings(n_sites: int = 5, per_site: int = 4) -> pd.DataFrame:
    rows = [(f"bdg2:s{s}_b{i}", f"S{s}") for s in range(n_sites) for i in range(per_site)]
    return pd.DataFrame(rows, columns=["building_id", "site_id"])


def _spec(**overrides) -> SplitSpec:
    base = {
        "train_end": date(2024, 4, 1),
        "val_end": date(2024, 5, 1),
        "test_end": date(2024, 6, 1),
    }
    base.update(overrides)
    return SplitSpec(**base)


# --- specification -----------------------------------------------------------------------------


def test_dates_must_be_ordered_and_fractions_sensible():
    with pytest.raises(ValidationError):
        _spec(train_end=date(2024, 5, 1), val_end=date(2024, 4, 1))
    with pytest.raises(ValidationError):
        _spec(test_end=date(2024, 4, 15))
    with pytest.raises(ValidationError):
        _spec(val_building_frac=0.6, test_building_frac=0.5)
    with pytest.raises(ValidationError):
        _spec(test_building_frac=0.0)


# --- building roles ----------------------------------------------------------------------------


def test_roles_cover_every_building_exactly_once():
    buildings = _buildings()
    roles = assign_roles(buildings, _spec())
    assert set(roles.index) == set(buildings["building_id"])
    assert set(roles) == {"train", "val", "test"}
    counts = roles.value_counts()
    assert counts["test"] == 5 and counts["val"] == 3 and counts["train"] == 12


def test_roles_are_deterministic_and_independent_of_row_order():
    buildings = _buildings()
    a = assign_roles(buildings, _spec(seed=3))
    shuffled = buildings.sample(frac=1.0, random_state=1)
    b = assign_roles(shuffled, _spec(seed=3)).reindex(a.index)
    pd.testing.assert_series_equal(a, b)
    assert not a.equals(assign_roles(buildings, _spec(seed=4)))


def test_by_site_keeps_whole_sites_together():
    buildings = _buildings()
    roles = assign_roles(buildings, _spec(by="site"))
    per_site = buildings.assign(role=buildings["building_id"].map(roles)).groupby("site_id")["role"]
    assert (per_site.nunique() == 1).all()
    assert set(roles) == {"train", "val", "test"}


def test_too_few_buildings_or_sites_is_an_error():
    with pytest.raises(ValueError, match="at least"):
        assign_roles(_buildings(n_sites=1, per_site=2), _spec())
    with pytest.raises(ValueError, match="site"):
        assign_roles(_buildings(n_sites=2, per_site=10), _spec(by="site"))


# --- partitions by building and time ------------------------------------------------------------


@pytest.fixture
def frame() -> tuple[pd.DataFrame, pd.Series]:
    buildings = _buildings(n_sites=5, per_site=4)
    roles = assign_roles(buildings, _spec())
    hours = pd.date_range("2024-03-25", "2024-06-05", freq="h", tz="UTC")
    rows = pd.DataFrame(
        {
            "building_id": np.repeat(buildings["building_id"].to_numpy(), len(hours)),
            "ts_utc": np.tile(hours, len(buildings)),
        }
    )
    rows["has_history"] = True
    rows["is_measured"] = True
    return rows, roles


def test_partitions_respect_building_roles_and_period_boundaries(frame):
    rows, roles = frame
    spec = _spec()
    parts = partition(rows, roles, spec)
    assert set(parts) == {"train", "val", "test_time", "test_building", "test_both"}
    role = lambda df: set(df["building_id"].map(roles))  # noqa: E731
    t0, t1, t2 = (pd.Timestamp(d, tz="UTC") for d in (spec.train_end, spec.val_end, spec.test_end))

    assert role(parts["train"]) == {"train"} and parts["train"]["ts_utc"].max() < t0
    assert role(parts["val"]) == {"val"}
    assert parts["val"]["ts_utc"].min() >= t0 and parts["val"]["ts_utc"].max() < t1
    assert role(parts["test_time"]) == {"train"} and parts["test_time"]["ts_utc"].min() >= t1
    assert role(parts["test_building"]) == {"test"} and parts["test_building"]["ts_utc"].max() < t0
    assert role(parts["test_both"]) == {"test"}
    assert parts["test_both"]["ts_utc"].min() >= t1 and parts["test_both"]["ts_utc"].max() < t2


def test_the_main_test_set_shares_neither_buildings_nor_time_with_training(frame):
    rows, roles = frame
    parts = partition(rows, roles, _spec())
    train_buildings = set(parts["train"]["building_id"])
    both = parts["test_both"]
    assert not (set(both["building_id"]) & train_buildings)
    assert both["ts_utc"].min() > parts["train"]["ts_utc"].max()


def test_unusable_rows_are_dropped_by_default(frame):
    rows, roles = frame
    rows = rows.copy()
    rows.loc[rows.index[:1000], "has_history"] = False
    rows.loc[rows.index[1000:2000], "is_measured"] = False
    kept = sum(len(p) for p in partition(rows, roles, _spec()).values())
    everything = sum(len(p) for p in partition(rows, roles, _spec(), only_usable=False).values())
    assert kept < everything


def test_open_ended_test_period_runs_to_the_last_row(frame):
    rows, roles = frame
    parts = partition(rows, roles, _spec(test_end=None))
    assert parts["test_both"]["ts_utc"].max() == rows["ts_utc"].max()
