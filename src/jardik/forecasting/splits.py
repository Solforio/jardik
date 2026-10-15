"""Evaluation splits: by building AND by time, never random rows.

Buildings are divided into train / validation / test roles. Time is divided into a training
period, a validation period and a test period. From these the backtest builds:

  train          training buildings,   training period   (what the model learns from)
  val            validation buildings, validation period (early stopping, never the test)
  test_time      training buildings,   test period       (known buildings, unseen future)
  test_building  test buildings,       training period   (unseen buildings, known period)
  test_both      test buildings,       test period       (unseen buildings AND unseen future:
                                                          the hardest case and the main one)
"""

from __future__ import annotations

from datetime import date
from typing import Literal, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

SCENARIOS = ("train", "val", "test_time", "test_building", "test_both")
MIN_BUILDINGS = 3


class SplitSpec(BaseModel):
    """Period boundaries are UTC calendar days: train < train_end <= val < val_end <= test."""

    model_config = ConfigDict(extra="forbid")

    train_end: date
    val_end: date
    test_end: date | None = None
    val_building_frac: float = Field(default=0.15, gt=0, lt=1)
    test_building_frac: float = Field(default=0.25, gt=0, lt=1)
    by: Literal["building", "site"] = "building"
    seed: int = 42

    @model_validator(mode="after")
    def _check(self) -> Self:
        if self.val_end <= self.train_end:
            raise ValueError("val_end must be after train_end")
        if self.test_end is not None and self.test_end <= self.val_end:
            raise ValueError("test_end must be after val_end")
        if self.val_building_frac + self.test_building_frac >= 1:
            raise ValueError("val and test building fractions must sum to less than 1")
        return self


def _ts(day: date | None) -> pd.Timestamp | None:
    return None if day is None else pd.Timestamp(day, tz="UTC")


def assign_roles(buildings: pd.DataFrame, spec: SplitSpec) -> pd.Series:
    """Give every building a role: 'train', 'val' or 'test' (deterministic for a given seed).

    buildings: columns building_id and site_id. With by='site' whole sites share one role, which
    tests generalisation to a new site (new weather, new operator) and is the harsher check.
    """
    table = buildings[["building_id", "site_id"]].drop_duplicates().sort_values("building_id")
    ids = table["building_id"].tolist()
    n = len(ids)
    n_test = max(1, round(n * spec.test_building_frac))
    n_val = max(1, round(n * spec.val_building_frac))
    if n < MIN_BUILDINGS or n_test + n_val >= n:
        raise ValueError(
            f"Splitting needs at least {MIN_BUILDINGS} buildings and room for a training set "
            f"(got {n} buildings: {n_test} test + {n_val} validation requested)"
        )
    rng = np.random.default_rng(spec.seed)
    roles = pd.Series("train", index=pd.Index(ids, name="building_id"), name="role")

    if spec.by == "building":
        order = [ids[i] for i in rng.permutation(n)]
        roles.loc[order[:n_test]] = "test"
        roles.loc[order[n_test : n_test + n_val]] = "val"
        return roles

    sites = sorted(table["site_id"].unique())
    if len(sites) < MIN_BUILDINGS:
        raise ValueError(
            f"Splitting by site needs at least {MIN_BUILDINGS} sites (got {len(sites)})"
        )
    size = table.groupby("site_id").size()
    shuffled = [sites[i] for i in rng.permutation(len(sites))]
    assigned: dict[str, str] = {}
    remaining = list(shuffled)
    for role, target in (("test", n_test), ("val", n_val)):
        count = 0
        # keep at least one site for every role that is still to be filled
        reserve = 1 if role == "val" else 2
        while count < target and len(remaining) > reserve:
            site = remaining.pop(0)
            assigned[site] = role
            count += int(size[site])
    for site in remaining:
        assigned[site] = "train"
    by_building = table["site_id"].map(assigned)
    return pd.Series(by_building.to_numpy(), index=pd.Index(ids, name="building_id"), name="role")


def partition(
    features: pd.DataFrame, roles: pd.Series, spec: SplitSpec, *, only_usable: bool = True
) -> dict[str, pd.DataFrame]:
    """Cut the feature frame into the five scenarios. With `only_usable`, rows without history or
    without a measured target are dropped."""
    t0, t1, t2 = _ts(spec.train_end), _ts(spec.val_end), _ts(spec.test_end)
    ts = features["ts_utc"]
    role = features["building_id"].map(roles)
    in_train = ts < t0
    in_val = (ts >= t0) & (ts < t1)
    in_test = (ts >= t1) if t2 is None else ((ts >= t1) & (ts < t2))

    usable = pd.Series(True, index=features.index)
    if only_usable:
        for column in ("has_history", "is_measured"):
            if column in features.columns:
                usable &= features[column].astype(bool)

    masks = {
        "train": (role == "train") & in_train,
        "val": (role == "val") & in_val,
        "test_time": (role == "train") & in_test,
        "test_building": (role == "test") & in_train,
        "test_both": (role == "test") & in_test,
    }
    return {name: features[mask & usable].copy() for name, mask in masks.items()}
