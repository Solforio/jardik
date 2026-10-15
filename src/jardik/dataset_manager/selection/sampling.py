"""Deterministic ordering of candidate buildings (the same request always picks the same ones)."""

from __future__ import annotations

from itertools import zip_longest
from typing import Literal

import numpy as np
import pandas as pd

Sampling = Literal["random", "stratified_site", "first"]


def order_candidates(pool: pd.DataFrame, sampling: Sampling, seed: int) -> list[str]:
    """Return every building of `pool` in the order in which they should be considered.

    The caller takes buildings from the front until it has enough valid ones, so a building
    rejected later (for example for missing weather) is simply replaced by the next one.
    The result does not depend on the row order of `pool`.

    random           seeded shuffle
    stratified_site  seeded shuffle inside each site, then one building per site in turn
    first            alphabetical order
    """
    if pool.empty:
        return []
    ordered = pool.sort_values("building_id", kind="stable")
    ids = ordered["building_id"].tolist()
    rng = np.random.default_rng(seed)

    if sampling == "first":
        return ids
    if sampling == "random":
        return [ids[i] for i in rng.permutation(len(ids))]

    per_site: list[list[str]] = []
    for _, group in ordered.groupby("site_id", sort=True):
        members = group["building_id"].tolist()
        per_site.append([members[i] for i in rng.permutation(len(members))])
    interleaved = zip_longest(*per_site)
    return [building for round_ in interleaved for building in round_ if building is not None]
