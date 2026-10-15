"""Optional, declared and counted imputation of short gaps."""

from __future__ import annotations

import numpy as np
import pandas as pd


def impute_short_gaps(values: pd.Series, max_gap: int) -> tuple[pd.Series, pd.Series]:
    """Linearly interpolate interior gaps of at most `max_gap` consecutive missing values.

    Longer gaps, and gaps at the start or end of the series, are left untouched (a partial fill
    of a long gap would invent data). Measured values are never changed.
    Returns (filled series, boolean mask of imputed positions). The input is not modified.
    """
    data = values.to_numpy(dtype=float, copy=True)
    imputed = np.zeros(len(data), dtype=bool)
    if max_gap > 0 and len(data):
        is_nan = np.isnan(data)
        edges = np.diff(np.concatenate(([False], is_nan, [False])).astype(int))
        starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
        for start, end in zip(starts, ends, strict=True):
            interior = start > 0 and end < len(data)
            if interior and (end - start) <= max_gap:
                data[start:end] = np.interp(
                    np.arange(start, end), [start - 1, end], [data[start - 1], data[end]]
                )
                imputed[start:end] = True
    return (
        pd.Series(data, index=values.index, name=values.name),
        pd.Series(imputed, index=values.index),
    )
