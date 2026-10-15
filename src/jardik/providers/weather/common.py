"""Helpers shared by weather adapters: build, clip and validate a canonical weather frame."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import pandas as pd

from jardik.core.interfaces import TimeWindow
from jardik.core.schemas import WEATHER_VARIABLES, validate_weather
from jardik.providers.base import ProviderError


def check_requested(variables: Sequence[str], available: Sequence[str]) -> list[str]:
    """Return the requested variables (deduplicated, order kept) or raise if any is unavailable."""
    if not variables:
        raise ProviderError("No weather variables requested")
    unknown = [v for v in variables if v not in WEATHER_VARIABLES]
    if unknown:
        raise ProviderError(f"Not canonical weather variable(s): {unknown}")
    missing = [v for v in variables if v not in available]
    if missing:
        raise ProviderError(
            f"Variable(s) {missing} not available from this provider. "
            f"Available: {sorted(available)}"
        )
    return list(dict.fromkeys(variables))


def build_weather_frame(
    *,
    point_id: pd.Series | str,
    ts_utc: pd.Series,
    source: str,
    product_type: str,
    issued_at: pd.Series | pd.Timestamp | None,
    series: Mapping[str, pd.Series],
) -> pd.DataFrame:
    """Assemble a frame that follows WEATHER_CONTRACT. `issued_at_utc` is always present."""
    index = ts_utc.index
    if issued_at is None:
        issued = pd.Series(pd.NaT, index=index, dtype="datetime64[ns, UTC]")
    elif isinstance(issued_at, pd.Timestamp):
        issued = pd.Series(issued_at, index=index).astype("datetime64[ns, UTC]")
    else:
        issued = issued_at.astype("datetime64[ns, UTC]")
    frame = pd.DataFrame(
        {
            "weather_point_id": point_id,
            "ts_utc": ts_utc.astype("datetime64[ns, UTC]"),
            "source": source,
            "product_type": product_type,
            "issued_at_utc": issued,
            **{name: values.astype(float) for name, values in series.items()},
        },
        index=index,
    )
    return frame


def finalize(frame: pd.DataFrame, window: TimeWindow | None = None) -> pd.DataFrame:
    """Clip to the half-open window, sort, validate against the contract and return."""
    out = frame
    if window is not None:
        keep = (out["ts_utc"] >= window.start.tz_convert("UTC")) & (
            out["ts_utc"] < window.end.tz_convert("UTC")
        )
        out = out.loc[keep]
    out = out.sort_values(["weather_point_id", "ts_utc"]).reset_index(drop=True)
    validate_weather(out)
    return out
