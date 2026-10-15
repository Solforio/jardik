"""Metadata filters. Each step records how many buildings it removed and why (the funnel)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from jardik.core.schemas import LocationQuality, derive_location_quality
from jardik.core.schemas.enums import LOCATION_QUALITY_ORDER
from jardik.dataset_manager.request import ExtractionRequest

FunnelStep = dict[str, Any]


def _norm(values: pd.Series) -> pd.Series:
    return values.astype("string").str.strip().str.casefold()


def location_quality_of(catalogue: pd.DataFrame) -> pd.Series:
    """Best available location level of each building, derived from the data itself."""
    return pd.Series(
        [
            derive_location_quality(
                None if pd.isna(r.lat) else r.lat,
                None if pd.isna(r.lon) else r.lon,
                None if pd.isna(r.postal_code) else r.postal_code,
                None if pd.isna(r.municipality) else r.municipality,
                None if pd.isna(r.province) else r.province,
                None if pd.isna(r.region) else r.region,
            ).value
            for r in catalogue.itertuples(index=False)
        ],
        index=catalogue.index,
    )


def apply_metadata_filters(
    catalogue: pd.DataFrame, request: ExtractionRequest
) -> tuple[pd.DataFrame, list[FunnelStep]]:
    """Apply every filter that the request specifies, in a fixed order.

    Returns the buildings that pass and the funnel: one dict per step with the buildings
    remaining, the number removed and a human-readable detail. The input is not modified.
    """
    current = catalogue.copy()
    funnel: list[FunnelStep] = [
        {"step": "dataset_catalogue", "remaining": len(current), "removed": 0, "detail": ""}
    ]

    def record(step: str, keep: pd.Series, detail: str) -> None:
        nonlocal current
        before = len(current)
        current = current[keep.fillna(False).to_numpy(dtype=bool)]
        funnel.append(
            {
                "step": step,
                "remaining": len(current),
                "removed": before - len(current),
                "detail": detail,
            }
        )

    area = request.area
    if request.usage_categories:
        wanted = [u.value for u in request.usage_categories]
        record(
            "usage_category",
            current["usage_category"].isin(wanted),
            "keep " + ", ".join(wanted),
        )
    if area.site_ids:
        record("site", current["site_id"].isin(area.site_ids), "keep " + ", ".join(area.site_ids))

    for step, column, values in (
        ("country", "country", area.countries),
        ("region", "region", area.regions),
        ("province", "province", area.provinces),
        ("municipality", "municipality", area.municipalities),
        ("postal_code", "postal_code", area.postal_codes),
    ):
        if values:
            unknown = int(current[column].isna().sum())
            record(
                step,
                _norm(current[column]).isin([v.strip().casefold() for v in values]),
                f"keep {', '.join(values)}; {unknown} with unknown value removed",
            )

    if area.bbox is not None:
        min_lat, min_lon, max_lat, max_lon = area.bbox
        inside = current["lat"].between(min_lat, max_lat) & current["lon"].between(min_lon, max_lon)
        record("bbox", inside, f"lat [{min_lat}, {max_lat}], lon [{min_lon}, {max_lon}]")

    if request.min_location_quality is not None:
        rank = {q.value: i for i, q in enumerate(LOCATION_QUALITY_ORDER)}
        limit = rank[LocationQuality(request.min_location_quality).value]
        quality = location_quality_of(current).map(rank)
        record(
            "min_location_quality",
            quality <= limit,
            f"at least {LocationQuality(request.min_location_quality).value}",
        )

    if request.contracted_power_kw is not None:
        low, high = request.contracted_power_kw
        power = current["contracted_power_kw"]
        record(
            "contracted_power",
            power.between(low, high),
            f"range [{low}, {high}] kW; {int(power.isna().sum())} with unknown value removed",
        )

    return current.reset_index(drop=True), funnel
