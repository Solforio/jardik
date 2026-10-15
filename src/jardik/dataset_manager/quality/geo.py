"""Geographic consistency of building coordinates."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

_COLUMNS = ["building_id", "issue", "detail"]


def check_geographic_consistency(
    catalogue: pd.DataFrame, bbox_by_timezone: Mapping[str, Mapping[str, float]]
) -> pd.DataFrame:
    """Flag buildings whose coordinates do not fit the region of their declared timezone.

    bbox_by_timezone: timezone -> {min_lat, max_lat, min_lon, max_lon}. A sign error in a
    longitude (a real case in Building Data Genome 2) lands the building in another continent
    and is caught here. Issues: no_coordinates, no_reference_region, outside_timezone_region.
    """
    rows: list[dict[str, str]] = []
    for b in catalogue.itertuples(index=False):
        if pd.isna(b.lat) or pd.isna(b.lon):
            rows.append({"building_id": b.building_id, "issue": "no_coordinates", "detail": ""})
            continue
        box = bbox_by_timezone.get(b.timezone)
        if box is None:
            rows.append(
                {
                    "building_id": b.building_id,
                    "issue": "no_reference_region",
                    "detail": f"no bounding box configured for {b.timezone}",
                }
            )
            continue
        inside = (
            box["min_lat"] <= b.lat <= box["max_lat"] and box["min_lon"] <= b.lon <= box["max_lon"]
        )
        if not inside:
            rows.append(
                {
                    "building_id": b.building_id,
                    "issue": "outside_timezone_region",
                    "detail": f"({b.lat}, {b.lon}) is outside the {b.timezone} region",
                }
            )
    return pd.DataFrame(rows, columns=_COLUMNS)
