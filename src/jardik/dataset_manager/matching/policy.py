"""Declared matching policy: how weather is tied to load records in space and time."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from jardik.core.schemas.enums import MatchingMethod


class MatchPolicy(BaseModel):
    """Everything that decides a match is declared here and recorded in the manifest.

    method:
      exact             weather timestamp must equal the load timestamp
      nearest_previous  latest weather record at or before the load timestamp, within the threshold
      nearest           closest weather record in either direction, within the threshold
      interpolate       linear interpolation between the two surrounding records (opt-in)
    """

    model_config = ConfigDict(extra="forbid")

    method: MatchingMethod = MatchingMethod.EXACT
    max_time_offset_min: int = Field(default=60, ge=0)
    max_distance_km: float = Field(default=50.0, gt=0)
    min_coverage: float = Field(
        default=0.9, ge=0, le=1, description="Minimum share of a building's records with weather."
    )

    @property
    def tolerance_min(self) -> int:
        """Effective time tolerance: exact matching never tolerates an offset."""
        return 0 if self.method == MatchingMethod.EXACT else self.max_time_offset_min
