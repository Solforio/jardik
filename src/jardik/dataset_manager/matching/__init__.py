"""Weather-building-time matching (critical path of the Dataset Manager)."""

from jardik.dataset_manager.matching.policy import MatchPolicy
from jardik.dataset_manager.matching.spatial import assign_weather_points
from jardik.dataset_manager.matching.temporal import MatchResult, match_weather
from jardik.dataset_manager.matching.validate import MatchingError, validate_join

__all__ = [
    "MatchPolicy",
    "MatchResult",
    "MatchingError",
    "assign_weather_points",
    "match_weather",
    "validate_join",
]
