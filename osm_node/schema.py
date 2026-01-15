"""Feature specifications for OSM node extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class FeatureSpec:
    """Specification for a feature to extract from OSM nodes.

    Attributes:
        name: Unique identifier for this feature (used in filenames)
        predicate: Function that takes a tag dict and returns True if the node matches
    """

    name: str
    predicate: Callable[[dict], bool]


def _pred_signal(tags: dict) -> bool:
    """Match traffic signals (highway=traffic_signals or crossing=traffic_signals)."""
    return tags.get("highway") == "traffic_signals" or tags.get("crossing") == "traffic_signals"


def _pred_stop(tags: dict) -> bool:
    """Match stop signs (highway=stop)."""
    return tags.get("highway") == "stop"


def _pred_calming(tags: dict) -> bool:
    """Match any traffic calming feature (traffic_calming=*)."""
    return "traffic_calming" in tags


def _pred_give_way(tags: dict) -> bool:
    """Match give way / yield signs (highway=give_way)."""
    return tags.get("highway") == "give_way"


def _pred_crossing(tags: dict) -> bool:
    """Match pedestrian crossings (highway=crossing)."""
    return tags.get("highway") == "crossing"


def _pred_level_crossing(tags: dict) -> bool:
    """Match railway level crossings (railway=level_crossing)."""
    return tags.get("railway") == "level_crossing"


def default_feature_specs() -> dict[str, FeatureSpec]:
    """Return the default set of feature specifications.

    Returns:
        Dictionary mapping feature names to their specifications.
        Default features: signals, stops, calming
    """
    return {
        "signals": FeatureSpec("signals", _pred_signal),
        "stops": FeatureSpec("stops", _pred_stop),
        "calming": FeatureSpec("calming", _pred_calming),
    }


def extended_feature_specs() -> dict[str, FeatureSpec]:
    """Return an extended set of feature specifications.

    Returns:
        Dictionary with additional features beyond the defaults:
        signals, stops, calming, give_way, crossing, level_crossing
    """
    specs = default_feature_specs()
    specs.update(
        {
            "give_way": FeatureSpec("give_way", _pred_give_way),
            "crossing": FeatureSpec("crossing", _pred_crossing),
            "level_crossing": FeatureSpec("level_crossing", _pred_level_crossing),
        }
    )
    return specs


def get_feature_specs(names: list[str] | None = None) -> dict[str, FeatureSpec]:
    """Get feature specifications by name.

    Args:
        names: List of feature names to include. If None, returns default specs.

    Returns:
        Dictionary of requested feature specifications.

    Raises:
        ValueError: If an unknown feature name is requested.
    """
    all_specs = extended_feature_specs()

    if names is None:
        return default_feature_specs()

    result = {}
    for name in names:
        if name not in all_specs:
            available = ", ".join(sorted(all_specs.keys()))
            raise ValueError(f"Unknown feature '{name}'. Available: {available}")
        result[name] = all_specs[name]

    return result
