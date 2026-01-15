"""Tests for schema module."""

import pytest

from osm_node.schema import (
    FeatureSpec,
    default_feature_specs,
    extended_feature_specs,
    get_feature_specs,
)


class TestFeatureSpec:
    """Tests for FeatureSpec dataclass."""

    def test_feature_spec_creation(self):
        """Test creating a FeatureSpec."""
        spec = FeatureSpec("test", lambda tags: "key" in tags)
        assert spec.name == "test"
        assert spec.predicate({"key": "value"}) is True
        assert spec.predicate({}) is False

    def test_feature_spec_frozen(self):
        """Test that FeatureSpec is immutable."""
        spec = FeatureSpec("test", lambda tags: True)
        with pytest.raises(AttributeError):
            spec.name = "modified"


class TestDefaultPredicates:
    """Tests for default feature predicates."""

    def test_signals_highway(self):
        """Test signals predicate with highway=traffic_signals."""
        specs = default_feature_specs()
        pred = specs["signals"].predicate
        assert pred({"highway": "traffic_signals"}) is True
        assert pred({"highway": "stop"}) is False

    def test_signals_crossing(self):
        """Test signals predicate with crossing=traffic_signals."""
        specs = default_feature_specs()
        pred = specs["signals"].predicate
        assert pred({"crossing": "traffic_signals"}) is True
        assert pred({"crossing": "unmarked"}) is False

    def test_stops(self):
        """Test stops predicate."""
        specs = default_feature_specs()
        pred = specs["stops"].predicate
        assert pred({"highway": "stop"}) is True
        assert pred({"highway": "give_way"}) is False

    def test_calming(self):
        """Test calming predicate matches any traffic_calming value."""
        specs = default_feature_specs()
        pred = specs["calming"].predicate
        assert pred({"traffic_calming": "hump"}) is True
        assert pred({"traffic_calming": "bump"}) is True
        assert pred({"traffic_calming": "table"}) is True
        assert pred({}) is False


class TestExtendedPredicates:
    """Tests for extended feature predicates."""

    def test_give_way(self):
        """Test give_way predicate."""
        specs = extended_feature_specs()
        pred = specs["give_way"].predicate
        assert pred({"highway": "give_way"}) is True
        assert pred({"highway": "stop"}) is False

    def test_crossing(self):
        """Test crossing predicate."""
        specs = extended_feature_specs()
        pred = specs["crossing"].predicate
        assert pred({"highway": "crossing"}) is True
        assert pred({"highway": "traffic_signals"}) is False

    def test_level_crossing(self):
        """Test level_crossing predicate."""
        specs = extended_feature_specs()
        pred = specs["level_crossing"].predicate
        assert pred({"railway": "level_crossing"}) is True
        assert pred({"railway": "station"}) is False


class TestGetFeatureSpecs:
    """Tests for get_feature_specs function."""

    def test_default_features(self):
        """Test that None returns default features."""
        specs = get_feature_specs(None)
        assert set(specs.keys()) == {"signals", "stops", "calming"}

    def test_specific_features(self):
        """Test requesting specific features."""
        specs = get_feature_specs(["signals", "give_way"])
        assert set(specs.keys()) == {"signals", "give_way"}

    def test_unknown_feature_raises(self):
        """Test that unknown features raise ValueError."""
        with pytest.raises(ValueError, match="Unknown feature 'nonexistent'"):
            get_feature_specs(["nonexistent"])

    def test_empty_list(self):
        """Test that empty list returns empty dict."""
        specs = get_feature_specs([])
        assert specs == {}
