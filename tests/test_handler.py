"""Tests for handler module."""

from __future__ import annotations

import pytest

from osm_node.handler import OsmiumTaggingHandler
from osm_node.schema import FeatureSpec, default_feature_specs


class MockNode:
    """Mock osmium Node for testing."""

    def __init__(self, node_id: int, tags: dict | None = None):
        self.id = node_id
        self.tags = tags if tags else {}


class TestOsmiumTaggingHandler:
    """Tests for OsmiumTaggingHandler."""

    def test_handler_initialization(self, tmp_path):
        """Test handler initializes correctly."""
        specs = default_feature_specs()
        handler = OsmiumTaggingHandler(specs, tmp_path)

        assert handler.nodes_processed == 0
        assert handler.nodes_matched == 0
        assert set(handler.buffers.keys()) == {"signals", "stops", "calming"}

    def test_node_without_tags(self, tmp_path):
        """Test that nodes without tags are skipped."""
        specs = default_feature_specs()
        handler = OsmiumTaggingHandler(specs, tmp_path)

        # Simulate processing a node without tags
        node = MockNode(123)
        handler.node(node)

        assert handler.nodes_processed == 1
        assert handler.nodes_matched == 0

    def test_node_with_matching_tags(self, tmp_path):
        """Test that matching nodes are captured."""
        specs = default_feature_specs()
        handler = OsmiumTaggingHandler(specs, tmp_path)

        # Traffic signal node
        node = MockNode(123, {"highway": "traffic_signals"})
        handler.node(node)

        assert handler.nodes_processed == 1
        assert handler.nodes_matched == 1
        assert handler.buffers["signals"].total_count == 1

    def test_node_matching_multiple_features(self, tmp_path):
        """Test node matching multiple features."""
        # Create custom specs where a node could match multiple
        specs = {
            "highway_any": FeatureSpec("highway_any", lambda t: "highway" in t),
            "signals": FeatureSpec("signals", lambda t: t.get("highway") == "traffic_signals"),
        }
        handler = OsmiumTaggingHandler(specs, tmp_path)

        node = MockNode(123, {"highway": "traffic_signals"})
        handler.node(node)

        # Should match both features
        assert handler.buffers["highway_any"].total_count == 1
        assert handler.buffers["signals"].total_count == 1

    def test_get_statistics(self, tmp_path):
        """Test statistics reporting."""
        specs = default_feature_specs()
        handler = OsmiumTaggingHandler(specs, tmp_path)

        # Process some nodes
        handler.node(MockNode(1, {"highway": "traffic_signals"}))
        handler.node(MockNode(2, {"highway": "stop"}))
        handler.node(MockNode(3, {}))
        handler.node(MockNode(4, {"traffic_calming": "hump"}))

        stats = handler.get_statistics()
        assert stats["nodes_processed"] == 4
        assert stats["nodes_matched"] == 3
        assert stats["feature_counts"]["signals"] == 1
        assert stats["feature_counts"]["stops"] == 1
        assert stats["feature_counts"]["calming"] == 1

    def test_finalize_buffers(self, tmp_path):
        """Test that buffers can be finalized to sorted output."""
        specs = default_feature_specs()
        handler = OsmiumTaggingHandler(specs, tmp_path)

        # Add some nodes in non-sorted order
        handler.node(MockNode(300, {"highway": "traffic_signals"}))
        handler.node(MockNode(100, {"highway": "traffic_signals"}))
        handler.node(MockNode(200, {"highway": "traffic_signals"}))

        # Finalize
        import numpy as np

        out_path = tmp_path / "signals.u64"
        count = handler.buffers["signals"].finalize(out_path)

        assert count == 3
        loaded = np.fromfile(out_path, dtype="<u8")
        np.testing.assert_array_equal(loaded, [100, 200, 300])
