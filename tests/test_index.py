"""Tests for index modules."""

import numpy as np
import pytest

from osm_node.index import RoaringIndex, SortedU64Index
from osm_node.writers import RoaringWriter, SortedU64Writer


class TestSortedU64Index:
    """Tests for SortedU64Index."""

    @pytest.fixture
    def populated_index(self, tmp_path):
        """Create a test index with sample data."""
        writer = SortedU64Writer(tmp_path)
        writer.write("signals", np.array([100, 200, 300, 400, 500], dtype=np.uint64))
        writer.write("stops", np.array([150, 250], dtype=np.uint64))
        writer.write("empty", np.array([], dtype=np.uint64))
        return SortedU64Index.load_dir(tmp_path)

    def test_load_dir(self, populated_index):
        """Test loading indices from directory."""
        assert set(populated_index.available_features()) == {"signals", "stops", "empty"}

    def test_contains_true(self, populated_index):
        """Test contains returns True for present IDs."""
        assert populated_index.contains("signals", 100) is True
        assert populated_index.contains("signals", 300) is True
        assert populated_index.contains("signals", 500) is True

    def test_contains_false(self, populated_index):
        """Test contains returns False for absent IDs."""
        assert populated_index.contains("signals", 101) is False
        assert populated_index.contains("signals", 999) is False
        assert populated_index.contains("signals", 0) is False

    def test_contains_empty_feature(self, populated_index):
        """Test contains on empty feature."""
        assert populated_index.contains("empty", 100) is False

    def test_contains_unknown_feature_raises(self, populated_index):
        """Test contains raises for unknown feature."""
        with pytest.raises(KeyError, match="Feature 'unknown' not loaded"):
            populated_index.contains("unknown", 100)

    def test_count_basic(self, populated_index):
        """Test counting matching IDs."""
        # All match
        count = populated_index.count("signals", [100, 200, 300])
        assert count == 3

        # Some match
        count = populated_index.count("signals", [100, 101, 200])
        assert count == 2

        # None match
        count = populated_index.count("signals", [101, 102, 103])
        assert count == 0

    def test_count_empty_input(self, populated_index):
        """Test counting with empty input."""
        count = populated_index.count("signals", [])
        assert count == 0

    def test_count_empty_feature(self, populated_index):
        """Test counting on empty feature."""
        count = populated_index.count("empty", [100, 200, 300])
        assert count == 0

    def test_count_all(self, populated_index):
        """Test count_all across features."""
        node_ids = [100, 150, 200, 250, 300]
        counts = populated_index.count_all(node_ids)
        assert counts["signals"] == 3  # 100, 200, 300
        assert counts["stops"] == 2  # 150, 250
        assert counts["empty"] == 0

    def test_get_size(self, populated_index):
        """Test get_size returns correct counts."""
        assert populated_index.get_size("signals") == 5
        assert populated_index.get_size("stops") == 2
        assert populated_index.get_size("empty") == 0

    def test_load_single_file(self, tmp_path):
        """Test loading a single file."""
        writer = SortedU64Writer(tmp_path)
        writer.write("test", np.array([1, 2, 3], dtype=np.uint64))

        index = SortedU64Index.load_file(tmp_path / "test.u64")
        assert index.contains("test", 2) is True
        assert index.contains("test", 4) is False


class TestRoaringIndex:
    """Tests for RoaringIndex."""

    @pytest.fixture
    def populated_index(self, tmp_path):
        """Create a test index with sample data."""
        writer = RoaringWriter(tmp_path)
        writer.write("signals", np.array([100, 200, 300, 400, 500], dtype=np.uint64))
        writer.write("stops", np.array([150, 250], dtype=np.uint64))
        writer.write("empty", np.array([], dtype=np.uint64))
        return RoaringIndex.load_dir(tmp_path)

    def test_load_dir(self, populated_index):
        """Test loading indices from directory."""
        assert set(populated_index.available_features()) == {"signals", "stops", "empty"}

    def test_contains_true(self, populated_index):
        """Test contains returns True for present IDs."""
        assert populated_index.contains("signals", 100) is True
        assert populated_index.contains("signals", 300) is True
        assert populated_index.contains("signals", 500) is True

    def test_contains_false(self, populated_index):
        """Test contains returns False for absent IDs."""
        assert populated_index.contains("signals", 101) is False
        assert populated_index.contains("signals", 999) is False
        assert populated_index.contains("signals", 0) is False

    def test_contains_empty_feature(self, populated_index):
        """Test contains on empty feature."""
        assert populated_index.contains("empty", 100) is False

    def test_count_basic(self, populated_index):
        """Test counting matching IDs."""
        count = populated_index.count("signals", [100, 200, 300])
        assert count == 3

        count = populated_index.count("signals", [100, 101, 200])
        assert count == 2

        count = populated_index.count("signals", [101, 102, 103])
        assert count == 0

    def test_count_all(self, populated_index):
        """Test count_all across features."""
        node_ids = [100, 150, 200, 250, 300]
        counts = populated_index.count_all(node_ids)
        assert counts["signals"] == 3
        assert counts["stops"] == 2
        assert counts["empty"] == 0


class TestRoaringIndex64Bit:
    """Tests for RoaringIndex with 64-bit IDs."""

    @pytest.fixture
    def large_id_index(self, tmp_path):
        """Create an index with IDs exceeding 32 bits."""
        writer = RoaringWriter(tmp_path)
        large_ids = np.array([2**40, 2**40 + 100, 2**50], dtype=np.uint64)
        writer.write("large", large_ids)
        return RoaringIndex.load_dir(tmp_path)

    def test_contains_large_ids(self, large_id_index):
        """Test contains with large IDs."""
        assert large_id_index.contains("large", 2**40) is True
        assert large_id_index.contains("large", 2**40 + 100) is True
        assert large_id_index.contains("large", 2**50) is True
        assert large_id_index.contains("large", 2**40 + 1) is False

    def test_count_large_ids(self, large_id_index):
        """Test counting large IDs."""
        count = large_id_index.count("large", [2**40, 2**40 + 1, 2**50])
        assert count == 2

    def test_is_64bit_flag(self, large_id_index):
        """Test that 64-bit flag is set correctly."""
        stats = large_id_index.get_statistics()
        assert stats["large"]["is_64bit"] is True
