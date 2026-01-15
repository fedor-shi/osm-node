"""Tests for writer modules."""

import numpy as np
import pytest

from osm_node.writers import RoaringWriter, SortedU64Writer


class TestSortedU64Writer:
    """Tests for SortedU64Writer."""

    def test_write_basic(self, tmp_path):
        """Test writing a basic u64 file."""
        writer = SortedU64Writer(tmp_path)
        ids = np.array([100, 200, 300], dtype=np.uint64)
        path = writer.write("test", ids)

        assert path.exists()
        assert path.suffix == ".u64"
        assert path.name == "test.u64"

        # Read back and verify
        loaded = np.fromfile(path, dtype="<u8")
        np.testing.assert_array_equal(loaded, ids)

    def test_write_empty(self, tmp_path):
        """Test writing an empty array."""
        writer = SortedU64Writer(tmp_path)
        ids = np.array([], dtype=np.uint64)
        path = writer.write("empty", ids)

        assert path.exists()
        assert path.stat().st_size == 0

    def test_write_large_ids(self, tmp_path):
        """Test writing IDs that exceed 32 bits."""
        writer = SortedU64Writer(tmp_path)
        ids = np.array([2**40, 2**50, 2**60], dtype=np.uint64)
        path = writer.write("large", ids)

        loaded = np.fromfile(path, dtype="<u8")
        np.testing.assert_array_equal(loaded, ids)

    def test_creates_directory(self, tmp_path):
        """Test that writer creates output directory if needed."""
        out_dir = tmp_path / "nested" / "path"
        writer = SortedU64Writer(out_dir)
        ids = np.array([1, 2, 3], dtype=np.uint64)
        path = writer.write("test", ids)

        assert path.exists()


class TestRoaringWriter:
    """Tests for RoaringWriter."""

    def test_write_basic_32bit(self, tmp_path):
        """Test writing IDs that fit in 32 bits."""
        writer = RoaringWriter(tmp_path)
        ids = np.array([100, 200, 300], dtype=np.uint64)
        path = writer.write("test", ids)

        assert path.exists()
        assert path.suffix == ".roar"
        assert path.name == "test.roar"

        # Verify it's valid roaring data (doesn't start with ROAR64)
        data = path.read_bytes()
        assert not data.startswith(b"ROAR64")

    def test_write_empty(self, tmp_path):
        """Test writing an empty array."""
        writer = RoaringWriter(tmp_path)
        ids = np.array([], dtype=np.uint64)
        path = writer.write("empty", ids)

        assert path.exists()
        # Should be a valid (empty) serialized bitmap
        data = path.read_bytes()
        assert len(data) > 0

    def test_write_large_ids(self, tmp_path):
        """Test writing IDs that exceed 32 bits uses ROAR64 format."""
        writer = RoaringWriter(tmp_path)
        ids = np.array([2**40 + 1, 2**40 + 2, 2**50], dtype=np.uint64)
        path = writer.write("large", ids)

        data = path.read_bytes()
        assert data.startswith(b"ROAR64")

    def test_write_mixed_ids(self, tmp_path):
        """Test writing mix of small and large IDs."""
        writer = RoaringWriter(tmp_path)
        # Mix of IDs: some fit in 32 bits, some don't
        ids = np.array([100, 2**40, 2**40 + 1], dtype=np.uint64)
        path = writer.write("mixed", ids)

        data = path.read_bytes()
        # Should use ROAR64 because of large IDs
        assert data.startswith(b"ROAR64")
