"""Tests for utility functions."""

import numpy as np
import pytest

from osm_node.utils import (
    ChunkedIdBuffer,
    merge_sorted_files,
    read_ids_from_file,
    sort_and_unique_chunks,
    write_ids_to_file,
)


class TestWriteReadIds:
    """Tests for write_ids_to_file and read_ids_from_file."""

    def test_write_and_read(self, tmp_path):
        """Test basic write and read cycle."""
        ids = [100, 200, 300, 400, 500]
        path = tmp_path / "test.u64"

        with open(path, "wb") as f:
            write_ids_to_file(f, ids)

        loaded = read_ids_from_file(path)
        np.testing.assert_array_equal(loaded, ids)

    def test_read_empty_file(self, tmp_path):
        """Test reading an empty file."""
        path = tmp_path / "empty.u64"
        path.write_bytes(b"")

        loaded = read_ids_from_file(path)
        assert len(loaded) == 0

    def test_read_nonexistent_file(self, tmp_path):
        """Test reading a nonexistent file."""
        path = tmp_path / "nonexistent.u64"
        loaded = read_ids_from_file(path)
        assert len(loaded) == 0


class TestMergeSortedFiles:
    """Tests for merge_sorted_files."""

    def test_merge_two_files(self, tmp_path):
        """Test merging two sorted files."""
        # Create two sorted files
        path1 = tmp_path / "a.u64"
        path2 = tmp_path / "b.u64"
        out_path = tmp_path / "merged.u64"

        np.array([100, 300, 500], dtype="<u8").tofile(path1)
        np.array([200, 400, 600], dtype="<u8").tofile(path2)

        count = merge_sorted_files([path1, path2], out_path, remove_inputs=False)

        assert count == 6
        loaded = np.fromfile(out_path, dtype="<u8")
        np.testing.assert_array_equal(loaded, [100, 200, 300, 400, 500, 600])

    def test_merge_with_duplicates(self, tmp_path):
        """Test merging files with duplicates."""
        path1 = tmp_path / "a.u64"
        path2 = tmp_path / "b.u64"
        out_path = tmp_path / "merged.u64"

        np.array([100, 200, 300], dtype="<u8").tofile(path1)
        np.array([200, 300, 400], dtype="<u8").tofile(path2)

        count = merge_sorted_files([path1, path2], out_path, remove_inputs=False)

        assert count == 4  # Unique count
        loaded = np.fromfile(out_path, dtype="<u8")
        np.testing.assert_array_equal(loaded, [100, 200, 300, 400])

    def test_merge_single_file(self, tmp_path):
        """Test merging a single file."""
        path = tmp_path / "single.u64"
        out_path = tmp_path / "merged.u64"

        np.array([100, 100, 200, 300], dtype="<u8").tofile(path)

        count = merge_sorted_files([path], out_path, remove_inputs=False)

        assert count == 3  # Unique
        loaded = np.fromfile(out_path, dtype="<u8")
        np.testing.assert_array_equal(loaded, [100, 200, 300])

    def test_merge_empty_list(self, tmp_path):
        """Test merging empty list of files."""
        out_path = tmp_path / "merged.u64"
        count = merge_sorted_files([], out_path)
        assert count == 0
        assert out_path.read_bytes() == b""


class TestSortAndUniqueChunks:
    """Tests for sort_and_unique_chunks."""

    def test_sort_single_chunk(self, tmp_path):
        """Test sorting a single unsorted chunk."""
        chunk_path = tmp_path / "chunk.u64"
        out_path = tmp_path / "sorted.u64"

        np.array([300, 100, 200, 100], dtype="<u8").tofile(chunk_path)

        count = sort_and_unique_chunks([chunk_path], out_path, remove_chunks=False)

        assert count == 3
        loaded = np.fromfile(out_path, dtype="<u8")
        np.testing.assert_array_equal(loaded, [100, 200, 300])

    def test_sort_multiple_chunks(self, tmp_path):
        """Test sorting and merging multiple unsorted chunks."""
        chunk1 = tmp_path / "chunk1.u64"
        chunk2 = tmp_path / "chunk2.u64"
        out_path = tmp_path / "sorted.u64"

        np.array([300, 100], dtype="<u8").tofile(chunk1)
        np.array([200, 100], dtype="<u8").tofile(chunk2)

        count = sort_and_unique_chunks([chunk1, chunk2], out_path, remove_chunks=False)

        assert count == 3  # 100, 200, 300
        loaded = np.fromfile(out_path, dtype="<u8")
        np.testing.assert_array_equal(loaded, [100, 200, 300])

    def test_empty_chunks(self, tmp_path):
        """Test with no chunks."""
        out_path = tmp_path / "sorted.u64"
        count = sort_and_unique_chunks([], out_path)
        assert count == 0


class TestChunkedIdBuffer:
    """Tests for ChunkedIdBuffer."""

    def test_add_and_finalize(self, tmp_path):
        """Test adding IDs and finalizing."""
        buffer = ChunkedIdBuffer("test", tmp_path, flush_threshold=3)

        buffer.add(100)
        buffer.add(200)
        buffer.add(300)
        buffer.add(400)  # Should trigger flush
        buffer.add(500)

        out_path = tmp_path / "test.u64"
        count = buffer.finalize(out_path)

        assert count == 5
        loaded = np.fromfile(out_path, dtype="<u8")
        np.testing.assert_array_equal(loaded, [100, 200, 300, 400, 500])

    def test_duplicate_handling(self, tmp_path):
        """Test that duplicates are removed on finalize."""
        buffer = ChunkedIdBuffer("test", tmp_path)

        for _ in range(3):
            buffer.add(100)
            buffer.add(200)

        out_path = tmp_path / "test.u64"
        count = buffer.finalize(out_path)

        assert count == 2  # Unique
        loaded = np.fromfile(out_path, dtype="<u8")
        np.testing.assert_array_equal(loaded, [100, 200])

    def test_unsorted_input(self, tmp_path):
        """Test that unsorted input is sorted on finalize."""
        buffer = ChunkedIdBuffer("test", tmp_path)

        buffer.add(300)
        buffer.add(100)
        buffer.add(200)

        out_path = tmp_path / "test.u64"
        buffer.finalize(out_path)

        loaded = np.fromfile(out_path, dtype="<u8")
        np.testing.assert_array_equal(loaded, [100, 200, 300])

    def test_total_count(self, tmp_path):
        """Test that total_count tracks all additions."""
        buffer = ChunkedIdBuffer("test", tmp_path)

        for i in range(10):
            buffer.add(i % 3)  # Will have duplicates

        assert buffer.total_count == 10
