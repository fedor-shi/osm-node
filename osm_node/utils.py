"""Utility functions for sorting and merging node ID chunks."""

from __future__ import annotations

import heapq
import os
import struct
import tempfile
from pathlib import Path
from typing import BinaryIO, Iterator

import numpy as np

# Maximum number of IDs to hold in memory before flushing
DEFAULT_CHUNK_SIZE = 100_000

# Threshold for switching to external sort (10M IDs = 80MB)
EXTERNAL_SORT_THRESHOLD = 10_000_000


def write_ids_to_file(file: BinaryIO, ids: list[int]) -> None:
    """Write a list of uint64 IDs to a binary file.

    Args:
        file: Open binary file handle
        ids: List of node IDs to write
    """
    data = struct.pack(f"<{len(ids)}Q", *ids)
    file.write(data)


def read_ids_from_file(path: Path) -> np.ndarray:
    """Read all uint64 IDs from a binary file.

    Args:
        path: Path to the binary file

    Returns:
        NumPy array of uint64 IDs
    """
    if not path.exists() or path.stat().st_size == 0:
        return np.array([], dtype=np.uint64)
    return np.fromfile(path, dtype="<u8")


def iter_ids_from_file(path: Path, chunk_size: int = 1_000_000) -> Iterator[np.ndarray]:
    """Iterate over IDs from a binary file in chunks.

    Args:
        path: Path to the binary file
        chunk_size: Number of IDs per chunk

    Yields:
        NumPy arrays of uint64 IDs
    """
    if not path.exists():
        return

    with open(path, "rb") as f:
        while True:
            data = f.read(chunk_size * 8)
            if not data:
                break
            yield np.frombuffer(data, dtype="<u8")


def merge_sorted_files(
    input_paths: list[Path],
    output_path: Path,
    remove_inputs: bool = True,
) -> int:
    """Merge multiple sorted uint64 files into one, removing duplicates.

    Args:
        input_paths: List of paths to sorted input files
        output_paths: Path for the merged output file
        remove_inputs: If True, delete input files after merging

    Returns:
        Number of unique IDs written
    """
    if not input_paths:
        # Create empty output
        output_path.write_bytes(b"")
        return 0

    # For single small file, just read, unique, write
    if len(input_paths) == 1:
        ids = read_ids_from_file(input_paths[0])
        unique_ids = np.unique(ids)
        unique_ids.tofile(output_path)
        if remove_inputs:
            input_paths[0].unlink()
        return len(unique_ids)

    # K-way merge with heap
    count = 0
    last_id = None

    def file_iterator(path: Path) -> Iterator[int]:
        """Yield IDs from a file one at a time."""
        for chunk in iter_ids_from_file(path):
            yield from chunk

    # Initialize heap with (value, iterator_index, iterator)
    iterators = [file_iterator(p) for p in input_paths]
    heap: list[tuple[int, int, Iterator[int]]] = []

    for i, it in enumerate(iterators):
        try:
            val = next(it)
            heapq.heappush(heap, (val, i, it))
        except StopIteration:
            pass

    with open(output_path, "wb") as out:
        buffer: list[int] = []

        while heap:
            val, idx, it = heapq.heappop(heap)

            # Skip duplicates
            if last_id is None or val != last_id:
                buffer.append(val)
                last_id = val
                count += 1

                # Flush buffer periodically
                if len(buffer) >= 100_000:
                    write_ids_to_file(out, buffer)
                    buffer.clear()

            # Get next value from the same iterator
            try:
                next_val = next(it)
                heapq.heappush(heap, (next_val, idx, it))
            except StopIteration:
                pass

        # Write remaining buffer
        if buffer:
            write_ids_to_file(out, buffer)

    # Clean up input files
    if remove_inputs:
        for p in input_paths:
            p.unlink(missing_ok=True)

    return count


def sort_and_unique_chunks(
    chunk_paths: list[Path],
    output_path: Path,
    tmp_dir: Path | None = None,
    remove_chunks: bool = True,
) -> int:
    """Sort and deduplicate IDs from multiple unsorted chunk files.

    For small total sizes, loads all into memory. For large sizes,
    sorts each chunk individually then performs k-way merge.

    Args:
        chunk_paths: List of paths to unsorted chunk files
        output_path: Path for the final sorted output
        tmp_dir: Directory for intermediate sorted chunks
        remove_chunks: If True, delete input chunks after processing

    Returns:
        Number of unique IDs written
    """
    if not chunk_paths:
        output_path.write_bytes(b"")
        return 0

    # Calculate total size
    total_ids = sum(p.stat().st_size // 8 for p in chunk_paths if p.exists())

    if total_ids <= EXTERNAL_SORT_THRESHOLD:
        # Small enough to fit in memory
        all_ids = np.concatenate([read_ids_from_file(p) for p in chunk_paths])
        unique_ids = np.unique(all_ids)
        unique_ids.tofile(output_path)

        if remove_chunks:
            for p in chunk_paths:
                p.unlink(missing_ok=True)

        return len(unique_ids)

    # External sort: sort each chunk, then merge
    if tmp_dir is None:
        tmp_dir = Path(tempfile.gettempdir())

    sorted_chunk_paths: list[Path] = []

    for i, chunk_path in enumerate(chunk_paths):
        ids = read_ids_from_file(chunk_path)
        ids.sort()
        sorted_path = tmp_dir / f"sorted_{i:04d}.u64"
        ids.tofile(sorted_path)
        sorted_chunk_paths.append(sorted_path)

        if remove_chunks:
            chunk_path.unlink(missing_ok=True)

    # Merge sorted chunks
    return merge_sorted_files(sorted_chunk_paths, output_path, remove_inputs=True)


class ChunkedIdBuffer:
    """Buffer for accumulating node IDs and flushing to temp files.

    Accumulates IDs in memory up to a threshold, then flushes to
    numbered chunk files on disk.
    """

    def __init__(
        self,
        feature_name: str,
        tmp_dir: Path,
        flush_threshold: int = DEFAULT_CHUNK_SIZE,
    ):
        """Initialize the buffer.

        Args:
            feature_name: Name of the feature (used in filenames)
            tmp_dir: Directory for temp chunk files
            flush_threshold: Number of IDs before auto-flush
        """
        self.feature_name = feature_name
        self.tmp_dir = Path(tmp_dir)
        self.flush_threshold = flush_threshold
        self.buffer: list[int] = []
        self.chunk_count = 0
        self.total_count = 0

        # Ensure tmp dir exists
        self.tmp_dir.mkdir(parents=True, exist_ok=True)

    def add(self, node_id: int) -> None:
        """Add a node ID to the buffer.

        Args:
            node_id: The node ID to add
        """
        self.buffer.append(node_id)
        self.total_count += 1

        if len(self.buffer) >= self.flush_threshold:
            self.flush()

    def flush(self) -> None:
        """Flush current buffer to a temp chunk file."""
        if not self.buffer:
            return

        chunk_path = self.tmp_dir / f"{self.feature_name}.part{self.chunk_count:04d}.u64"
        with open(chunk_path, "wb") as f:
            write_ids_to_file(f, self.buffer)

        self.buffer.clear()
        self.chunk_count += 1

    def get_chunk_paths(self) -> list[Path]:
        """Get list of all chunk file paths.

        Returns:
            List of paths to chunk files (flushes buffer first)
        """
        self.flush()
        return [
            self.tmp_dir / f"{self.feature_name}.part{i:04d}.u64"
            for i in range(self.chunk_count)
        ]

    def finalize(self, output_path: Path) -> int:
        """Finalize the buffer, sort/merge all chunks to output.

        Args:
            output_path: Path for the final sorted output

        Returns:
            Number of unique IDs written
        """
        chunk_paths = self.get_chunk_paths()
        return sort_and_unique_chunks(
            chunk_paths,
            output_path,
            tmp_dir=self.tmp_dir,
            remove_chunks=True,
        )
