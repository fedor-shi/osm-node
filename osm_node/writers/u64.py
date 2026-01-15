"""Sorted uint64 index writer."""

from pathlib import Path

import numpy as np

from osm_node.writers.base import BaseWriter


class SortedU64Writer(BaseWriter):
    """Writer for sorted uint64 index files.

    Writes node IDs as raw little-endian uint64 arrays.
    These files can be memory-mapped with numpy for near-zero RAM usage.
    """

    @property
    def extension(self) -> str:
        """File extension for u64 files."""
        return ".u64"

    def write(self, feature_name: str, ids: np.ndarray) -> Path:
        """Write sorted unique IDs to a .u64 file.

        Args:
            feature_name: Name of the feature
            ids: Sorted unique uint64 array of node IDs

        Returns:
            Path to the written index file
        """
        output_path = self.get_output_path(feature_name)

        # Ensure correct dtype
        if ids.dtype != np.uint64:
            ids = ids.astype(np.uint64)

        # Write as raw little-endian uint64
        ids.tofile(output_path)

        return output_path
