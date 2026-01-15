"""Sorted uint64 index loader with memmap support."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from osm_node.index.base import BaseIndex


class SortedU64Index(BaseIndex):
    """Index loader for sorted uint64 files.

    Uses numpy memory mapping for near-zero RAM usage.
    Membership testing uses binary search (O(log n)).
    """

    def __init__(self):
        """Initialize the index."""
        super().__init__()
        self.features: dict[str, np.ndarray] = {}

    @property
    def extension(self) -> str:
        """File extension for u64 files."""
        return ".u64"

    @classmethod
    def load_dir(cls, path: str | Path) -> "SortedU64Index":
        """Load all .u64 files from a directory.

        Args:
            path: Directory containing .u64 files

        Returns:
            Loaded index instance
        """
        index = cls()
        dir_path = Path(path)

        for file_path in dir_path.glob("*.u64"):
            feature_name = file_path.stem

            if file_path.stat().st_size == 0:
                # Empty file - use empty array
                index.features[feature_name] = np.array([], dtype=np.uint64)
            else:
                # Memory-map the file for near-zero RAM usage
                index.features[feature_name] = np.memmap(
                    file_path,
                    dtype="<u8",
                    mode="r",
                )

        return index

    @classmethod
    def load_file(cls, path: str | Path, feature_name: str | None = None) -> "SortedU64Index":
        """Load a single .u64 file.

        Args:
            path: Path to the .u64 file
            feature_name: Name for the feature (defaults to filename stem)

        Returns:
            Loaded index instance
        """
        index = cls()
        file_path = Path(path)

        if feature_name is None:
            feature_name = file_path.stem

        if file_path.stat().st_size == 0:
            index.features[feature_name] = np.array([], dtype=np.uint64)
        else:
            index.features[feature_name] = np.memmap(file_path, dtype="<u8", mode="r")

        return index

    def contains(self, feature: str, node_id: int) -> bool:
        """Check if a node ID is in the feature set using binary search.

        Args:
            feature: Feature name to check
            node_id: Node ID to look up

        Returns:
            True if node_id is in the feature set

        Raises:
            KeyError: If feature is not loaded
        """
        if feature not in self.features:
            raise KeyError(f"Feature '{feature}' not loaded. Available: {self.available_features()}")

        arr = self.features[feature]
        if len(arr) == 0:
            return False

        # Binary search
        idx = np.searchsorted(arr, node_id)
        return bool(idx < len(arr) and arr[idx] == node_id)

    def count(self, feature: str, node_ids: Iterable[int]) -> int:
        """Count how many node IDs are in the feature set.

        Uses vectorized binary search for efficiency.

        Args:
            feature: Feature name to check
            node_ids: Iterable of node IDs to test

        Returns:
            Number of node IDs that are in the feature set

        Raises:
            KeyError: If feature is not loaded
        """
        if feature not in self.features:
            raise KeyError(f"Feature '{feature}' not loaded. Available: {self.available_features()}")

        arr = self.features[feature]
        if len(arr) == 0:
            return 0

        # Convert to numpy array for vectorized operations
        ids = np.asarray(list(node_ids), dtype=np.uint64)
        if len(ids) == 0:
            return 0

        # Vectorized binary search
        indices = np.searchsorted(arr, ids)

        # Check which indices are valid and match
        valid = indices < len(arr)
        matches = valid & (arr[np.minimum(indices, len(arr) - 1)] == ids)

        return int(np.sum(matches))

    def get_size(self, feature: str) -> int:
        """Get the number of IDs in a feature set.

        Args:
            feature: Feature name

        Returns:
            Number of node IDs in the feature set
        """
        if feature not in self.features:
            raise KeyError(f"Feature '{feature}' not loaded")
        return len(self.features[feature])

    def get_statistics(self) -> dict:
        """Get statistics about loaded indices.

        Returns:
            Dictionary with size info for each feature
        """
        return {
            feature: {
                "count": len(arr),
                "size_bytes": arr.nbytes if hasattr(arr, "nbytes") else len(arr) * 8,
            }
            for feature, arr in self.features.items()
        }
