"""Roaring bitmap index loader."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from pyroaring import BitMap

from osm_node.index.base import BaseIndex


class Roaring64Wrapper:
    """Wrapper for handling 64-bit node IDs with 32-bit roaring bitmaps.

    For IDs that fit in 32 bits, uses a single BitMap.
    For larger IDs, uses the ROAR64 format with grouped bitmaps.
    """

    def __init__(self):
        """Initialize the wrapper."""
        self.is_64bit = False
        self.bitmap: BitMap | None = None
        self.high_groups: dict[int, BitMap] = {}

    @classmethod
    def deserialize(cls, data: bytes) -> "Roaring64Wrapper":
        """Deserialize from bytes.

        Args:
            data: Serialized bitmap data

        Returns:
            Deserialized wrapper
        """
        wrapper = cls()

        if len(data) == 0:
            wrapper.bitmap = BitMap()
            return wrapper

        # Check for ROAR64 magic header
        if data[:6] == b"ROAR64":
            wrapper.is_64bit = True
            offset = 8  # Skip magic + padding

            num_groups = int.from_bytes(data[offset : offset + 4], "little")
            offset += 4

            for _ in range(num_groups):
                high = int.from_bytes(data[offset : offset + 4], "little")
                offset += 4
                size = int.from_bytes(data[offset : offset + 4], "little")
                offset += 4
                bm = BitMap.deserialize(data[offset : offset + size])
                offset += size
                wrapper.high_groups[high] = bm
        else:
            # Standard 32-bit bitmap
            wrapper.bitmap = BitMap.deserialize(data)

        return wrapper

    def __contains__(self, node_id: int) -> bool:
        """Check if a node ID is in the bitmap.

        Args:
            node_id: Node ID to check

        Returns:
            True if present
        """
        if self.is_64bit:
            high = (node_id >> 32) & 0xFFFFFFFF
            low = node_id & 0xFFFFFFFF
            if high in self.high_groups:
                return low in self.high_groups[high]
            return False
        else:
            if self.bitmap is None:
                return False
            # For 32-bit, just check directly (truncate if needed)
            if node_id > 0xFFFFFFFF:
                return False
            return node_id in self.bitmap

    def __len__(self) -> int:
        """Get total number of elements."""
        if self.is_64bit:
            return sum(len(bm) for bm in self.high_groups.values())
        else:
            return len(self.bitmap) if self.bitmap else 0


class RoaringIndex(BaseIndex):
    """Index loader for roaring bitmap files.

    Uses pyroaring for fast O(1) membership testing.
    """

    def __init__(self):
        """Initialize the index."""
        super().__init__()
        self.features: dict[str, Roaring64Wrapper] = {}

    @property
    def extension(self) -> str:
        """File extension for roaring files."""
        return ".roar"

    @classmethod
    def load_dir(cls, path: str | Path) -> "RoaringIndex":
        """Load all .roar files from a directory.

        Args:
            path: Directory containing .roar files

        Returns:
            Loaded index instance
        """
        index = cls()
        dir_path = Path(path)

        for file_path in dir_path.glob("*.roar"):
            feature_name = file_path.stem
            data = file_path.read_bytes()
            index.features[feature_name] = Roaring64Wrapper.deserialize(data)

        return index

    @classmethod
    def load_file(cls, path: str | Path, feature_name: str | None = None) -> "RoaringIndex":
        """Load a single .roar file.

        Args:
            path: Path to the .roar file
            feature_name: Name for the feature (defaults to filename stem)

        Returns:
            Loaded index instance
        """
        index = cls()
        file_path = Path(path)

        if feature_name is None:
            feature_name = file_path.stem

        data = file_path.read_bytes()
        index.features[feature_name] = Roaring64Wrapper.deserialize(data)

        return index

    def contains(self, feature: str, node_id: int) -> bool:
        """Check if a node ID is in the feature set.

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

        return node_id in self.features[feature]

    def count(self, feature: str, node_ids: Iterable[int]) -> int:
        """Count how many node IDs are in the feature set.

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

        wrapper = self.features[feature]
        return sum(1 for nid in node_ids if nid in wrapper)

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
                "count": len(wrapper),
                "is_64bit": wrapper.is_64bit,
            }
            for feature, wrapper in self.features.items()
        }
