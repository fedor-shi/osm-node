"""Roaring bitmap index writer."""

from pathlib import Path

import numpy as np
from pyroaring import BitMap

from osm_node.writers.base import BaseWriter


class RoaringWriter(BaseWriter):
    """Writer for roaring bitmap index files.

    Uses pyroaring to create compressed bitmap representations
    of node ID sets. Very compact and supports O(1) membership tests.
    """

    @property
    def extension(self) -> str:
        """File extension for roaring files."""
        return ".roar"

    def write(self, feature_name: str, ids: np.ndarray) -> Path:
        """Write sorted unique IDs to a .roar file.

        Args:
            feature_name: Name of the feature
            ids: Sorted unique uint64 array of node IDs

        Returns:
            Path to the written index file
        """
        output_path = self.get_output_path(feature_name)

        # pyroaring's BitMap64 handles 64-bit integers
        # Standard BitMap only handles 32-bit, so we need to handle this carefully
        # For OSM node IDs which can exceed 32 bits, we use a workaround:
        # We'll check if IDs fit in 32 bits, otherwise we fall back to storing
        # both high and low parts

        if len(ids) == 0:
            # Empty bitmap
            bm = BitMap()
            output_path.write_bytes(bm.serialize())
            return output_path

        max_id = ids.max() if len(ids) > 0 else 0

        if max_id <= 0xFFFFFFFF:
            # All IDs fit in 32 bits - use standard BitMap
            bm = BitMap(ids.astype(np.uint32))
            output_path.write_bytes(bm.serialize())
        else:
            # Some IDs exceed 32 bits - use a two-file approach
            # Store high 32 bits and low 32 bits separately
            # This is a pragmatic workaround for pyroaring's 32-bit limitation
            high = (ids >> 32).astype(np.uint32)
            low = (ids & 0xFFFFFFFF).astype(np.uint32)

            # Group by high bits
            unique_highs = np.unique(high)

            # Write a custom format:
            # - 8 bytes: magic "ROAR64\x00\x00"
            # - 4 bytes: number of high groups
            # For each group:
            #   - 4 bytes: high value
            #   - 4 bytes: size of serialized bitmap
            #   - N bytes: serialized bitmap of low values
            with open(output_path, "wb") as f:
                f.write(b"ROAR64\x00\x00")
                f.write(len(unique_highs).to_bytes(4, "little"))

                for h in unique_highs:
                    mask = high == h
                    low_vals = low[mask]
                    bm = BitMap(low_vals)
                    serialized = bm.serialize()

                    f.write(int(h).to_bytes(4, "little"))
                    f.write(len(serialized).to_bytes(4, "little"))
                    f.write(serialized)

        return output_path
