"""PyOsmium handler for extracting control nodes from PBF files."""

from __future__ import annotations

from pathlib import Path

import osmium

from osm_node.schema import FeatureSpec
from osm_node.utils import ChunkedIdBuffer


class OsmiumTaggingHandler(osmium.SimpleHandler):
    """Osmium handler that extracts node IDs matching feature predicates.

    This handler streams through a PBF file, evaluates feature predicates
    on each node's tags, and writes matching node IDs to chunked temp files
    for later sorting and indexing.
    """

    def __init__(
        self,
        feature_specs: dict[str, FeatureSpec],
        tmp_dir: Path,
        flush_threshold: int = 100_000,
    ):
        """Initialize the handler.

        Args:
            feature_specs: Dictionary of feature specifications to extract
            tmp_dir: Directory for temporary chunk files
            flush_threshold: Number of IDs per feature before flushing to disk
        """
        super().__init__()
        self.feature_specs = feature_specs
        self.tmp_dir = Path(tmp_dir)
        self.flush_threshold = flush_threshold

        # Create a buffer for each feature
        self.buffers: dict[str, ChunkedIdBuffer] = {
            name: ChunkedIdBuffer(name, self.tmp_dir, flush_threshold)
            for name in feature_specs
        }

        # Statistics
        self.nodes_processed = 0
        self.nodes_matched = 0

    def node(self, n: osmium.osm.Node) -> None:
        """Process a node from the PBF file.

        Args:
            n: The osmium Node object
        """
        self.nodes_processed += 1

        # Skip nodes without tags (most nodes are just geometry points)
        if not n.tags:
            return

        # Convert tags to dict for predicate evaluation
        tags = dict(n.tags)
        matched = False

        # Evaluate all predicates
        for name, spec in self.feature_specs.items():
            if spec.predicate(tags):
                self.buffers[name].add(n.id)
                matched = True

        if matched:
            self.nodes_matched += 1

    def get_statistics(self) -> dict:
        """Get extraction statistics.

        Returns:
            Dictionary with processing statistics
        """
        return {
            "nodes_processed": self.nodes_processed,
            "nodes_matched": self.nodes_matched,
            "feature_counts": {
                name: buffer.total_count for name, buffer in self.buffers.items()
            },
        }


def extract_features(
    pbf_path: str | Path,
    feature_specs: dict[str, FeatureSpec],
    tmp_dir: Path,
    flush_threshold: int = 100_000,
) -> tuple[OsmiumTaggingHandler, dict[str, list[Path]]]:
    """Extract features from a PBF file.

    Args:
        pbf_path: Path to the PBF file
        feature_specs: Dictionary of feature specifications
        tmp_dir: Directory for temporary files
        flush_threshold: Number of IDs per feature before flushing

    Returns:
        Tuple of (handler with stats, dict mapping feature names to chunk paths)
    """
    handler = OsmiumTaggingHandler(feature_specs, tmp_dir, flush_threshold)

    # Process the PBF file - don't need node locations for just extracting IDs
    handler.apply_file(str(pbf_path), locations=False)

    # Get chunk paths for each feature
    chunk_paths = {name: buffer.get_chunk_paths() for name, buffer in handler.buffers.items()}

    return handler, chunk_paths
