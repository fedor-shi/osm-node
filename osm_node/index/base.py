"""Base interface for index loaders."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable


class BaseIndex(ABC):
    """Abstract base class for index loaders.

    Indexes provide fast membership testing and counting for node IDs.
    """

    def __init__(self):
        """Initialize the index."""
        self.features: dict = {}

    @classmethod
    @abstractmethod
    def load_dir(cls, path: str | Path) -> "BaseIndex":
        """Load all index files from a directory.

        Args:
            path: Directory containing index files

        Returns:
            Loaded index instance
        """
        ...

    @property
    @abstractmethod
    def extension(self) -> str:
        """File extension for this index format."""
        ...

    def available_features(self) -> list[str]:
        """Get list of available feature names.

        Returns:
            List of feature names that can be queried
        """
        return list(self.features.keys())

    @abstractmethod
    def contains(self, feature: str, node_id: int) -> bool:
        """Check if a node ID is in the feature set.

        Args:
            feature: Feature name to check
            node_id: Node ID to look up

        Returns:
            True if node_id is in the feature set
        """
        ...

    @abstractmethod
    def count(self, feature: str, node_ids: Iterable[int]) -> int:
        """Count how many node IDs are in the feature set.

        Args:
            feature: Feature name to check
            node_ids: Iterable of node IDs to test

        Returns:
            Number of node IDs that are in the feature set
        """
        ...

    def count_all(self, node_ids: Iterable[int]) -> dict[str, int]:
        """Count node IDs across all features.

        Args:
            node_ids: Iterable of node IDs to test

        Returns:
            Dictionary mapping feature names to counts
        """
        # Convert to list to allow multiple iterations
        ids_list = list(node_ids)
        return {feature: self.count(feature, ids_list) for feature in self.features}
