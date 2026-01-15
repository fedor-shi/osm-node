"""Base interface for index writers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class BaseWriter(ABC):
    """Abstract base class for index writers.

    Writers are responsible for converting sorted unique node ID arrays
    into persistent index files.
    """

    def __init__(self, output_dir: Path):
        """Initialize the writer.

        Args:
            output_dir: Directory where index files will be written
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    @abstractmethod
    def extension(self) -> str:
        """File extension for this index format (e.g., '.u64', '.roar')."""
        ...

    @abstractmethod
    def write(self, feature_name: str, ids: np.ndarray) -> Path:
        """Write sorted unique IDs to an index file.

        Args:
            feature_name: Name of the feature
            ids: Sorted unique uint64 array of node IDs

        Returns:
            Path to the written index file
        """
        ...

    def get_output_path(self, feature_name: str) -> Path:
        """Get the output path for a feature.

        Args:
            feature_name: Name of the feature

        Returns:
            Full path for the index file
        """
        return self.output_dir / f"{feature_name}{self.extension}"
