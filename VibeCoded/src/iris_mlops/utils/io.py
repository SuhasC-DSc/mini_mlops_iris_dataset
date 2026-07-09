"""Filesystem and serialization helper utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib


class JoblibSerializer:
    """Persists and loads arbitrary Python objects using Joblib.

    This class encapsulates all Joblib I/O behind a small interface so that
    other components (trainer, predictor) depend on an abstraction rather
    than calling `joblib` directly, in line with the Dependency Inversion
    Principle.
    """

    @staticmethod
    def save(obj: Any, path: Path) -> None:
        """Persist an object to disk.

        Args:
            obj: The Python object to serialize.
            path: Destination file path. Parent directories are created
                automatically.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(obj, path)

    @staticmethod
    def load(path: Path) -> Any:
        """Load a previously persisted object.

        Args:
            path: Path to the serialized object.

        Returns:
            The deserialized Python object.

        Raises:
            FileNotFoundError: If the artifact does not exist.
        """
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")
        return joblib.load(path)


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write a dictionary to disk as pretty-printed JSON.

    Args:
        data: Dictionary to serialize.
        path: Destination path. Parent directories are created automatically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def read_json(path: Path) -> dict[str, Any]:
    """Read a JSON file from disk.

    Args:
        path: Path to the JSON file.

    Returns:
        The parsed dictionary.
    """
    with path.open("r", encoding="utf-8") as fh:
        result: dict[str, Any] = json.load(fh)
        return result
