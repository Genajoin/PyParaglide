"""
Pickle file I/O utilities.

Simple wrapper for saving/loading Python objects to pickle files.
"""

import os
import pickle
from pathlib import Path
from typing import Any


class BinObj:
    """Pickle file I/O wrapper."""

    @classmethod
    def save(cls, obj: Any, name: str, path: str | Path | None = None, protocol: int = 4) -> None:
        """
        Save object to pickle file.

        Args:
            obj: Python object to save
            name: Base filename (without .pkl extension)
            path: Directory path (defaults to current directory)
            protocol: Pickle protocol version (default: 4)
        """
        if path is None:
            path = "."
        path = Path(path)

        path.mkdir(parents=True, exist_ok=True)
        with open(path / f"{name}.pkl", 'wb') as f:
            pickle.dump(obj, f, protocol)

    @classmethod
    def load(cls, name: str, path: str | Path | None = None) -> Any:
        """
        Load object from pickle file.

        Args:
            name: Base filename (without .pkl extension)
            path: Directory path (defaults to current directory)

        Returns:
            The loaded Python object
        """
        if path is None:
            path = "."
        path = Path(path)

        with open(path / f"{name}.pkl", 'rb') as f:
            return pickle.loads(f.read(), encoding='latin1')

    @classmethod
    def exists(cls, name: str, path: str | Path | None = None) -> bool:
        """
        Check if pickle file exists.

        Args:
            name: Base filename (without .pkl extension)
            path: Directory path (defaults to current directory)

        Returns:
            True if file exists
        """
        if path is None:
            path = "."
        path = Path(path)

        return (path / f"{name}.pkl").is_file()
