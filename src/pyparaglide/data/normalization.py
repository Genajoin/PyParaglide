"""
Data normalization utilities for weather data.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Normalization:
    """Normalization coefficients for weather data."""

    other_mean: np.ndarray
    other_std: np.ndarray
    humidity_mean: np.ndarray
    humidity_std: np.ndarray

    def save(self, path: Path | str) -> None:
        """Save normalization coefficients to file."""
        import pickle

        with open(path, "wb") as f:
            pickle.dump([self.other_mean, self.other_std, self.humidity_mean, self.humidity_std], f)

    @staticmethod
    def load(path: Path | str) -> "Normalization":
        """Load normalization coefficients from file."""
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)
        return Normalization(other_mean=data[0], other_std=data[1], humidity_mean=data[2], humidity_std=data[3])


def compute_normalization_coeffs(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute mean and std for normalization.

    Args:
        data: Array of shape (n_samples, n_features)

    Returns:
        (mean, std) tuple
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    # Avoid division by zero
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std


def apply_normalization(data: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Apply z-score normalization to data.

    Args:
        data: Array to normalize
        mean: Mean values
        std: Standard deviation values

    Returns:
        Normalized data
    """
    return (data - mean) / std
