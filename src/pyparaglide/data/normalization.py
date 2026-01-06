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
    thermo_mean: np.ndarray | None = None  # NEW: thermodynamic parameters
    thermo_std: np.ndarray | None = None   # NEW

    def save(self, path: Path | str, suffix: str = "") -> None:
        """Save normalization coefficients to file."""
        import pickle

        # Build data list based on what's available
        if self.thermo_mean is not None:
            data = [
                self.other_mean, self.other_std,
                self.humidity_mean, self.humidity_std,
                self.thermo_mean, self.thermo_std,
            ]
        else:
            data = [
                self.other_mean, self.other_std,
                self.humidity_mean, self.humidity_std,
            ]

        # Add suffix if provided
        save_path = Path(path)
        if suffix:
            stem = save_path.stem.split("_normalization_cells")[0]
            save_path = save_path.parent / f"normalization_cells_{suffix}{save_path.suffix}"

        with open(save_path, "wb") as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path: Path | str) -> "Normalization":
        """Load normalization coefficients from file."""
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)

        # Handle both old (4 elements) and new (6 elements) formats
        if len(data) == 6:
            return Normalization(
                other_mean=data[0], other_std=data[1],
                humidity_mean=data[2], humidity_std=data[3],
                thermo_mean=data[4], thermo_std=data[5],
            )
        else:
            # Legacy format without thermo
            return Normalization(
                other_mean=data[0], other_std=data[1],
                humidity_mean=data[2], humidity_std=data[3],
                thermo_mean=None, thermo_std=None,
            )


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


def wind_uv_to_direction_bins(uv: np.ndarray, n_bins: int = 8) -> np.ndarray:
    """
    Convert U,V wind components to n-direction bins.

    This matches the original implementation in neural_network/inc/utils.py:55-71.

    Args:
        uv: (nb_samples, 2) array with U,V components
        n_bins: Number of direction bins (default 8 for N, NE, E, SE, S, SW, W, NW)

    Returns:
        (nb_samples, n_bins) array with magnitude in the direction bin

    Direction mapping (n_bins=8):
        0 = West (u<0, v=0)
        1 = Southwest
        2 = South (u=0, v<0)
        3 = Southeast
        4 = East (u>0, v=0)
        5 = Northeast
        6 = North (u=0, v>0)
        7 = Northwest
    """
    import math

    nb_samples = uv.shape[0]
    u, v = uv[:, 0], uv[:, 1]

    # Calculate angle: arctan2(v, u) gives angle from East, CCW
    # Map to [0, n_bins) and shift by n_bins//2 to align East with bin 0
    angle = np.mod((np.around(np.arctan2(v, u) / (2.0 * math.pi) * n_bins) + n_bins // 2).astype(int), n_bins)

    # Result array with magnitude in direction bin
    result = np.zeros((nb_samples, n_bins), dtype=np.float32)
    magnitude = np.sqrt(u * u + v * v)
    result[np.arange(nb_samples), angle] = magnitude

    return result
