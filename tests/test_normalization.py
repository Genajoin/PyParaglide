"""
Tests for data normalization module.
"""

import numpy as np
import pytest

from pyparaglide.data.normalization import (
    Normalization,
    apply_normalization,
    compute_normalization_coeffs,
)


class TestNormalization:
    """Test normalization functions."""

    def test_compute_normalization_coeffs_basic(self):
        """Test basic normalization coefficient computation."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

        mean, std = compute_normalization_coeffs(data)

        # Check means
        assert mean.shape == (2,)
        np.testing.assert_array_almost_equal(mean, [3.0, 4.0])

        # Check stds (population std)
        expected_std = np.std(data, axis=0, ddof=0)
        np.testing.assert_array_almost_equal(std, expected_std)

    def test_compute_normalization_coeffs_constant(self):
        """Test normalization with constant values."""
        data = np.array([[5.0, 5.0], [5.0, 5.0]], dtype=np.float32)

        mean, std = compute_normalization_coeffs(data)

        np.testing.assert_array_almost_equal(mean, [5.0, 5.0])
        # Std should be small but not zero to avoid division by zero
        assert np.all(std >= 0)

    def test_apply_normalization(self):
        """Test applying normalization to data."""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32)

        mean, std = compute_normalization_coeffs(data)
        normalized = apply_normalization(data, mean, std)

        # Check that normalized data has approx mean=0, std=1
        assert normalized.shape == data.shape
        np.testing.assert_array_almost_equal(np.mean(normalized, axis=0), [0.0, 0.0], decimal=5)
        np.testing.assert_array_almost_equal(np.std(normalized, axis=0, ddof=0), [1.0, 1.0], decimal=5)

    def test_normalization_dataclass(self):
        """Test Normalization dataclass."""
        norm = Normalization(
            other_mean=np.array([1.0, 2.0, 3.0]),
            other_std=np.array([0.5, 1.0, 1.5]),
            humidity_mean=np.array([4.0]),
            humidity_std=np.array([2.0]),
        )

        assert norm.other_mean.shape == (3,)
        assert norm.other_std.shape == (3,)
        assert norm.humidity_mean.shape == (1,)
        assert norm.humidity_std.shape == (1,)

    def test_normalization_save_load(self, tmp_path):
        """Test saving and loading Normalization."""
        import pickle

        norm = Normalization(
            other_mean=np.array([1.0, 2.0]),
            other_std=np.array([0.5, 1.0]),
            humidity_mean=np.array([4.0]),
            humidity_std=np.array([2.0]),
        )

        # Save
        path = tmp_path / "normalization.pkl"
        norm.save(path)
        assert path.exists()

        # Load
        loaded = Normalization.load(path)
        np.testing.assert_array_equal(loaded.other_mean, norm.other_mean)
        np.testing.assert_array_equal(loaded.other_std, norm.other_std)
        np.testing.assert_array_equal(loaded.humidity_mean, norm.humidity_mean)
        np.testing.assert_array_equal(loaded.humidity_std, norm.humidity_std)
