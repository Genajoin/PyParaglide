"""
Tests for data normalization module.
"""

import numpy as np
import pytest

from pyparaglide.data.normalization import (
    Normalization,
    apply_normalization,
    compute_normalization_coeffs,
    wind_uv_to_direction_bins,
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


class TestWindDirectionEncoding:
    """Test wind direction encoding functions."""

    def test_wind_uv_to_direction_bins_east(self):
        """Test East wind (u>0, v=0) maps to bin 4 (per original formula)."""
        # East wind: u=10, v=0
        uv = np.array([[10.0, 0.0]], dtype=np.float32)
        result = wind_uv_to_direction_bins(uv, n_bins=8)

        # Magnitude should be in bin 4 (East per original formula)
        assert result.shape == (1, 8)
        np.testing.assert_array_almost_equal(result[0, 4], 10.0)  # magnitude in bin 4
        result[0, 4] = 0
        np.testing.assert_array_almost_equal(result[0], np.zeros(8))  # other bins zero

    def test_wind_uv_to_direction_bins_north(self):
        """Test North wind (u=0, v>0) maps to bin 6 (per original formula)."""
        # North wind: u=0, v=10
        uv = np.array([[0.0, 10.0]], dtype=np.float32)
        result = wind_uv_to_direction_bins(uv, n_bins=8)

        # Magnitude should be in bin 6 (North per original formula)
        np.testing.assert_array_almost_equal(result[0, 6], 10.0)
        result[0, 6] = 0
        np.testing.assert_array_almost_equal(result[0], np.zeros(8))

    def test_wind_uv_to_direction_bins_west(self):
        """Test West wind (u<0, v=0) maps to bin 0 (per original formula)."""
        # West wind: u=-10, v=0
        uv = np.array([[-10.0, 0.0]], dtype=np.float32)
        result = wind_uv_to_direction_bins(uv, n_bins=8)

        # Magnitude should be in bin 0 (West per original formula)
        np.testing.assert_array_almost_equal(result[0, 0], 10.0)
        result[0, 0] = 0
        np.testing.assert_array_almost_equal(result[0], np.zeros(8))

    def test_wind_uv_to_direction_bins_south(self):
        """Test South wind (u=0, v<0) maps to bin 2 (per original formula)."""
        # South wind: u=0, v=-10
        uv = np.array([[0.0, -10.0]], dtype=np.float32)
        result = wind_uv_to_direction_bins(uv, n_bins=8)

        # Magnitude should be in bin 2 (South per original formula)
        np.testing.assert_array_almost_equal(result[0, 2], 10.0)
        result[0, 2] = 0
        np.testing.assert_array_almost_equal(result[0], np.zeros(8))

    def test_wind_uv_to_direction_bins_diagonal(self):
        """Test diagonal wind directions."""
        # Northeast wind: u=10, v=10
        uv = np.array([[10.0, 10.0]], dtype=np.float32)
        result = wind_uv_to_direction_bins(uv, n_bins=8)

        # Should map to bin 5 (Northeast per original formula)
        magnitude = 10.0 * np.sqrt(2)
        assert result[0, 5] > magnitude * 0.9  # Allow small rounding error
        assert result[0, 5] < magnitude * 1.1
        np.testing.assert_array_almost_equal(np.sum(result), magnitude, decimal=2)

    def test_wind_uv_to_direction_bins_multiple_samples(self):
        """Test multiple wind vectors at once."""
        uv = np.array([
            [10.0, 0.0],   # East -> bin 4
            [0.0, 10.0],   # North -> bin 6
            [-10.0, 0.0],  # West -> bin 0
        ], dtype=np.float32)
        result = wind_uv_to_direction_bins(uv, n_bins=8)

        assert result.shape == (3, 8)
        np.testing.assert_array_almost_equal(result[0, 4], 10.0)  # East
        np.testing.assert_array_almost_equal(result[1, 6], 10.0)  # North
        np.testing.assert_array_almost_equal(result[2, 0], 10.0)  # West

    def test_wind_uv_to_direction_bins_zero_wind(self):
        """Test zero wind (calm)."""
        uv = np.array([[0.0, 0.0]], dtype=np.float32)
        result = wind_uv_to_direction_bins(uv, n_bins=8)

        # All bins should be zero
        np.testing.assert_array_almost_equal(result[0], np.zeros(8))

    def test_convert_wind_matrix(self):
        """Test the full convert_wind_matrix function from dataset.py."""
        from pyparaglide.data.dataset import convert_wind_matrix

        # Create wind matrix with all 5 altitudes
        # Format: U0, V0, U1, V1, U2, V2, U3, V3, U4, V4
        wind_matrix = np.array([
            [10.0, 0.0, 0.0, 10.0, -5.0, 0.0, 5.0, 5.0, 0.0, -10.0],  # 5 altitudes
            [-5.0, 0.0, 5.0, 5.0, 10.0, 0.0, 0.0, -10.0, 0.0, 10.0],
        ], dtype=np.float32)

        result = convert_wind_matrix(wind_matrix, wind_dim=8)

        # Shape should be (2 samples, 5 altitudes * 8 bins) = (2, 40)
        assert result.shape == (2, 40)

        # Check first sample, first altitude (East wind -> bin 4)
        np.testing.assert_array_almost_equal(result[0, 4], 10.0)
        np.testing.assert_array_almost_equal(result[0, :4], 0.0)
        np.testing.assert_array_almost_equal(result[0, 5:8], 0.0)

        # Check first sample, second altitude (North wind -> bin 6)
        np.testing.assert_array_almost_equal(result[0, 8 + 6], 10.0)
