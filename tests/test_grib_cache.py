"""
Unit tests for GRIB file caching.
"""

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import pytest

from pyparaglide.preprocessing.cache import GribCache


@pytest.fixture
def temp_cache_dir():
    """Create a temporary directory for cache tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_grib_file(temp_cache_dir):
    """Create a sample GRIB-like file for testing."""
    grib_file = temp_cache_dir / "test.grb2"
    # Create a small test file with some content
    content = b"fake grib content for testing"
    grib_file.write_bytes(content)
    return grib_file


@pytest.fixture
def sample_values():
    """Create sample weather values array."""
    # Shape: (nb_cells, 65) - 15 cells, 65 parameters
    np.random.seed(42)
    return np.random.rand(15, 65).astype(np.float32)


@pytest.fixture
def sample_config():
    """Create sample configuration."""
    cells_latlon = [(45.0 + i * 0.1, 13.0 + i * 0.1) for i in range(15)]
    return {
        'bbox': (45.0, 47.0, 13.0, 15.0),
        'cells_latlon': cells_latlon,
        'nb_cells': 15,
    }


@pytest.fixture
def sample_params():
    """Create sample GRIB parameters."""
    # Simplified list of 65 parameters
    params = []
    for i in range(65):
        params.append((f"param_{i}", [('level', i)]))
    return params


class TestGribCache:
    """Test suite for GribCache class."""

    def test_init_creates_directory(self, temp_cache_dir):
        """Test that cache directory is created on init."""
        cache_dir = temp_cache_dir / "cache"
        cache = GribCache(cache_dir)
        assert cache_dir.exists()
        assert cache_dir.is_dir()

    def test_get_cache_path(self, temp_cache_dir, sample_grib_file):
        """Test cache path generation from GRIB filename."""
        cache = GribCache(temp_cache_dir)

        # Test standard GRIB filename
        grib_path = Path("gfsanl_3_20240601_0600_000.grb2")
        cache_path = cache.get_cache_path(grib_path)

        assert cache_path == Path("2024/06/gfsanl_3_20240601_0600_000.npz")

    def test_get_cache_path_with_full_path(self, temp_cache_dir, sample_grib_file):
        """Test cache path generation with full GRIB path."""
        cache = GribCache(temp_cache_dir)

        # Test with full path
        grib_path = Path("/some/path/gfsanl_3_20241231_1800_000.grb2")
        cache_path = cache.get_cache_path(grib_path)

        assert cache_path == Path("2024/12/gfsanl_3_20241231_1800_000.npz")

    def test_save_creates_cache_file(self, temp_cache_dir, sample_grib_file, sample_values, sample_config, sample_params):
        """Test that save creates cache file."""
        cache = GribCache(temp_cache_dir)

        cache.save(sample_grib_file, sample_values, sample_config)

        # Check that cache file exists
        expected_path = temp_cache_dir / cache.get_cache_path(sample_grib_file)
        assert expected_path.exists()

    def test_save_load_roundtrip(self, temp_cache_dir, sample_grib_file, sample_values, sample_config, sample_params):
        """Test that saved values can be loaded correctly."""
        cache = GribCache(temp_cache_dir)

        # Save
        cache.save(sample_grib_file, sample_values, sample_config)

        # Load (flatten=True by default)
        loaded_flat = cache.load(sample_grib_file, flatten=True)
        np.testing.assert_array_equal(loaded_flat, sample_values.flatten())

        # Load (flatten=False for 2D array)
        loaded_2d = cache.load(sample_grib_file, flatten=False)
        np.testing.assert_array_equal(loaded_2d, sample_values)

    def test_load_nonexistent_raises_error(self, temp_cache_dir, sample_grib_file):
        """Test that loading non-existent cache raises error."""
        cache = GribCache(temp_cache_dir)

        with pytest.raises(FileNotFoundError):
            cache.load(sample_grib_file)

    def test_is_valid_with_missing_cache(self, temp_cache_dir, sample_grib_file, sample_config):
        """Test that is_valid returns False for missing cache."""
        cache = GribCache(temp_cache_dir)

        assert not cache.is_valid(sample_grib_file, sample_config)

    def test_is_valid_with_valid_cache(self, temp_cache_dir, sample_grib_file, sample_values, sample_config, sample_params):
        """Test that is_valid returns True for valid cache."""
        cache = GribCache(temp_cache_dir)

        # Save cache
        cache.save(sample_grib_file, sample_values, sample_config)

        # Verify it's valid
        assert cache.is_valid(sample_grib_file, sample_config)

    def test_is_valid_invalidate_on_nb_cells_mismatch(self, temp_cache_dir, sample_grib_file, sample_values, sample_config, sample_params):
        """Test that is_valid returns False when nb_cells changes."""
        cache = GribCache(temp_cache_dir)

        # Save cache
        cache.save(sample_grib_file, sample_values, sample_config)

        # Change nb_cells in config
        different_config = sample_config.copy()
        different_config['nb_cells'] = 999

        # Verify cache is invalid
        assert not cache.is_valid(sample_grib_file, different_config)

    def test_invalidate_removes_cache(self, temp_cache_dir, sample_grib_file, sample_values, sample_config, sample_params):
        """Test that invalidate removes cache file."""
        cache = GribCache(temp_cache_dir)

        # Save cache
        cache.save(sample_grib_file, sample_values, sample_config)
        cache_path = temp_cache_dir / cache.get_cache_path(sample_grib_file)
        assert cache_path.exists()

        # Invalidate
        result = cache.invalidate(sample_grib_file)

        # Verify removed
        assert result is True
        assert not cache_path.exists()

    def test_invalidate_nonexistent_returns_false(self, temp_cache_dir, sample_grib_file):
        """Test that invalidating non-existent cache returns False."""
        cache = GribCache(temp_cache_dir)

        result = cache.invalidate(sample_grib_file)
        assert result is False

    def test_clear_all_removes_all_cache(self, temp_cache_dir, sample_grib_file, sample_values, sample_config, sample_params):
        """Test that clear_all removes all cached files."""
        cache = GribCache(temp_cache_dir)

        # Create multiple cache files
        for i in range(5):
            grib_file = temp_cache_dir / f"test_{i}.grb2"
            grib_file.write_bytes(b"content")
            cache.save(grib_file, sample_values, sample_config)

        # Clear all
        count = cache.clear_all()

        # Verify all removed
        assert count >= 5
        cache_files = list(temp_cache_dir.rglob("*.npz"))
        assert len(cache_files) == 0

    def test_get_stats(self, temp_cache_dir, sample_grib_file, sample_values, sample_config, sample_params):
        """Test that get_stats returns correct statistics."""
        cache = GribCache(temp_cache_dir)

        # Save cache
        cache.save(sample_grib_file, sample_values, sample_config)

        # Get stats
        stats = cache.get_stats()

        assert 'cache_dir' in stats
        assert 'cached_files' in stats
        assert 'total_size_bytes' in stats
        assert 'total_size_mb' in stats
        assert stats['cached_files'] >= 1
        assert stats['total_size_bytes'] > 0

    def test_manifest_operations(self, temp_cache_dir, sample_grib_file):
        """Test manifest save/load operations."""
        cache = GribCache(temp_cache_dir)

        # Update manifest
        cache.update_manifest(sample_grib_file, "cached")

        # Verify manifest exists
        assert cache.manifest_path.exists()

        # Load manifest
        manifest = cache._load_manifest()

        assert manifest['version'] == '1.0'
        assert 'entries' in manifest
        assert str(sample_grib_file) in manifest['entries']


@pytest.mark.integration
class TestGribCacheIntegration:
    """Integration tests for GribCache with realistic scenarios."""

    def test_cache_multiple_hours_same_day(self, temp_cache_dir, sample_config, sample_params):
        """Test caching multiple hours for the same day."""
        cache = GribCache(temp_cache_dir)

        # Simulate 3 hours for one day
        for hour in [6, 12, 18]:
            grib_file = temp_cache_dir / f"gfsanl_3_20240601_{hour:02d}00_000.grb2"
            grib_file.write_bytes(b"grib content")
            values = np.random.rand(15, 65).astype(np.float32)

            cache.save(grib_file, values, sample_config)

        # Verify all cached
        stats = cache.get_stats()
        assert stats['cached_files'] == 3

    def test_cache_hit_scenario(self, temp_cache_dir, sample_grib_file, sample_values, sample_config, sample_params):
        """Test typical cache hit scenario."""
        cache = GribCache(temp_cache_dir)

        # First run: cache miss, save to cache
        assert not cache.is_valid(sample_grib_file, sample_config)
        cache.save(sample_grib_file, sample_values, sample_config)

        # Second run: cache hit
        assert cache.is_valid(sample_grib_file, sample_config)
        loaded = cache.load(sample_grib_file, flatten=False)
        np.testing.assert_array_equal(loaded, sample_values)


class TestGribCachePerCell:
    """Test suite for per-cell caching functionality."""

    def test_get_cell_cache_path(self, temp_cache_dir):
        """Test per-cell cache path generation."""
        cache = GribCache(temp_cache_dir)

        grib_path = Path("gfsanl_3_20240601_0600_000.grb2")
        cell_path = cache.get_cell_cache_path(grib_path, 45.0, 13.0)

        # Should be: cells/45_13/2024/06/gfsanl_3_20240601_0600_000.npz
        expected = Path("cells/45_13/2024/06/gfsanl_3_20240601_0600_000.npz")
        assert cell_path == expected

    def test_get_cell_cache_path_negative_coords(self, temp_cache_dir):
        """Test per-cell cache path with negative longitude."""
        cache = GribCache(temp_cache_dir)

        grib_path = Path("gfsanl_3_20240601_0600_000.grb2")
        cell_path = cache.get_cell_cache_path(grib_path, -10.0, -122.0)

        # Should handle negative coordinates
        expected = Path("cells/-10_-122/2024/06/gfsanl_3_20240601_0600_000.npz")
        assert cell_path == expected

    def test_save_load_cell_roundtrip(self, temp_cache_dir, sample_grib_file):
        """Test that per-cell saved values can be loaded correctly."""
        cache = GribCache(temp_cache_dir)

        # Create cell values (shape: 1, 69)
        np.random.seed(42)
        cell_values = np.random.rand(1, 69).astype(np.float32)

        # Save per cell
        cache.save_cell(sample_grib_file, 45.0, 13.0, cell_values)

        # Load per cell
        loaded = cache.load_cell(sample_grib_file, 45.0, 13.0)

        np.testing.assert_array_equal(loaded, cell_values)

    def test_load_nonexistent_cell_raises_error(self, temp_cache_dir, sample_grib_file):
        """Test that loading non-existent per-cell cache raises error."""
        cache = GribCache(temp_cache_dir)

        with pytest.raises(FileNotFoundError):
            cache.load_cell(sample_grib_file, 45.0, 13.0)

    def test_has_cell_cache(self, temp_cache_dir, sample_grib_file):
        """Test checking which cells have cache."""
        cache = GribCache(temp_cache_dir)

        cells_latlon = [(45.0, 13.0), (46.0, 13.0), (45.0, 14.0)]

        # Initially, no cells cached
        cached, missing = cache.has_cell_cache(sample_grib_file, cells_latlon)
        assert len(cached) == 0
        assert len(missing) == 3

        # Cache one cell
        cell_values = np.random.rand(1, 69).astype(np.float32)
        cache.save_cell(sample_grib_file, 45.0, 13.0, cell_values)

        # Now one cell cached
        cached, missing = cache.has_cell_cache(sample_grib_file, cells_latlon)
        assert (45.0, 13.0) in cached
        assert len(cached) == 1
        assert len(missing) == 2

    def test_is_valid_cell(self, temp_cache_dir, sample_grib_file):
        """Test per-cell cache validation."""
        cache = GribCache(temp_cache_dir)

        # Initially not valid
        assert not cache.is_valid_cell(sample_grib_file, 45.0, 13.0)

        # Save cell
        cell_values = np.random.rand(1, 69).astype(np.float32)
        cache.save_cell(sample_grib_file, 45.0, 13.0, cell_values)

        # Now valid
        assert cache.is_valid_cell(sample_grib_file, 45.0, 13.0)

    def test_is_valid_cell_rejects_wrong_shape(self, temp_cache_dir, sample_grib_file):
        """Test that is_valid_cell rejects wrong shape arrays."""
        from datetime import datetime

        cache = GribCache(temp_cache_dir)

        # Save with wrong shape
        wrong_values = np.random.rand(5, 10).astype(np.float32)

        # Manually create cache file with wrong shape
        cache_path = cache._get_full_cell_cache_path(sample_grib_file, 45.0, 13.0)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            cache_path,
            values=wrong_values,
            cell_lat=45.0,
            cell_lon=13.0,
            extraction_time=datetime.utcnow().isoformat() + 'Z',
        )

        # Should be invalid due to wrong shape
        assert not cache.is_valid_cell(sample_grib_file, 45.0, 13.0)

    def test_load_cells_batch(self, temp_cache_dir, sample_grib_file):
        """Test loading multiple cells in batch."""
        cache = GribCache(temp_cache_dir)

        cells_latlon = [(45.0, 13.0), (46.0, 13.0), (45.0, 14.0)]

        # Save each cell
        np.random.seed(42)
        for cell_lat, cell_lon in cells_latlon:
            cell_values = np.random.rand(1, 69).astype(np.float32)
            cache.save_cell(sample_grib_file, cell_lat, cell_lon, cell_values)

        # Load batch (flattened)
        loaded_flat = cache.load_cells_batch(sample_grib_file, cells_latlon, flatten=True)
        assert loaded_flat.shape == (3 * 69,)  # 3 cells * 69 params

        # Load batch (not flattened)
        loaded_2d = cache.load_cells_batch(sample_grib_file, cells_latlon, flatten=False)
        assert loaded_2d.shape == (3, 69)

    def test_load_cells_batch_partial_cache(self, temp_cache_dir, sample_grib_file):
        """Test loading batch when only some cells are cached."""
        cache = GribCache(temp_cache_dir)

        cells_latlon = [(45.0, 13.0), (46.0, 13.0), (45.0, 14.0)]

        # Only cache first cell
        cell_values = np.random.rand(1, 69).astype(np.float32)
        cache.save_cell(sample_grib_file, 45.0, 13.0, cell_values)

        # Should only load the cached cell
        loaded = cache.load_cells_batch(sample_grib_file, cells_latlon, flatten=False)
        assert loaded.shape == (1, 69)

    def test_bbox_expansion_reuse(self, temp_cache_dir, sample_config):
        """Test that per-cell cache is reused when bbox expands."""
        cache = GribCache(temp_cache_dir)

        # Original bbox: 45-46, 13-14 (4 cells)
        original_cells = [(45.0, 13.0), (45.0, 14.0), (46.0, 13.0), (46.0, 14.0)]

        # Simulate caching for original bbox
        for hour in [6, 12, 18]:
            grib_file = temp_cache_dir / f"gfsanl_3_20240601_{hour:02d}00_000.grb2"
            grib_file.write_bytes(b"grib content")

            for cell_lat, cell_lon in original_cells:
                cell_values = np.random.rand(1, 69).astype(np.float32)
                cache.save_cell(grib_file, cell_lat, cell_lon, cell_values)

        # Expanded bbox: 45-47, 13-14 (6 cells, added 2 new)
        expanded_cells = [(45.0, 13.0), (45.0, 14.0), (46.0, 13.0), (46.0, 14.0), (47.0, 13.0), (47.0, 14.0)]

        # Check which cells are cached
        grib_file = temp_cache_dir / "gfsanl_3_20240601_0600_000.grb2"
        cached, missing = cache.has_cell_cache(grib_file, expanded_cells)

        # Original 4 cells should be cached
        assert len(cached) == 4
        for cell in original_cells:
            assert cell in cached

        # New 2 cells should be missing
        assert len(missing) == 2
        assert (47.0, 13.0) in missing
        assert (47.0, 14.0) in missing

    def test_dual_format_compatibility(self, temp_cache_dir, sample_grib_file, sample_values, sample_config):
        """Test that legacy and per-cell formats can coexist."""
        cache = GribCache(temp_cache_dir)

        # Save in legacy format
        cache.save(sample_grib_file, sample_values, sample_config)

        # Save in per-cell format
        for cell_lat, cell_lon in sample_config['cells_latlon']:
            cell_idx = sample_config['cells_latlon'].index((cell_lat, cell_lon))
            cell_values = sample_values[cell_idx:cell_idx+1, :]
            cache.save_cell(sample_grib_file, cell_lat, cell_lon, cell_values)

        # Both should exist
        legacy_path = cache._get_full_cache_path(sample_grib_file)
        assert legacy_path.exists()

        # Check per-cell cache
        cached, missing = cache.has_cell_cache(sample_grib_file, sample_config['cells_latlon'])
        assert len(cached) == len(sample_config['cells_latlon'])
        assert len(missing) == 0
