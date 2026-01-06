"""
GRIB file cache for incremental dataset building.

Caches extracted weather parameter values from GRIB files to avoid
reprocessing when rebuilding datasets with modified date ranges.
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


class GribCache:
    """
    Manages cached GRIB extractions.

    Each GRIB file's extracted data is cached as an .npz file with metadata
    for validation. Cache is organized by date: cache/grib/YYYY/MM/file.npz
    """

    MANIFEST_FILE = "cache_manifest.json"

    def __init__(self, cache_dir: Path):
        """
        Initialize cache manager.

        Args:
            cache_dir: Root cache directory (e.g., data/pkl/cache/grib/)
        """
        self.cache_dir = Path(cache_dir)
        self.manifest_path = self.cache_dir / self.MANIFEST_FILE
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_cache_path(self, grib_path: Path) -> Path:
        """
        Get cache file path for a GRIB file.

        Converts: gfsanl_3_20240601_0600_000.grb2 → 2024/06/gfsanl_3_20240601_0600_000.npz

        Args:
            grib_path: Path to GRIB file (can be full path or just filename)

        Returns:
            Path to cache file (relative to cache_dir)
        """
        filename = Path(grib_path).name
        # Extract date components: gfsanl_3_YYYYMMDD_HH00_000.grb2
        parts = filename.replace('.grb2', '').split('_')
        if len(parts) >= 3:
            date_str = parts[2]  # YYYYMMDD
            year = date_str[:4]
            month = date_str[4:6]
            # Replace .grb2 with .npz
            cache_filename = filename.replace('.grb2', '.npz')
            return Path(year) / month / cache_filename
        # Fallback: store in 'unknown' subdirectory
        return Path("unknown") / filename.replace('.grb2', '.npz')

    def _get_full_cache_path(self, grib_path: Path) -> Path:
        """Get full path to cache file."""
        return self.cache_dir / self.get_cache_path(grib_path)

    def _compute_md5(self, file_path: Path) -> str:
        """
        Compute MD5 hash of a file.

        Args:
            file_path: Path to file

        Returns:
            Hexadecimal MD5 hash string
        """
        md5 = hashlib.md5()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                md5.update(chunk)
        return md5.hexdigest()

    def is_valid(self, grib_path: Path, config: dict) -> bool:
        """
        Check if cached data is valid for current configuration.

        Validates:
        1. Cache file exists
        2. GRIB file exists
        3. bbox matches (detects cell grid changes)
        4. nb_cells matches (detects grid size changes)

        Note: MD5 validation removed to avoid reading 500MB GRIB files.
        NOAA GRIB files don't change after download.

        Args:
            grib_path: Path to source GRIB file (can be str or Path)
            config: Configuration dict with 'bbox', 'cells_latlon', 'nb_cells'

        Returns:
            True if cache is valid and can be used
        """
        # Ensure grib_path is a Path object (may come as str from multiprocessing)
        grib_path = Path(grib_path)
        cache_path = self._get_full_cache_path(grib_path)

        if not cache_path.exists():
            return False

        if not grib_path.exists():
            return False

        try:
            cache_data = np.load(cache_path, allow_pickle=True)

            # Check bbox - handle None case
            cached_bbox = cache_data['bbox']
            current_bbox = config.get('bbox')
            # Handle numpy.array(None) case - convert to None
            if hasattr(cached_bbox, 'dtype') and cached_bbox.dtype == object and cached_bbox.size == 1:
                cached_bbox = cached_bbox.item()  # Extract scalar from array
            # Convert both to tuple for comparison (cached might be numpy array)
            if cached_bbox is not None and hasattr(cached_bbox, '__iter__') and not isinstance(cached_bbox, str):
                cached_bbox = tuple(cached_bbox)
            if current_bbox is not None:
                current_bbox = tuple(current_bbox)
            if cached_bbox != current_bbox:
                return False

            # Check nb_cells
            cached_nb_cells = int(cache_data['nb_cells'])
            current_nb_cells = config.get('nb_cells')
            if cached_nb_cells != current_nb_cells:
                return False

            return True

        except (KeyError, IOError, OSError, ValueError, TypeError):
            return False

    def load(self, grib_path: Path, flatten: bool = True) -> np.ndarray:
        """
        Load cached values.

        Args:
            grib_path: Path to source GRIB file (can be str or Path)
            flatten: If True, return flat array (nb_cells * 69,), else (nb_cells, 69)

        Returns:
            Cached values array - flat if flatten=True, else (nb_cells, 69)

        Raises:
            FileNotFoundError: If cache file doesn't exist
            ValueError: If cache file is corrupted
        """
        grib_path = Path(grib_path)
        cache_path = self._get_full_cache_path(grib_path)

        if not cache_path.exists():
            raise FileNotFoundError(f"Cache not found for {grib_path}")

        try:
            cache_data = np.load(cache_path)
            values = cache_data['values']
            if flatten:
                return values.flatten()
            return values
        except Exception as e:
            raise ValueError(f"Failed to load cache for {grib_path}: {e}")

    def save(
        self,
        grib_path: Path,
        values: np.ndarray,
        config: dict,
        params: list | None = None,
    ) -> None:
        """
        Save extracted values to cache.

        Args:
            grib_path: Path to source GRIB file (can be str or Path)
            values: Extracted parameter values array of shape (nb_cells, 69)
            config: Configuration dict with 'bbox', 'cells_latlon', 'nb_cells'
            params: List of 69 (name, level) tuples used for extraction (optional)
        """
        grib_path = Path(grib_path)
        cache_path = self._get_full_cache_path(grib_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare metadata - only numpy-serializable values
        bbox_val = config.get('bbox')
        # Convert bbox to tuple for storage, or None if not provided
        if bbox_val is not None:
            bbox_val = tuple(bbox_val)

        metadata = {
            'values': values,
            'cells_latlon': np.array(config.get('cells_latlon', []), dtype=np.float32),
            'bbox': bbox_val,
            'nb_cells': config.get('nb_cells', 0),
            'extraction_time': datetime.utcnow().isoformat() + 'Z',
        }

        # Save as compressed NPZ
        np.savez_compressed(cache_path, **metadata)

    def invalidate(self, grib_path: Path) -> bool:
        """
        Remove cached file.

        Args:
            grib_path: Path to source GRIB file (can be str or Path)

        Returns:
            True if file was removed, False if it didn't exist
        """
        grib_path = Path(grib_path)
        cache_path = self._get_full_cache_path(grib_path)

        if cache_path.exists():
            cache_path.unlink()
            return True
        return False

    def clear_all(self) -> int:
        """
        Clear all cached files.

        Returns:
            Number of files removed
        """
        count = 0
        for npz_file in self.cache_dir.rglob("*.npz"):
            npz_file.unlink()
            count += 1

        # Remove manifest if exists
        if self.manifest_path.exists():
            self.manifest_path.unlink()

        return count

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dict with cache statistics
        """
        npz_files = list(self.cache_dir.rglob("*.npz"))
        total_size = sum(f.stat().st_size for f in npz_files)

        return {
            'cache_dir': str(self.cache_dir),
            'cached_files': len(npz_files),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
        }

    def update_manifest(self, grib_path: Path, status: str = "cached") -> None:
        """
        Update cache manifest with a file entry.

        Args:
            grib_path: Path to source GRIB file
            status: Status string ("cached", "skipped", "failed")
        """
        manifest = self._load_manifest()

        cache_key = str(grib_path)
        manifest['entries'][cache_key] = {
            'cache_file': str(self.get_cache_path(grib_path)),
            'status': status,
            'updated': datetime.utcnow().isoformat() + 'Z',
        }

        self._save_manifest(manifest)

    def _load_manifest(self) -> dict:
        """Load or create manifest."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                pass

        # Create new manifest
        return {
            'version': '1.0',
            'created': datetime.utcnow().isoformat() + 'Z',
            'entries': {},
        }

    def _save_manifest(self, manifest: dict) -> None:
        """Save manifest to file."""
        manifest['last_updated'] = datetime.utcnow().isoformat() + 'Z'
        with open(self.manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
