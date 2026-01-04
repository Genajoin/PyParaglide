"""
GeoTIFF elevation data reader.

Reads elevation data from GeoTIFF files (ETOPO 2022, GLOBE, etc.)
Compatible interface with ElevationReader for drop-in replacement.
"""

import math
from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
from rasterio.transform import rowcol


class GeoTiffReader:
    """
    Read elevation data from GeoTIFF files.

    Compatible with ElevationReader interface for use in BuildTerrainPhase.
    Uses rasterio for efficient GeoTIFF reading.
    """

    def __init__(self, elevation_dir: str | Path):
        """
        Initialize GeoTIFF reader.

        Args:
            elevation_dir: Directory containing elevation.tif file
        """
        self.elevation_dir = Path(elevation_dir)
        self.geotiff_path = self.elevation_dir / "elevation.tif"

        if not self.geotiff_path.exists():
            raise FileNotFoundError(f"GeoTIFF not found: {self.geotiff_path}")

        # Open dataset (keep open for efficient reading)
        self.dataset = rasterio.open(self.geotiff_path)

    def close(self) -> None:
        """Close the GeoTIFF dataset."""
        if self.dataset:
            self.dataset.close()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def get_elevation(self, lat: float, lon: float) -> Optional[int]:
        """
        Get elevation at coordinates.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Elevation in meters or None if outside bounds
        """
        try:
            # Transform lat/lon to pixel coordinates
            row, col = rowcol(self.dataset.transform, lon, lat)

            # Check bounds
            if not (0 <= row < self.dataset.height and 0 <= col < self.dataset.width):
                return None

            # Read single pixel
            window = rasterio.windows.Window(col, row, 1, 1)
            data = self.dataset.read(1, window=window)

            if data.size == 0:
                return None

            elevation = int(data[0, 0])

            # Check for nodata value
            if self.dataset.nodata is not None and elevation == self.dataset.nodata:
                return None

            return elevation

        except (IndexError, ValueError):
            return None

    def get_mountainess(
        self, lat: float, lon: float, grid_size: int = 5, radius_km: float = 5.0
    ) -> float:
        """
        Calculate terrain mountainess by sampling elevation around point.

        Uses grid_size x grid_size points in a circle of radius_km.
        Formula: (max_elev - min_elev) / 800, clamped to [0, 1].

        Args:
            lat: Center latitude
            lon: Center longitude
            grid_size: Grid size (5 = 5x5 = 25 points)
            radius_km: Analysis area radius in kilometers

        Returns:
            Value 0.0-1.0 (0 = plain, 1 = mountains)
        """
        # Conversion factors: degrees to kilometers
        # 1° latitude ≈ 111 km everywhere
        # 1° longitude ≈ 111 km * cos(latitude)
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * math.cos(math.radians(lat))

        # Grid step in degrees
        step_lat = (radius_km * 2) / (grid_size - 1) / km_per_deg_lat
        step_lon = (radius_km * 2) / (grid_size - 1) / km_per_deg_lon

        # Start position (top left of grid)
        start_lat = lat - radius_km / km_per_deg_lat
        start_lon = lon - radius_km / km_per_deg_lon

        elevations = []

        # Sample elevation at grid points
        for i in range(grid_size):
            for j in range(grid_size):
                sample_lat = start_lat + i * step_lat
                sample_lon = start_lon + j * step_lon
                elev = self.get_elevation(sample_lat, sample_lon)
                if elev is not None:
                    elevations.append(elev)

        # If couldn't read any elevation
        if not elevations:
            return 0.0

        # Calculate mountainess: (max - min) / 800
        elev_range = max(elevations) - min(elevations)
        mountainess = elev_range / 800.0

        # Clamp to [0, 1]
        return max(0.0, min(1.0, mountainess))
