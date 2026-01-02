"""
Elevation tile reading for takeoff elevation and mountainess calculation.

Analogous to PHP getElevation() function from www/apps/api/get.php.
"""

import math
import struct
from pathlib import Path
from typing import Optional

from pyparaglide.preprocessing.utils.tiles_maths import TilesMaths


class ElevationReader:
    """Reads elevation from elevation tiles by coordinates."""

    def __init__(self, elevation_dir: str | Path | None = None, zoom: int = 7):
        """
        Args:
            elevation_dir: Path to elevation tiles directory
            zoom: Zoom level (default: 7, available: 5, 6, 7)
        """
        if elevation_dir is None:
            project_root = Path(__file__).parent.parent.parent.parent.parent
            elevation_dir = project_root / "tiler" / "_cache" / "elevation"
        self.elevation_dir = Path(elevation_dir)
        self.zoom = zoom

    def get_elevation(self, lat: float, lon: float) -> Optional[int]:
        """
        Returns elevation in meters by coordinates.

        Args:
            lat: Latitude
            lon: Longitude

        Returns:
            Elevation in meters or None if file not found
        """
        # Convert lat/lon to tile coords
        coords = TilesMaths.LatLonToTileCoords(self.zoom, lat, lon)

        # Path to elevation file
        filepath = self.elevation_dir / str(self.zoom) / str(coords['tx']) / f"{coords['ty']}.elev"

        if not filepath.exists():
            return None

        try:
            with open(filepath, 'rb') as f:
                # Read entire file
                content = f.read()

                # Calculate offset (256x256 pixels, 2 bytes per pixel)
                offset = 2 * (coords['x'] * 256 + coords['y'])

                if offset + 2 > len(content):
                    return None

                # Read 2 bytes (big-endian)
                str_val = content[offset:offset+2]
                elevation = (str_val[0] << 8) + str_val[1]

                return elevation
        except (IOError, OSError):
            return None

    def get_mountainess(self, lat: float, lon: float, grid_size: int = 5, radius_km: float = 5.0) -> float:
        """
        Calculates terrain mountainess by elevation around point.

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
