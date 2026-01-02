"""
Tile coordinate math utilities.

Original source: neural_network/inc/tiles_maths.py

Reference: http://www.maptiler.org/google-maps-coordinates-tile-bounds-projection/
"""

import math
from typing import Dict, Any


class TilesMaths:
    """Coordinate conversion utilities for map tiles."""

    ORIGIN_SHIFT = 2.0 * math.pi * 6378137.0 / 2.0
    INITIAL_RESOLUTION = 2.0 * math.pi * 6378137.0

    @staticmethod
    def Resolution(zoom: int) -> float:
        """Get resolution at given zoom level."""
        return TilesMaths.INITIAL_RESOLUTION / (2.0 ** zoom)

    @staticmethod
    def MetersToPixels(mx: float, my: float, zoom: int) -> tuple[float, float]:
        """Convert meters to pixel coordinates."""
        res = TilesMaths.Resolution(zoom)
        px = (mx + TilesMaths.ORIGIN_SHIFT) / res
        py = (my + TilesMaths.ORIGIN_SHIFT) / res
        return (px, py)

    @staticmethod
    def LatLonToMeters(lat: float, lon: float) -> tuple[float, float]:
        """
        Convert lat/lon in WGS84 Datum to XY in Spherical Mercator EPSG:900913.
        """
        mx = lon * TilesMaths.ORIGIN_SHIFT / 180.0
        my = math.log(math.tan((90.0 + lat) * math.pi / 360.0)) / (math.pi / 180.0)
        my = my * TilesMaths.ORIGIN_SHIFT / 180.0
        return (mx, my)

    @staticmethod
    def LatLonToTileCoords(zoom: int, lat: float, lon: float) -> Dict[str, int]:
        """
        Convert lat/lon to tile coordinates.

        Returns dict with keys: tx, ty, x, y
        - tx, ty: Tile coordinates
        - x, y: Pixel coordinates within tile (0-255)
        """
        meters = TilesMaths.LatLonToMeters(-lat, lon)
        tx_ty = TilesMaths.MetersToPixels(meters[0], meters[1], zoom)

        x = max(0, min(255, int(math.fmod(tx_ty[0], 1) * 256.0)))
        y = max(0, min(255, int(math.fmod(tx_ty[1], 1) * 256.0)))

        return {
            'tx': int(tx_ty[0]),
            'ty': int(tx_ty[1]),
            'x': x,
            'y': y,
        }
