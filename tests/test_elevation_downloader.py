"""Tests for SRTM elevation downloader."""

import pytest
from pathlib import Path
from pyparaglide.downloads.elevation_downloader import ElevationDownloader


class TestElevationDownloader:
    """Test CGIAR SRTM tile coordinate calculations."""

    def test_cgiar_indices_alps(self):
        """Alps region (Northern/Eastern hemisphere)."""
        downloader = ElevationDownloader(".", (45, 47, 10, 12))
        # lat_tile = (60 - 45) // 5 = 3, lon_tile = 10 // 5 + 37 = 39
        assert downloader._cgiar_tile_indices(45, 10) == (39, 3)

    def test_cgiar_indices_south_america(self):
        """South America (Southern/Western hemisphere)."""
        downloader = ElevationDownloader(".", (-35, -25, -75, -65))
        # lat_tile = (60 - (-30)) // 5 = 18, lon_tile = -70 // 5 + 37 = 23
        assert downloader._cgiar_tile_indices(-30, -70) == (23, 18)

    def test_cgiar_indices_africa(self):
        """Africa (mixed hemispheres)."""
        downloader = ElevationDownloader(".", (-10, 0, 15, 25))
        # lat_tile = (60 - (-5)) // 5 = 13, lon_tile = 20 // 5 + 37 = 41
        assert downloader._cgiar_tile_indices(-5, 20) == (41, 13)

    def test_cgiar_indices_australia(self):
        """Australia (Southern/Eastern hemisphere)."""
        downloader = ElevationDownloader(".", (-35, -25, 135, 145))
        # lat_tile = (60 - (-30)) // 5 = 18, lon_tile = 140 // 5 + 37 = 65
        assert downloader._cgiar_tile_indices(-30, 140) == (65, 18)

    def test_cgiar_indices_equator(self):
        """Equator (0, 0) baseline."""
        downloader = ElevationDownloader(".", (0, 1, 0, 1))
        # lat_tile = (60 - 0) // 5 = 12, lon_tile = 0 // 5 + 37 = 37
        assert downloader._cgiar_tile_indices(0, 0) == (37, 12)

    def test_cgiar_indices_limits(self):
        """SRTM coverage boundaries."""
        downloader = ElevationDownloader(".", (0, 1, 0, 1))

        # North limit (+60°)
        lon_tile, lat_tile = downloader._cgiar_tile_indices(55, 0)
        assert lat_tile == 1  # (60 - 55) // 5 = 1

        # South limit (-60°)
        lon_tile, lat_tile = downloader._cgiar_tile_indices(-60, 0)
        assert lat_tile == 24  # (60 - (-60)) // 5 = 24

        # West limit (-180°)
        lon_tile, lat_tile = downloader._cgiar_tile_indices(0, -180)
        assert lon_tile == 1  # -180 // 5 + 37 = -36 + 37 = 1

        # East limit (+175°)
        lon_tile, lat_tile = downloader._cgiar_tile_indices(0, 175)
        assert lon_tile == 72  # 175 // 5 + 37 = 35 + 37 = 72

    @pytest.mark.parametrize("lat,lon,expected_lon_tile,expected_lat_tile", [
        (45, 10, 39, 3),       # Alps
        (-30, -70, 23, 18),    # S. America
        (0, 0, 37, 12),        # Equator
        (-5, 20, 41, 13),      # Africa
        (-30, 140, 65, 18),    # Australia
        (30, 80, 53, 6),       # Himalayas
    ])
    def test_cgiar_indices_parametrized(self, lat, lon, expected_lon_tile, expected_lat_tile):
        """Parametrized test for multiple regions."""
        downloader = ElevationDownloader(".", (0, 1, 0, 1))
        lon_tile, lat_tile = downloader._cgiar_tile_indices(lat, lon)
        assert lon_tile == expected_lon_tile, f"Wrong lon_tile for lat={lat}, lon={lon}"
        assert lat_tile == expected_lat_tile, f"Wrong lat_tile for lat={lat}, lon={lon}"
