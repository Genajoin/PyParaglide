"""Tests for GFS Forecast downloader."""

import datetime as dt

import pytest

from pyparaglide.downloads.gfs_forecast_downloader import GFSForecastDownloader


class TestGFSForecastDownloader:
    """Test GFS Forecast downloader URL building and file naming."""

    def test_nomads_url_construction(self):
        """Test NOMADS URL is built correctly."""
        downloader = GFSForecastDownloader(".")

        url = downloader._build_nomads_url("20250104", 0)

        assert "filter_gfs_0p25.pl" in url
        assert "file=gfs.t00z.pgrb2.0p25.f000" in url
        assert "dir=%2Fgfs.20250104%2F00" in url
        assert "var_TMP=on" in url
        assert "var_UGRD=on" in url
        assert "var_VGRD=on" in url
        assert "lev_700_mb=on" in url
        assert "lev_1000_mb=on" in url

    def test_forecast_filename_format(self):
        """Test forecast file naming matches forecast command expectation."""
        downloader = GFSForecastDownloader(".")

        # forecast command expects f"{date.strftime('%Y%m%d')}-{hour:02d}.grib2"
        # See cli/__init__.py line 970
        date = dt.date(2025, 1, 4)
        hour = 6

        fname = f"{date.strftime('%Y%m%d')}-{hour:02d}.grib2"
        assert fname == "20250104-06.grib2"

        # Verify this matches what the downloader would create
        expected_path = downloader.forecasts_dir / fname
        assert expected_path.name == "20250104-06.grib2"

    def test_forecast_hours_default(self):
        """Test default forecast hours match forecast command expectation."""
        downloader = GFSForecastDownloader(".")

        # forecast command expects [6, 12, 18]
        # See cli/__init__.py line 969
        assert downloader.forecast_hours == [6, 12, 18]

    @pytest.mark.parametrize(
        "date_str,expected", [("2025-01-04", "20250104"), ("2024-12-31", "20241231")]
    )
    def test_date_parsing(self, date_str, expected):
        """Test date string parsing."""
        parsed = dt.datetime.strptime(date_str, "%Y-%m-%d").date()
        forecast_time = parsed.strftime("%Y%m%d")

        assert forecast_time == expected

    def test_forecasts_dir_creation(self, tmp_path):
        """Test that forecasts directory is created."""
        downloader = GFSForecastDownloader(tmp_path)

        assert downloader.forecasts_dir.exists()
        assert downloader.forecasts_dir.name == "forecasts"

    def test_nomads_url_different_hours(self):
        """Test NOMADS URL for different forecast hours."""
        downloader = GFSForecastDownloader(".")

        url_00 = downloader._build_nomads_url("20250104", 0)
        url_06 = downloader._build_nomads_url("20250104", 6)
        url_12 = downloader._build_nomads_url("20250104", 12)
        url_18 = downloader._build_nomads_url("20250104", 18)

        assert "file=gfs.t00z.pgrb2.0p25.f000" in url_00
        assert "dir=%2Fgfs.20250104%2F00" in url_00

        assert "file=gfs.t06z.pgrb2.0p25.f000" in url_06
        assert "dir=%2Fgfs.20250104%2F06" in url_06

        assert "file=gfs.t12z.pgrb2.0p25.f000" in url_12
        assert "dir=%2Fgfs.20250104%2F12" in url_12

        assert "file=gfs.t18z.pgrb2.0p25.f000" in url_18
        assert "dir=%2Fgfs.20250104%2F18" in url_18

    def test_nomads_url_contains_all_required_vars(self):
        """Test that NOMADS URL contains all required weather variables."""
        downloader = GFSForecastDownloader(".")
        url = downloader._build_nomads_url("20250104", 0)

        # Temperature
        assert "var_TMP=on" in url

        # Wind components
        assert "var_UGRD=on" in url
        assert "var_VGRD=on" in url

        # Humidity
        assert "var_RH=on" in url

        # Precipitable water
        assert "var_PWAT=on" in url

        # Cloud water
        assert "var_CWAT=on" in url

        # Geopotential height
        assert "var_HGT=on" in url

        # Vertical velocity
        assert "var_VVEL=on" in url

        # Absolute vorticity
        assert "var_ABSV=on" in url

    def test_nomads_url_contains_all_required_levels(self):
        """Test that NOMADS URL contains all required pressure levels."""
        downloader = GFSForecastDownloader(".")
        url = downloader._build_nomads_url("20250104", 0)

        # Pressure levels
        assert "lev_200_mb=on" in url
        assert "lev_300_mb=on" in url
        assert "lev_400_mb=on" in url
        assert "lev_500_mb=on" in url
        assert "lev_600_mb=on" in url
        assert "lev_700_mb=on" in url
        assert "lev_800_mb=on" in url
        assert "lev_900_mb=on" in url
        assert "lev_1000_mb=on" in url

        # Entire atmosphere
        assert "lev_entire_atmosphere=on" in url
