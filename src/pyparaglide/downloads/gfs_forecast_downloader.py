"""
GFS Forecast data downloader from NOAA NOMADS.

Downloads GFS Forecast GRIB files for generating flyability predictions.
Uses NOMADS filter API for targeted parameter downloads.
"""

import datetime as dt
import urllib.error
import urllib.request
from pathlib import Path
from threading import Lock
from typing import Literal

from tqdm import tqdm


class GFSForecastDownloader:
    """
    Download GFS Forecast data from NOAA NOMADS.

    Downloads forecast GRIB files for specific dates and forecast hours.
    Files are saved with naming convention: YYYYMMDD-HH.grib2
    """

    # NOMADS filter parameters (matching legacy GfsData from neural_network/inc/dataset.py)
    _FORECAST_LEVELS = [
        "lev_200_mb=on",
        "lev_300_mb=on",
        "lev_400_mb=on",
        "lev_500_mb=on",
        "lev_600_mb=on",
        "lev_700_mb=on",
        "lev_800_mb=on",
        "lev_900_mb=on",
        "lev_1000_mb=on",
        "lev_entire_atmosphere=on",
        "lev_entire_atmosphere_%5C%28considered_as_a_single_layer%5C%29=on",
    ]

    _FORECAST_VARS = [
        "var_VVEL=on",  # Vertical velocity
        "var_ABSV=on",  # Absolute vorticity
        "var_CWAT=on",  # Cloud water
        "var_HGT=on",  # Geopotential height
        "var_PWAT=on",  # Precipitable water
        "var_RH=on",  # Relative humidity
        "var_TMP=on",  # Temperature
        "var_UGRD=on",  # U component of wind
        "var_VGRD=on",  # V component of wind
    ]

    GRID = "0p25"  # GFS resolution

    def __init__(
        self,
        data_dir: Path | str,
        max_retries: int = 3,
    ):
        """
        Initialize GFS Forecast downloader.

        Args:
            data_dir: Output directory for forecast GRIB files
            max_retries: Maximum retry attempts
        """
        self.data_dir = Path(data_dir)
        self.max_retries = max_retries
        self.min_file_size = 5 * 1024 * 1024  # 5 MB minimum

        # Create forecasts subdirectory
        self.forecasts_dir = self.data_dir / "forecasts"
        self.forecasts_dir.mkdir(parents=True, exist_ok=True)

        # Initialize tqdm lock for thread safety
        tqdm.set_lock(Lock())

        # Default forecast hours (matches forecast command expectation)
        self.forecast_hours = [6, 12, 18]

    def download_forecast(
        self,
        target_date: dt.date | str,
        forecast_hour: int,
    ) -> dict[Literal["downloaded", "skipped", "failed", "size_mb"], int | float]:
        """
        Download GFS forecast for a specific date and hour.

        Args:
            target_date: Target date (YYYY-MM-DD string or date object)
            forecast_hour: Forecast cycle hour (0, 6, 12, or 18)

        Returns:
            Dictionary with download statistics
        """
        # Parse date
        if isinstance(target_date, str):
            target_date = dt.datetime.strptime(target_date, "%Y-%m-%d").date()

        # Build NOMADS URL
        forecast_time = target_date.strftime("%Y%m%d")
        url = self._build_nomads_url(forecast_time, forecast_hour)

        # Output filename: YYYYMMDD-HH.grib2
        # This matches what forecast command expects (line 970 in cli/__init__.py)
        dest_fname = f"{forecast_time}-{forecast_hour:02d}.grib2"
        dest_path = self.forecasts_dir / dest_fname

        # Check if already exists and is valid
        if dest_path.exists() and dest_path.stat().st_size > self.min_file_size:
            return {"downloaded": 0, "skipped": 1, "failed": 0, "size_mb": 0}

        # Download
        try:
            size_mb = self._download_with_retry(url, dest_path)
            return {"downloaded": 1, "skipped": 0, "failed": 0, "size_mb": size_mb}
        except Exception as e:
            # Remove partial file
            if dest_path.exists():
                dest_path.unlink()
            return {"downloaded": 0, "skipped": 0, "failed": 1, "size_mb": 0}

    def download_day(
        self,
        target_date: dt.date | str,
    ) -> dict[str, int | float]:
        """
        Download all forecast hours for a target date.

        This matches the forecast command's expectation (line 969-970 in cli/__init__.py).

        Args:
            target_date: Target date (YYYY-MM-DD string or date object)

        Returns:
            Dictionary with aggregated statistics
        """
        if isinstance(target_date, str):
            target_date = dt.datetime.strptime(target_date, "%Y-%m-%d").date()

        stats = {"downloaded": 0, "skipped": 0, "failed": 0, "total_mb": 0.0}

        for hour in self.forecast_hours:
            result = self.download_forecast(target_date, hour)
            stats["downloaded"] += result["downloaded"]
            stats["skipped"] += result["skipped"]
            stats["failed"] += result["failed"]
            stats["total_mb"] += result["size_mb"]

        return stats

    def _build_nomads_url(self, forecast_time: str, forecast_hour: int) -> str:
        """
        Build NOMADS filter API URL for GFS forecast download.

        Matches the URL pattern from legacy code (neural_network/forecast.py:129).

        Format:
        https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl?file=gfs.tHHz.pgrb2.0p25.f000&...&dir=%2Fgfs.YYYYMMDD%2Fatmos

        Args:
            forecast_time: Forecast date as YYYYMMDD string
            forecast_hour: Forecast cycle hour (0, 6, 12, 18)

        Returns:
            Complete NOMADS API URL
        """
        # Base URL
        base_url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_{self.GRID}.pl"

        # File parameter: gfs.tHHz.pgrb2.0p25.f000
        # f000 = analysis time (start of forecast cycle)
        file_param = f"file=gfs.t{forecast_hour:02d}z.pgrb2.{self.GRID}.f000"

        # Level and variable parameters (from legacy GfsData)
        levels_param = "&".join(self._FORECAST_LEVELS)
        vars_param = "&".join(self._FORECAST_VARS)

        # Directory parameter: %2Fgfs.YYYYMMDD%2FHH
        # %2F is URL-encoded /
        dir_param = f"dir=%2Fgfs.{forecast_time}%2F{forecast_hour:02d}"

        # Spatial coverage (global)
        region_param = "leftlon=0&rightlon=360&toplat=90&bottomlat=-90"

        # Full URL
        url = f"{base_url}?{file_param}&{levels_param}&{vars_param}&{region_param}&{dir_param}"

        return url

    def _download_with_retry(self, url: str, dest_path: Path) -> float:
        """
        Download with retry logic.

        Args:
            url: URL to download from
            dest_path: Destination file path

        Returns:
            Size in MB

        Raises:
            RuntimeError: If all retry attempts fail
        """
        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Stream download with progress
                with tqdm(
                    desc=f"  {dest_path.name}",
                    unit="B",
                    unit_scale=True,
                    unit_divisor=1024 * 1024,
                ) as pbar:
                    req = urllib.request.Request(url)
                    with urllib.request.urlopen(req, timeout=60) as response:
                        with open(dest_path, "wb") as f:
                            while True:
                                chunk = response.read(131072)  # 128KB chunks
                                if not chunk:
                                    break
                                f.write(chunk)
                                pbar.update(len(chunk))

                # Verify file size
                if dest_path.stat().st_size < self.min_file_size:
                    raise RuntimeError(f"Downloaded file too small: {dest_path.stat().st_size} bytes")

                size_mb = dest_path.stat().st_size / (1024 * 1024)
                return size_mb

            except (urllib.error.URLError, urllib.error.HTTPError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    import time

                    wait_time = 2**attempt
                    time.sleep(wait_time)
                else:
                    raise RuntimeError(f"Failed after {self.max_retries} attempts: {e}") from e
            except Exception:
                raise

        raise RuntimeError(f"Failed after {self.max_retries} attempts") from last_error
