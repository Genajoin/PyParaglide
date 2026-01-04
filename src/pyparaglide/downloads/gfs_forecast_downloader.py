"""
GFS Forecast data downloader from NOAA NOMADS.

Downloads GFS Forecast GRIB files for generating flyability predictions.
Uses NOMADS filter API for targeted parameter downloads.
"""

import datetime as dt
import re
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
        force: bool = False,
    ) -> dict[Literal["downloaded", "skipped", "failed", "size_mb"], int | float]:
        """
        Download GFS forecast for a specific date and hour.

        For future dates, automatically finds the best available forecast run
        and uses the correct forecast offset (f006, f012, etc.).

        Args:
            target_date: Target date (YYYY-MM-DD string or date object)
            forecast_hour: Forecast hour (6, 12, or 18)
            force: Force re-download even if file exists

        Returns:
            Dictionary with download statistics
        """
        # Parse date
        if isinstance(target_date, str):
            target_date = dt.datetime.strptime(target_date, "%Y-%m-%d").date()

        target_datetime = dt.datetime.combine(target_date, dt.time(forecast_hour, 0))

        # Output filename: YYYYMMDD-HH.grib2 (matches forecast command expectation)
        dest_fname = f"{target_date.strftime('%Y%m%d')}-{forecast_hour:02d}.grib2"
        dest_path = self.forecasts_dir / dest_fname

        # Check if already exists and is valid
        if not force and dest_path.exists() and dest_path.stat().st_size > self.min_file_size:
            return {"downloaded": 0, "skipped": 1, "failed": 0, "size_mb": 0}

        # Get available forecast runs from NOMADS
        available_runs = self._get_available_forecast_runs(limit=8)

        # Find the best forecast run for this target time
        last_error = None

        for run_str in available_runs:
            # Parse run time: YYYYMMDDHH
            run_datetime = dt.datetime.strptime(run_str, "%Y%m%d%H")

            # Calculate offset from run time to target time
            offset_hours = int((target_datetime - run_datetime).total_seconds() / 3600)

            # Check if this run can provide forecast for target time
            # GFS forecasts go up to 16 days = 384 hours
            if 0 <= offset_hours <= 384:
                try:
                    url, _ = self._build_nomads_url_with_offset(run_str, offset_hours)
                    size_mb = self._download_with_retry(url, dest_path)
                    return {"downloaded": 1, "skipped": 0, "failed": 0, "size_mb": size_mb}
                except Exception as e:
                    last_error = e
                    # Try next available run
                    continue

        # All attempts failed
        if dest_path.exists():
            dest_path.unlink()

        error_msg = f"Failed to download forecast for {target_date} {forecast_hour:02d}Z"
        if last_error:
            error_msg += f": {last_error}"
        return {"downloaded": 0, "skipped": 0, "failed": 1, "size_mb": 0}

    def download_day(
        self,
        target_date: dt.date | str,
        force: bool = False,
    ) -> dict[str, int | float]:
        """
        Download all forecast hours for a target date.

        This matches the forecast command's expectation (line 969-970 in cli/__init__.py).

        Args:
            target_date: Target date (YYYY-MM-DD string or date object)
            force: Force re-download even if file exists

        Returns:
            Dictionary with aggregated statistics
        """
        if isinstance(target_date, str):
            target_date = dt.datetime.strptime(target_date, "%Y-%m-%d").date()

        stats = {"downloaded": 0, "skipped": 0, "failed": 0, "total_mb": 0.0}

        for hour in self.forecast_hours:
            result = self.download_forecast(target_date, hour, force=force)
            stats["downloaded"] += result["downloaded"]
            stats["skipped"] += result["skipped"]
            stats["failed"] += result["failed"]
            stats["total_mb"] += result["size_mb"]

        return stats

    def download_days_ahead(
        self,
        days: int = 10,
        run_date: dt.date | str | None = None,
        force: bool = False,
    ) -> dict[str, int | float]:
        """
        Download forecast for N days ahead from latest (or specified) forecast run.

        This matches the legacy forecast.py behavior which downloads for days=0..9
        from the last available forecast run.

        Args:
            days: Number of days to download (default 10)
            run_date: Specific forecast run date to use (auto-detect if None)
            force: Force re-download even if file exists

        Returns:
            Dictionary with aggregated statistics
        """
        # Get available forecast runs
        available_runs = self._get_available_forecast_runs(limit=4)

        if run_date is None:
            # Use the most recent run
            run_str = available_runs[0]
        else:
            # Find the specified run
            if isinstance(run_date, str):
                run_date = dt.datetime.strptime(run_date, "%Y-%m-%d").date()

            run_str = None
            for rs in available_runs:
                run_dt = dt.datetime.strptime(rs, "%Y%m%d%H").date()
                if run_dt == run_date:
                    run_str = rs
                    break

            if run_str is None:
                raise ValueError(f"Forecast run for {run_date} not found in available runs")

        # Parse run datetime
        run_datetime = dt.datetime.strptime(run_str, "%Y%m%d%H")

        stats = {"downloaded": 0, "skipped": 0, "failed": 0, "total_mb": 0.0}

        # Download for each day and hour
        for day_offset in range(days):
            target_date = run_datetime.date() + dt.timedelta(days=day_offset)

            for target_hour in self.forecast_hours:
                target_datetime = dt.datetime.combine(target_date, dt.time(target_hour, 0))

                # Calculate forecast offset
                offset_hours = int((target_datetime - run_datetime).total_seconds() / 3600)

                # Skip if offset is negative (target is before run)
                if offset_hours < 0:
                    continue

                # Skip if beyond GFS forecast range
                if offset_hours > 384:
                    continue

                # Output filename
                dest_fname = f"{target_date.strftime('%Y%m%d')}-{target_hour:02d}.grib2"
                dest_path = self.forecasts_dir / dest_fname

                # Check if already exists
                if not force and dest_path.exists() and dest_path.stat().st_size > self.min_file_size:
                    stats["skipped"] += 1
                    continue

                # Download
                try:
                    url, _ = self._build_nomads_url_with_offset(run_str, offset_hours)
                    size_mb = self._download_with_retry(url, dest_path)
                    stats["downloaded"] += 1
                    stats["total_mb"] += size_mb
                except Exception:
                    if dest_path.exists():
                        dest_path.unlink()
                    stats["failed"] += 1

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

    def _get_available_forecast_runs(self, limit: int = 4) -> list[str]:
        """
        Scrape NOMADS to find last available forecast runs.

        Matches legacy behavior from neural_network/forecast.py:108-123.

        Args:
            limit: Maximum number of runs to return (default 4)

        Returns:
            List of forecast run strings like ["2026010418", "2026010412", ...]
            Format: YYYYMMDDHH (date + hour of forecast run)
        """
        base_url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_{self.GRID}.pl"

        try:
            # Get list of available forecast dates
            with urllib.request.urlopen(base_url, timeout=30) as response:
                html = response.read().decode('utf-8')

            # Parse forecast dates from HTML links
            # Pattern: ">gfs.YYYYMMDD</a>"
            forecast_dates = re.findall(r'">gfs\.([0-9]+)</a>', html)

            if not forecast_dates:
                raise RuntimeError(f"No forecast dates found on NOMADS")

            runs: list[str] = []

            # Check last 2 dates for available hours
            for date_str in forecast_dates[:2]:
                dir_url = f"{base_url}?dir=%2Fgfs.{date_str}"
                try:
                    with urllib.request.urlopen(dir_url, timeout=30) as response:
                        html2 = response.read().decode('utf-8')

                    # Parse forecast hours from directory listing
                    # Pattern: "gfs.YYYYMMDD%2FHH"
                    hours = re.findall(f'gfs\\.{date_str}%2F([0-9]{{2}})', html2)

                    # Combine date + hour
                    for hour in hours:
                        runs.append(f"{date_str}{hour}")

                except urllib.error.URLError:
                    continue

            if not runs:
                raise RuntimeError(f"No forecast runs found on NOMADS")

            return runs[:limit]

        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to fetch available forecast runs: {e}") from e

    def _build_nomads_url_with_offset(
        self,
        run_time_str: str,
        forecast_offset: int,
    ) -> tuple[str, dt.datetime]:
        """
        Build NOMADS URL with forecast offset (f000, f006, f012, etc.).

        Args:
            run_time_str: Forecast run time as YYYYMMDDHH string
            forecast_offset: Hours into the forecast (0, 6, 12, ..., 384)

        Returns:
            Tuple of (complete URL, forecast run datetime)

        Example:
            run_time_str="2026010418", offset=12
            -> URL with f012, run_datetime=2026-01-04 18:00
        """
        # Parse run time: YYYYMMDDHH
        run_date = dt.datetime.strptime(run_time_str, "%Y%m%d%H")

        # File parameter: gfs.tHHz.pgrb2.0p25.fOFFSET
        file_param = f"file=gfs.t{run_date.hour:02d}z.pgrb2.{self.GRID}.f{forecast_offset:03d}"

        # Level and variable parameters
        levels_param = "&".join(self._FORECAST_LEVELS)
        vars_param = "&".join(self._FORECAST_VARS)

        # Directory parameter: %2Fgfs.YYYYMMDD%2FHH%2Fatmos
        # Format: /gfs.20260104/12/atmos (date + hour + atmos)
        dir_param = f"dir=%2Fgfs.{run_date.strftime('%Y%m%d')}%2F{run_date.hour:02d}%2Fatmos"

        # Spatial coverage (global)
        region_param = "leftlon=0&rightlon=360&toplat=90&bottomlat=-90"

        # Full URL
        base_url = f"https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_{self.GRID}.pl"
        url = f"{base_url}?{file_param}&{levels_param}&{vars_param}&{region_param}&{dir_param}"

        return url, run_date

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
