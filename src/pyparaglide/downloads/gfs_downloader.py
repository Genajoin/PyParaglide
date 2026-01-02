"""
GFS data downloader from NOAA.

Downloads GFS Analysis GRIB files for specified date ranges.
"""

import datetime as dt
import os
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dateutil.rrule import rrule, DAILY
from pathlib import Path
from threading import Lock
from typing import Literal

import pygrib
from tqdm import tqdm


class GFSDownloader:
    """
    Download GFS Analysis data from NOAA.

    Supports parallel downloads, resume, and GRIB filtering.
    """

    # Required parameters for filtering
    REQUIRED_PARAMS = {
        "Temperature": ["isobaricInhPa"],
        "U component of wind": ["isobaricInhPa"],
        "V component of wind": ["isobaricInhPa"],
        "Relative humidity": ["isobaricInhPa"],
        "Geopotential Height": ["isobaricInhPa"],
        "Vertical velocity": ["isobaricInhPa"],
        "Absolute vorticity": ["isobaricInhPa"],
        "Precipitable water": ["entireAtmosphere", "unknown"],
        "Cloud water": ["entireAtmosphere", "unknown"],
    }

    def __init__(
        self,
        data_dir: Path | str,
        hours: list[int] | None = None,
        max_retries: int = 3,
        workers: int = 1,
        filter_grib: bool = False,
    ):
        """
        Initialize GFS downloader.

        Args:
            data_dir: Output directory for GRIB files
            hours: UTC hours to download (default: [0, 6, 12, 18])
            max_retries: Maximum retry attempts
            workers: Number of parallel download workers
            filter_grib: Filter GRIB files after download to reduce size
        """
        self.data_dir = Path(data_dir)
        self.hours = hours or [0, 6, 12, 18]
        self.max_retries = max_retries
        self.workers = workers
        self.filter_grib = filter_grib
        self.min_file_size = 10 * 1024 * 1024  # 10 MB

    def download_range(
        self,
        start_date: dt.date | str,
        end_date: dt.date | str,
    ) -> dict[Literal["downloaded", "skipped", "failed", "total_mb"], int]:
        """
        Download GFS data for a date range.

        Args:
            start_date: Start date (YYYY-MM-DD string or date object)
            end_date: End date (YYYY-MM-DD string or date object)

        Returns:
            Dictionary with statistics
        """
        # Parse dates
        if isinstance(start_date, str):
            start_date = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
        if isinstance(end_date, str):
            end_date = dt.datetime.strptime(end_date, "%Y-%m-%d").date()

        # Generate list of days
        days = list(rrule(DAILY, dtstart=start_date, until=end_date))
        total_tasks = len(days) * len(self.hours)

        print(f"Downloading GFS Analysis data:")
        print(f"  Date range: {start_date} to {end_date}")
        print(f"  Hours: {self.hours}")
        print(f"  Output: {self.data_dir}")
        print(f"  Total files: {total_tasks}")
        print(f"  Workers: {self.workers}")
        if self.filter_grib:
            print(f"  Filtering: ENABLED (~50%% size reduction)")
        print()

        # Thread-safe statistics
        stats = {"downloaded": 0, "skipped": 0, "failed": 0, "total_mb": 0.0}
        stats_lock = Lock()

        # Generate task list
        tasks = []
        task_id = 0
        for day in days:
            for hour in self.hours:
                tasks.append((task_id, day, hour))
                task_id += 1

        def update_stats(result: dict) -> None:
            with stats_lock:
                if result["status"] == "downloaded":
                    stats["downloaded"] += 1
                    stats["total_mb"] += result["size_mb"]
                elif result["status"] == "skipped":
                    stats["skipped"] += 1
                else:
                    stats["failed"] += 1

        try:
            if self.workers == 1:
                # Sequential mode
                with tqdm(total=total_tasks, desc="Files", unit="file") as pbar:
                    for task_id, day, hour in tasks:
                        pbar.set_postfix_str(f"{day.strftime('%Y-%m-%d')} {hour:02d}:00")
                        result = self._download_file(day, hour)
                        update_stats(result)
                        pbar.update(1)
            else:
                # Parallel mode
                with tqdm(total=total_tasks, desc="Files", unit="file", position=0, leave=True) as main_pbar:
                    with ThreadPoolExecutor(max_workers=self.workers) as executor:
                        futures = {}
                        for task_id, day, hour in tasks:
                            future = executor.submit(self._download_file, day, hour)
                            futures[future] = (task_id, day, hour)

                        for future in as_completed(futures):
                            task_id, day, hour = futures[future]
                            try:
                                result = future.result()
                                update_stats(result)
                                main_pbar.set_postfix_str(f"{day.strftime('%Y-%m-%d')} {hour:02d}:00")
                            except Exception:
                                with stats_lock:
                                    stats["failed"] += 1
                            main_pbar.update(1)

        except KeyboardInterrupt:
            print()
            print("Download interrupted by user.")
            print("Partial files can be resumed on next run.")
            return stats

        # Print statistics
        print()
        print("Download complete!")
        print(f"  Downloaded: {stats['downloaded']} files ({stats['total_mb']:.1f} MB)")
        print(f"  Skipped: {stats['skipped']} files")
        print(f"  Failed: {stats['failed']} files")

        return stats

    def _download_file(self, day: dt.date, hour: int) -> dict:
        """
        Download a single GFS file.

        Returns:
            dict with keys: status ('downloaded', 'skipped', 'failed'), fname, size_mb
        """
        dth = day + dt.timedelta(hours=hour)

        # Create destination directory
        dest_dir = self.data_dir / dth.strftime("%Y-%m")
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Filenames
        modern_fname = f"gfs.t{hour:02d}z.pgrb2.0p25.f000"
        legacy_fname = f"gfsanl_3_{dth.strftime('%Y%m%d_%H')}00_000.grb2"
        dest_path = dest_dir / legacy_fname

        # AWS S3 URL
        url = (
            f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
            f"gfs.{dth.strftime('%Y%m%d')}/{hour:02d}/atmos/"
            f"{modern_fname}"
        )

        # Check if file exists (resume support)
        resume_pos = 0
        if dest_path.exists():
            resume_pos = dest_path.stat().st_size

        # Download
        result = self._download_with_resume(url, dest_path, resume_pos)

        if result == "skip":
            return {"status": "skipped", "fname": legacy_fname, "size_mb": 0}
        elif result:
            if dest_path.exists() and dest_path.stat().st_size > self.min_file_size:
                size_mb = dest_path.stat().st_size / (1024 * 1024)

                # Filter if requested
                if self.filter_grib:
                    self._filter_grib_file(dest_path)

                return {"status": "downloaded", "fname": legacy_fname, "size_mb": size_mb}
            else:
                # Remove incomplete file
                if dest_path.exists():
                    dest_path.unlink()
                return {"status": "failed", "fname": legacy_fname, "size_mb": 0}
        else:
            return {"status": "failed", "fname": legacy_fname, "size_mb": 0}

    def _download_with_resume(self, url: str, dest_path: Path, resume_pos: int) -> Literal[True, False, "skip"]:
        """
        Download a file with resume support.

        Returns:
            True: Download succeeded
            False: Download failed
            "skip": File already complete
        """
        for attempt in range(self.max_retries):
            try:
                req = urllib.request.Request(url)
                if resume_pos > 0:
                    req.add_header("Range", f"bytes={resume_pos}-")

                with urllib.request.urlopen(req, timeout=30) as response:
                    if response.status == 206:  # Partial Content
                        mode = "ab"
                    elif response.status == 200:  # OK
                        if resume_pos > 0:
                            resume_pos = 0
                        mode = "wb"
                    elif response.status == 416:  # Range not satisfiable
                        return "skip"
                    else:
                        raise urllib.error.HTTPError(url, response.status, response.reason, None, None)

                    # Download
                    chunk_size = 8192
                    with open(dest_path, mode) as f:
                        while True:
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)

                return True

            except urllib.error.HTTPError as e:
                if e.code == 404:
                    return False
                if attempt < self.max_retries - 1:
                    import time

                    time.sleep(2**attempt)
                else:
                    return False
            except Exception:
                if attempt < self.max_retries - 1:
                    import time

                    time.sleep(2**attempt)
                else:
                    return False

        return False

    def _filter_grib_file(self, filepath: Path) -> None:
        """
        Filter GRIB file to keep only required parameters.

        Reduces file size by ~50%.
        """
        original_size = filepath.stat().st_size

        grb = pygrib.open(str(filepath))

        # Collect messages to keep
        filtered = []
        original_count = 0
        for msg in grb:
            original_count += 1
            if msg.name in self.REQUIRED_PARAMS:
                if msg.typeOfLevel in self.REQUIRED_PARAMS[msg.name]:
                    filtered.append(msg)

        kept_count = len(filtered)

        # Write filtered file
        temp_path = filepath.with_suffix(".filtered.tmp")
        with open(temp_path, "wb") as f:
            for msg in filtered:
                f.write(msg.tostring())

        # Replace original
        temp_path.replace(filepath)

        filtered_size = filepath.stat().st_size
        saved_mb = (original_size - filtered_size) / (1024 * 1024)

        print(f"  Filtered: {original_count} -> {kept_count} messages ({kept_count * 100 // original_count}% kept, saved {saved_mb:.1f} MB)")
