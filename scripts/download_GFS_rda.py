#!/usr/bin/env python3
"""
Download GFS Analysis data from NCAR RDA ds084.1 for paraglidable training.

Requirements:
    - Register at https://rda.ucar.edu/login/
    - Install: pip install httpx python-dotenv tqdm
    - Create .env file with UCAR_EMAIL and UCAR_PASS

Usage:
    python download_GFS_rda.py 2012-07-01 2012-07-31 data/gfs/anl 34 42 20 28
"""

import argparse
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

try:
    import httpx
    from tqdm import tqdm
    from dotenv import load_dotenv
except ImportError as e:
    print(f"ERROR: Missing required module: {e}")
    print("Install with: pip install httpx python-dotenv tqdm")
    sys.exit(1)

load_dotenv()


class RDAAuth:
    """Handle RDA UCAR authentication."""

    def __init__(self):
        self.email = os.environ.get("UCAR_EMAIL")
        self.password = os.environ.get("UCAR_PASS")

        if not self.email or not self.password:
            print("ERROR: UCAR_EMAIL and UCAR_PASS must be set in .env file")
            print("Create a .env file with:")
            print("  UCAR_EMAIL=your@email.com")
            print("  UCAR_PASS=your_password")
            sys.exit(1)

        self.cookies = self._auth()

    def _auth(self):
        """Authenticate with RDA and get cookies."""
        auth_url = "https://rda.ucar.edu/cgi-bin/login"
        auth_data = {
            "email": self.email,
            "passwd": self.password,
            "action": "login",
        }

        print(f"Authenticating as {self.email}...")
        res = httpx.post(auth_url, data=auth_data)
        if res.status_code != 200:
            print(f"ERROR: Authentication failed (status {res.status_code})")
            sys.exit(1)

        print("Authentication successful!")
        return res.cookies


def download_file(url: str, cookies, timeout: int = 60) -> str:
    """Download a file to temporary location with progress bar."""
    with tempfile.NamedTemporaryFile(delete=False) as file:
        num_of_retries = 5
        retry_wait = 5

        for i in range(num_of_retries):
            try:
                with httpx.stream("GET", url, cookies=cookies, timeout=timeout) as res:
                    if res.status_code == 404:
                        return None

                    total = int(res.headers.get("Content-Length", 0))
                    with tqdm(total=total, unit_scale=True, unit_divisor=1024, unit="B") as progress:
                        num_bytes_downloaded = res.num_bytes_downloaded
                        for chunk in res.iter_bytes():
                            file.write(chunk)
                            progress.update(res.num_bytes_downloaded - num_bytes_downloaded)
                            num_bytes_downloaded = res.num_bytes_downloaded
                break
            except httpx.TimeoutException:
                if i == num_of_retries - 1:
                    raise
                print(f"Timeout, retrying in {retry_wait}s...")
                time.sleep(retry_wait)
                retry_wait *= 2
            except Exception as e:
                if i == num_of_retries - 1:
                    raise
                print(f"Error: {e}, retrying...")
                time.sleep(retry_wait)

        return file.name


def build_rda_url(date: datetime, hour: int) -> str:
    """Build RDA URL for GFS analysis file (forecast hour 0 only)."""
    ymd = date.strftime("%Y%m%d")
    return f"https://rda.ucar.edu/data/ds084.1/{date.year}/{ymd}/gfs.0p25.{ymd}{hour:02d}.f000.grib2"


def download_gfs_hour(date: datetime, hour: int, dest_dir: str, auth: RDAAuth, stats: dict):
    """Download GFS data for a specific day and hour."""
    dth = date + timedelta(hours=hour)

    # Create destination directory
    dest_dir_path = Path(dest_dir) / dth.strftime("%Y-%m")
    dest_dir_path.mkdir(parents=True, exist_ok=True)

    # Legacy filename for compatibility with GribReader
    legacy_fname = f"gfsanl_3_{dth.strftime('%Y%m%d_%H')}00_000.grb2"
    legacy_full_path = dest_dir_path / legacy_fname

    # Skip if file already exists and is large enough
    min_expected_size = 10 * 1024 * 1024  # 10 MB
    if legacy_full_path.exists() and legacy_full_path.stat().st_size > min_expected_size:
        stats['skipped'] += 1
        return True

    # Build URL and download
    url = build_rda_url(date, hour)
    print(f"  Downloading: {url}")

    try:
        tmp_path = download_file(url, auth.cookies)

        if tmp_path is None:
            print(f"  File not found (404): {url}")
            stats['failed'] += 1
            return False

        # Move to final destination
        Path(tmp_path).rename(legacy_full_path)

        size_mb = legacy_full_path.stat().st_size / (1024 * 1024)
        print(f"  Saved: {legacy_full_path} ({size_mb:.1f} MB)")
        stats['downloaded'] += 1
        return True

    except Exception as e:
        print(f"  Failed: {e}")
        stats['failed'] += 1
        return False


def download_weather_data(start_date, end_date, hours, dest_dir, stats):
    """Main download function."""
    # Generate list of days
    days = []
    current = start_date
    while current <= end_date:
        days.append(current)
        current += timedelta(days=1)

    total_tasks = len(days) * len(hours)

    print(f"\nDownloading GFS from NCAR RDA ds084.1:")
    print(f"  Date range: {start_date.date()} to {end_date.date()}")
    print(f"  Hours: {hours}")
    print(f"  Output: {dest_dir}")
    print(f"  Total files: {total_tasks}")
    print()

    # Authenticate
    auth = RDAAuth()

    # Download files
    for day in days:
        for hour in hours:
            print(f"\n[{day.date()} {hour:02d}:00 UTC]")
            download_gfs_hour(day, hour, dest_dir, auth, stats)


def main():
    parser = argparse.ArgumentParser(
        description="Download GFS Analysis data from NCAR RDA ds084.1"
    )
    parser.add_argument("start_date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("end_date", help="End date (YYYY-MM-DD)")
    parser.add_argument("data_dir", help="Output directory (e.g., data/gfs/anl)")
    parser.add_argument("--hours", default="6,12,18", help="UTC hours (default: 6,12,18)")

    args = parser.parse_args()

    # Parse dates
    start = datetime.strptime(args.start_date, '%Y-%m-%d')
    end = datetime.strptime(args.end_date, '%Y-%m-%d')

    # Parse hours
    hours = [int(h.strip()) for h in args.hours.split(',')]

    # Statistics
    stats = {'downloaded': 0, 'skipped': 0, 'failed': 0}

    # Download
    try:
        download_weather_data(start, end, hours, args.data_dir, stats)
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)

    # Print statistics
    print()
    print("=" * 50)
    print("Download complete!")
    print(f"  Downloaded: {stats['downloaded']} files")
    print(f"  Skipped (already exists): {stats['skipped']} files")
    print(f"  Failed: {stats['failed']} files")
    print("=" * 50)

    if stats['failed'] > 0:
        print("\nNote: Some files may not be available in the RDA archive.")
        print("This is normal for older data or specific dates.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
