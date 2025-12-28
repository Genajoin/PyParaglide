#!/usr/bin/env python3
"""
Download GFS Analysis data from NOAA nomads archive.

Downloads historical GFS Analysis GRIB files for specified date range.
"""

import argparse
import datetime
import os
import sys
import time
import urllib.request
import urllib.error
from dateutil.rrule import rrule, DAILY
from tqdm import tqdm
import colorama

colorama.init(autoreset=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download GFS Analysis data from NOAA nomads archive"
    )
    parser.add_argument(
        "--start-date",
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end-date",
        required=True,
        help="End date (YYYY-MM-DD, inclusive)"
    )
    parser.add_argument(
        "--data-dir",
        default="data/gfs/anl",
        help="Output directory (default: data/gfs/anl)"
    )
    parser.add_argument(
        "--hours",
        default="0,6,12,18",
        help="UTC hours to download, comma-separated (default: 0,6,12,18)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retry attempts for failed downloads (default: 3)"
    )
    parser.add_argument(
        "--filter",
        action="store_true",
        help="Filter GRIB files after download to reduce size by ~50%% (keeps only parameters needed for training)"
    )
    return parser.parse_args()


def download_gfs_file(url, dest_path, max_retries=3):
    """Download a single GFS file with retry logic."""
    for attempt in range(max_retries):
        try:
            urllib.request.urlretrieve(url, dest_path)
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # File doesn't exist (old data may not be available)
                return False
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return False
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"  Retry {attempt + 1}/{max_retries} after {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  Failed after {max_retries} attempts: {e}")
                return False
    return False


def filter_grib_file(filepath):
    """
    Filter GRIB file to keep only parameters needed for neural network training.

    Reduces file size by ~50% by removing unused parameters.
    Preserves full compatibility with GribReader.

    Args:
        filepath: Path to GRIB file to filter (modified in place)
    """
    try:
        import pygrib
    except ImportError:
        print("  Warning: pygrib not available, skipping filtering")
        return

    # Required parameters (must match neural_network/inc/dataset.py)
    REQUIRED_PARAMS = {
        'Temperature': ['isobaricInhPa'],
        'U component of wind': ['isobaricInhPa'],
        'V component of wind': ['isobaricInhPa'],
        'Relative humidity': ['isobaricInhPa'],
        'Geopotential Height': ['isobaricInhPa'],
        'Vertical velocity': ['isobaricInhPa'],
        'Absolute vorticity': ['isobaricInhPa'],
        'Precipitable water': ['entireAtmosphere', 'unknown'],
        'Cloud water': ['entireAtmosphere', 'unknown'],
    }

    # Get original size before filtering
    original_size = os.path.getsize(filepath)

    grb = pygrib.open(filepath)

    # Collect messages to keep
    filtered = []
    original_count = 0
    for msg in grb:
        original_count += 1
        if msg.name in REQUIRED_PARAMS:
            if msg.typeOfLevel in REQUIRED_PARAMS[msg.name]:
                filtered.append(msg)

    kept_count = len(filtered)

    # Write filtered file
    temp_path = filepath + '.filtered.tmp'
    with open(temp_path, 'wb') as f:
        for msg in filtered:
            f.write(msg.tostring())

    # Replace original
    os.replace(temp_path, filepath)

    filtered_size = os.path.getsize(filepath)
    saved_mb = (original_size - filtered_size) / (1024 * 1024)

    print(f"  Filtered: {original_count} -> {kept_count} messages ({kept_count*100//original_count}% kept, saved {saved_mb:.1f} MB)")


def download_gfs_hour(day, hour, data_directory, stats, max_retries=3, filter_after_download=False):
    """Download GFS data for a specific day and hour."""
    min_expected_gfs_file_size = 10000000  # 10MB

    dth = day + datetime.timedelta(hours=hour)

    # Create destination directory
    dest_dir = os.path.join(data_directory, dth.strftime("%Y-%m"))
    os.makedirs(dest_dir, exist_ok=True)

    # Modern GFS format (2021+): gfs.tHHz.pgrb2.0p25.f000
    # Legacy format (<2021): gfsanl_3_YYYYMMDD_HH00_000.grb[2]

    # Try modern format first (AWS S3)
    modern_fname = f"gfs.t{hour:02d}z.pgrb2.0p25.f000"
    modern_full_path = os.path.join(dest_dir, modern_fname)

    # Also save as legacy filename for compatibility with GribReader
    legacy_fname = f"gfsanl_3_{dth.strftime('%Y%m%d_%H')}00_000.grb2"
    legacy_full_path = os.path.join(dest_dir, legacy_fname)

    # Skip if file already exists
    if os.path.isfile(legacy_full_path) and os.path.getsize(legacy_full_path) > min_expected_gfs_file_size:
        stats['skipped'] += 1
        return True

    # AWS S3 URL (modern format, available from 2021+)
    aws_url = (
        f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
        f"gfs.{dth.strftime('%Y%m%d')}/{hour:02d}/atmos/"
        f"{modern_fname}"
    )

    # Try to download
    if download_gfs_file(aws_url, legacy_full_path, max_retries):
        # Verify download succeeded
        if os.path.isfile(legacy_full_path) and os.path.getsize(legacy_full_path) > min_expected_gfs_file_size:
            stats['downloaded'] += 1

            # Filter if requested
            if filter_after_download:
                filter_grib_file(legacy_full_path)

            return True
        else:
            stats['failed'] += 1
            if os.path.isfile(legacy_full_path):
                os.remove(legacy_full_path)  # Remove incomplete file

    stats['failed'] += 1
    return False


def download_weather_data(args):
    """Main download function."""
    # Parse dates
    start = datetime.datetime.strptime(args.start_date, '%Y-%m-%d').date()
    end = datetime.datetime.strptime(args.end_date, '%Y-%m-%d').date()

    # Parse hours
    hours = [int(h.strip()) for h in args.hours.split(',')]

    # Statistics
    stats = {
        'downloaded': 0,
        'skipped': 0,
        'failed': 0
    }

    # Generate list of days
    days = list(rrule(DAILY, dtstart=start, until=end))
    total_tasks = len(days) * len(hours)

    print(f"Downloading GFS Analysis data:")
    print(f"  Date range: {args.start_date} to {args.end_date}")
    print(f"  Hours: {', '.join(map(str, hours))} UTC")
    print(f"  Output: {args.data_dir}")
    print(f"  Total files: {total_tasks}")
    if args.filter:
        print(f"  Filtering: ENABLED (files will be reduced by ~50%%)")
    print()

    # Download with progress bar
    with tqdm(total=total_tasks, desc="Downloading", unit="file") as pbar:
        for day in days:
            for hour in hours:
                pbar.set_postfix_str(f"{day.strftime('%Y-%m-%d')} {hour:02d}:00")
                download_gfs_hour(day, hour, args.data_dir, stats, args.max_retries, args.filter)
                pbar.update(1)

    # Print statistics
    print()
    print("Download complete!")
    print(f"  Downloaded: {stats['downloaded']} files")
    print(f"  Skipped (already exists): {stats['skipped']} files")
    print(f"  Failed: {stats['failed']} files")

    if stats['failed'] > 0:
        print()
        print(colorama.Fore.YELLOW + "Warning: Some files failed to download.")
        print("This may be normal for older data (404 errors).")
        return 1

    return 0


if __name__ == "__main__":
    args = parse_args()
    sys.exit(download_weather_data(args))
