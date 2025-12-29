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
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from dateutil.rrule import rrule, DAILY
from tqdm import tqdm
import colorama

colorama.init(autoreset=True)
# Thread-safe tqdm
tqdm.set_lock(Lock())


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
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel download workers (default: 1)"
    )
    return parser.parse_args()


def download_gfs_file(url, dest_path, max_retries=3, progress_callback=None):
    """Download a single GFS file with retry logic and resume support.

    Args:
        url: URL to download from
        dest_path: Destination file path
        max_retries: Maximum retry attempts
        progress_callback: Optional callback(bytes_downloaded, total_bytes) for progress updates

    Returns:
        True: Download succeeded
        False: Download failed (404 or other error)
        'skip': File already complete
    """
    # Check if file exists partially (for resume)
    resume_pos = 0
    if os.path.isfile(dest_path):
        resume_pos = os.path.getsize(dest_path)
        if resume_pos > 0 and progress_callback:
            progress_callback(0, None)  # Signal resume start

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url)
            if resume_pos > 0:
                req.add_header('Range', f'bytes={resume_pos}-')

            with urllib.request.urlopen(req) as response:
                # Check if server supports range requests
                if response.status == 206:  # Partial Content
                    mode = 'ab'  # append binary
                elif response.status == 200:  # OK (full download)
                    if resume_pos > 0:
                        # Server doesn't support resume, start over
                        resume_pos = 0
                    mode = 'wb'  # write binary
                else:
                    raise urllib.error.HTTPError(url, response.status, response.reason, None, None)

                # Get total size if available
                total_size = None
                if 'Content-Length' in response.headers:
                    content_length = int(response.headers['Content-Length'])
                    # For 206: Content-Length is remaining bytes
                    # For 200: Content-Length is total file size
                    if response.status == 200:
                        total_size = content_length
                    else:
                        total_size = resume_pos + content_length

                    if progress_callback and total_size:
                        progress_callback(resume_pos, total_size)

                # Download with progress
                with open(dest_path, mode) as f:
                    downloaded = resume_pos
                    chunk_size = 8192
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        downloaded += len(chunk)
                        if progress_callback and total_size:
                            progress_callback(downloaded, total_size)

                return True

        except urllib.error.HTTPError as e:
            if e.code == 404:
                # File doesn't exist (old data may not be available)
                return False
            if e.code == 416:
                # Range not satisfiable - file already complete
                return 'skip'
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                time.sleep(wait_time)
            else:
                return False
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
            else:
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


def download_worker(task_id, day, hour, data_directory, max_retries=3, filter_after_download=False, progress_position=None):
    """Worker function for parallel downloads.

    Args:
        task_id: Unique identifier for this task
        day: datetime.date object
        hour: Hour (0-23)
        data_directory: Base directory for downloads
        max_retries: Maximum retry attempts
        filter_after_download: Whether to filter GRIB after download
        progress_position: Position for tqdm progress bar (for multi-line display)

    Returns:
        dict with keys: status ('downloaded', 'skipped', 'failed'), fname, size_mb
    """
    min_expected_gfs_file_size = 10000000  # 10MB
    dth = day + datetime.timedelta(hours=hour)

    # Create destination directory
    dest_dir = os.path.join(data_directory, dth.strftime("%Y-%m"))
    os.makedirs(dest_dir, exist_ok=True)

    # Build filename
    modern_fname = f"gfs.t{hour:02d}z.pgrb2.0p25.f000"
    legacy_fname = f"gfsanl_3_{dth.strftime('%Y%m%d_%H')}00_000.grb2"
    legacy_full_path = os.path.join(dest_dir, legacy_fname)

    # Build URL
    aws_url = (
        f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
        f"gfs.{dth.strftime('%Y%m%d')}/{hour:02d}/atmos/"
        f"{modern_fname}"
    )

    # Get remote file size for progress bar
    remote_size = None
    try:
        req = urllib.request.Request(aws_url, method='HEAD')
        with urllib.request.urlopen(req, timeout=10) as response:
            if 'Content-Length' in response.headers:
                remote_size = int(response.headers['Content-Length'])
    except:
        pass  # Will work without progress

    # Check resume position
    resume_pos = 0
    if os.path.isfile(legacy_full_path):
        resume_pos = os.path.getsize(legacy_full_path)

    # Create progress bar for this file
    file_pbar = None
    if progress_position is not None and remote_size:
        file_pbar = tqdm(
            total=remote_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024*1024,
            position=progress_position,
            leave=False,
            desc=f"{dth.strftime('%m-%d %H:%M')}",
            initial=resume_pos,
            disable=False
        )

    def progress_callback(downloaded, total):
        if file_pbar:
            file_pbar.n = downloaded
            file_pbar.refresh()

    # Download
    result = download_gfs_file(aws_url, legacy_full_path, max_retries, progress_callback)

    # Close file progress bar
    if file_pbar:
        file_pbar.close()

    # Process result
    if result == 'skip':
        return {'status': 'skipped', 'fname': legacy_fname, 'size_mb': 0}
    elif result:
        if os.path.isfile(legacy_full_path) and os.path.getsize(legacy_full_path) > min_expected_gfs_file_size:
            size_mb = os.path.getsize(legacy_full_path) / (1024 * 1024)

            # Filter if requested
            if filter_after_download:
                filter_grib_file(legacy_full_path)

            return {'status': 'downloaded', 'fname': legacy_fname, 'size_mb': size_mb}
        else:
            # Remove incomplete file
            if os.path.isfile(legacy_full_path):
                os.remove(legacy_full_path)
            return {'status': 'failed', 'fname': legacy_fname, 'size_mb': 0}
    else:
        return {'status': 'failed', 'fname': legacy_fname, 'size_mb': 0}


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

    # AWS S3 URL (modern format, available from 2021+)
    aws_url = (
        f"https://noaa-gfs-bdp-pds.s3.amazonaws.com/"
        f"gfs.{dth.strftime('%Y%m%d')}/{hour:02d}/atmos/"
        f"{modern_fname}"
    )

    # Try to download (with resume support)
    result = download_gfs_file(aws_url, legacy_full_path, max_retries)

    if result == 'skip':
        # File already complete
        stats['skipped'] += 1
        return True
    elif result:
        # Download succeeded (new or resumed)
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
    else:
        stats['failed'] += 1

    return False


def download_weather_data(args):
    """Main download function with support for parallel workers."""
    # Parse dates
    start = datetime.datetime.strptime(args.start_date, '%Y-%m-%d').date()
    end = datetime.datetime.strptime(args.end_date, '%Y-%m-%d').date()

    # Parse hours
    hours = [int(h.strip()) for h in args.hours.split(',')]

    # Thread-safe statistics
    stats = {
        'downloaded': 0,
        'skipped': 0,
        'failed': 0,
        'total_mb': 0.0
    }
    stats_lock = Lock()

    # Generate list of days
    days = list(rrule(DAILY, dtstart=start, until=end))
    total_tasks = len(days) * len(hours)

    print(f"Downloading GFS Analysis data:")
    print(f"  Date range: {args.start_date} to {args.end_date}")
    print(f"  Hours: {', '.join(map(str, hours))} UTC")
    print(f"  Output: {args.data_dir}")
    print(f"  Total files: {total_tasks}")
    print(f"  Workers: {args.workers}")
    if args.filter:
        print(f"  Filtering: ENABLED (files will be reduced by ~50%%)")
    print()

    # Generate task list
    tasks = []
    task_id = 0
    for day in days:
        for hour in hours:
            tasks.append((task_id, day, hour))
            task_id += 1

    def update_stats(result):
        """Thread-safe statistics update."""
        with stats_lock:
            if result['status'] == 'downloaded':
                stats['downloaded'] += 1
                stats['total_mb'] += result['size_mb']
            elif result['status'] == 'skipped':
                stats['skipped'] += 1
            else:
                stats['failed'] += 1

    # Download
    try:
        if args.workers == 1:
            # Sequential mode (for backward compatibility and single worker)
            with tqdm(total=total_tasks, desc="Files", unit="file") as pbar:
                for task_id, day, hour in tasks:
                    pbar.set_postfix_str(f"{day.strftime('%Y-%m-%d')} {hour:02d}:00")
                    worker_result = download_worker(
                        task_id, day, hour, args.data_dir,
                        args.max_retries, args.filter, progress_position=None
                    )
                    update_stats(worker_result)
                    pbar.update(1)
        else:
            # Parallel mode
            with tqdm(total=total_tasks, desc="Files", unit="file", position=0, leave=True) as main_pbar:
                with ThreadPoolExecutor(max_workers=args.workers) as executor:
                    # Submit all tasks
                    futures = {}
                    for task_id, day, hour in tasks:
                        future = executor.submit(
                            download_worker,
                            task_id, day, hour, args.data_dir,
                            args.max_retries, args.filter,
                            progress_position=(task_id % args.workers) + 1  # Position 1..N
                        )
                        futures[future] = (task_id, day, hour)

                    # Process completed tasks
                    try:
                        for future in as_completed(futures):
                            task_id, day, hour = futures[future]
                            try:
                                result = future.result()
                                update_stats(result)
                                main_pbar.set_postfix_str(f"{day.strftime('%Y-%m-%d')} {hour:02d}:00")
                            except Exception as e:
                                with stats_lock:
                                    stats['failed'] += 1
                            main_pbar.update(1)
                    except KeyboardInterrupt:
                        # Cancel remaining futures
                        for future in futures:
                            future.cancel()
                        raise  # Re-raise to outer handler
    except KeyboardInterrupt:
        print()
        print(colorama.Fore.YELLOW + "Download interrupted by user.")
        print("Partial files can be resumed on next run.")
        return 130  # Standard exit code for SIGINT

    # Print statistics
    print()
    print("Download complete!")
    print(f"  Downloaded: {stats['downloaded']} files ({stats['total_mb']:.1f} MB)")
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
