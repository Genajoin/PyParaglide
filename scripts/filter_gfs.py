#!/usr/bin/env python3
"""
Filter GFS GRIB files to keep only parameters needed for neural network training.

Reduces file size by ~50% by removing unused parameters while preserving
full compatibility with GribReader (neural_network/inc/grib_reader.py).

Usage:
    # Filter single file
    python filter_gfs.py path/to/file.grb2

    # Filter directory
    python filter_gfs.py data/gfs/anl/2025-06/

    # Dry run (show what would be done)
    python filter_gfs.py data/gfs/anl/ --dry-run
"""
import argparse
import os
import sys
from pathlib import Path

# Required parameters (must match neural_network/inc/dataset.py)
# These are the exact parameter names used by GribReader
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


def filter_grib_file(input_path, dry_run=False, verbose=False):
    """
    Filter GRIB file to keep only required parameters.

    Args:
        input_path: Path to input GRIB file
        dry_run: If True, don't modify files
        verbose: Print detailed information

    Returns:
        dict with statistics: {original_count, kept_count, removed_count,
                               size_before, size_after}
    """
    try:
        import pygrib
    except ImportError:
        print("Error: pygrib not available. Install with: pip install pygrib")
        sys.exit(1)

    original_size = os.path.getsize(input_path)

    # Open and analyze
    grb = pygrib.open(input_path)
    messages_to_keep = []
    removed_params = set()
    original_count = 0

    for msg in grb:
        original_count += 1
        param_name = msg.name
        level_type = msg.typeOfLevel

        if param_name in REQUIRED_PARAMS:
            if level_type in REQUIRED_PARAMS[param_name]:
                messages_to_keep.append(msg)
            else:
                removed_params.add(f"{param_name} ({level_type})")
        else:
            removed_params.add(f"{param_name} ({level_type})")

    kept_count = len(messages_to_keep)
    removed_count = original_count - kept_count

    if verbose:
        print(f"  File: {input_path}")
        print(f"  Original messages: {original_count}")
        print(f"  Kept messages: {kept_count} ({kept_count*100//original_count}%)")
        print(f"  Removed messages: {removed_count} ({removed_count*100//original_count}%)")

        if removed_params and len(removed_params) <= 20:
            print(f"  Removed types: {', '.join(sorted(removed_params)[:10])}")
        elif len(removed_params) > 20:
            print(f"  Removed types: {len(removed_params)} different parameter types")

    if dry_run:
        return {
            'original_count': original_count,
            'kept_count': kept_count,
            'removed_count': removed_count,
            'size_before': original_size,
            'size_after': original_size * kept_count // original_count,
        }

    # Write filtered file
    temp_path = input_path + '.tmp'
    with open(temp_path, 'wb') as f:
        for msg in messages_to_keep:
            f.write(msg.tostring())

    # Get new size and replace original
    filtered_size = os.path.getsize(temp_path)
    os.replace(temp_path, input_path)

    return {
        'original_count': original_count,
        'kept_count': kept_count,
        'removed_count': removed_count,
        'size_before': original_size,
        'size_after': filtered_size,
    }


def find_grib_files(path):
    """Find all GRIB files in path (file or directory)."""
    path = Path(path)

    if path.is_file():
        if path.suffix in ['.grb2', '.grb', '.grib2', '.grib']:
            return [path]
        return []

    if path.is_dir():
        files = []
        for ext in ['.grb2', '.grb', '.grib2', '.grib']:
            files.extend(path.glob(f'**/*{ext}'))
        return sorted(files)

    return []


def format_size(bytes_size):
    """Format bytes to human readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def main():
    parser = argparse.ArgumentParser(
        description="Filter GFS GRIB files to keep only parameters needed for neural network training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        'path',
        help='GRIB file or directory containing GRIB files'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without making changes'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed information for each file'
    )

    args = parser.parse_args()

    # Find GRIB files
    files = find_grib_files(args.path)

    if not files:
        print(f"No GRIB files found in: {args.path}")
        return 1

    print(f"Found {len(files)} GRIB file(s)")

    # Process files
    try:
        from tqdm import tqdm
        file_iter = tqdm(files, desc="Processing")
    except ImportError:
        file_iter = files
        print("Tip: Install tqdm for progress bar: pip install tqdm")

    total_stats = {
        'original_count': 0,
        'kept_count': 0,
        'removed_count': 0,
        'size_before': 0,
        'size_after': 0,
    }

    for filepath in file_iter:
        stats = filter_grib_file(
            str(filepath),
            dry_run=args.dry_run,
            verbose=args.verbose
        )

        for key in total_stats:
            total_stats[key] += stats[key]

    # Print summary
    print()
    print("=" * 60)
    print("Summary:")
    print("=" * 60)

    if total_stats['original_count'] > 0:
        kept_pct = total_stats['kept_count'] * 100 // total_stats['original_count']
        removed_pct = total_stats['removed_count'] * 100 // total_stats['original_count']
    else:
        kept_pct = removed_pct = 0

    print(f"  Total GRIB messages: {total_stats['original_count']}")
    print(f"  Kept: {total_stats['kept_count']} ({kept_pct}%)")
    print(f"  Removed: {total_stats['removed_count']} ({removed_pct}%)")
    print()
    print(f"  Size before: {format_size(total_stats['size_before'])}")
    print(f"  Size after:  {format_size(total_stats['size_after'])}")
    print(f"  Space saved: {format_size(total_stats['size_before'] - total_stats['size_after'])}")
    print("=" * 60)

    if args.dry_run:
        print()
        print("DRY RUN - No files were modified. Run without --dry-run to apply changes.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
