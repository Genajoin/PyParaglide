#!/usr/bin/env python3
"""
Unified dataset builder for Paraglidable neural network training.

Replaces:
- scripts/build_pkl_dataset.py
- scripts/build_pkl_from_xcontest.py

Usage:
    python scripts/build_dataset.py
    python scripts/build_dataset.py --no-flights
    python scripts/build_dataset.py --dates 2021-06-01:2021-08-31 --bbox 45,47,13,15
"""

import argparse
import multiprocessing
import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import List, Tuple, Optional

# Try to load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # .env support is optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_dataset.phases import (
    BuildCellsPhase,
    BuildMeteoPhase,
    BuildFlightsPhase,
    BuildTerrainPhase,
)

# Default paths
DEFAULT_GFS_DIR = PROJECT_ROOT / "data" / "gfs" / "anl"
DEFAULT_ELEVATION_DIR = PROJECT_ROOT / "tiler" / "_cache" / "elevation"
DEFAULT_FLIGHTS_DIR = PROJECT_ROOT / "data" / "flights"
DEFAULT_OUT_DIR = PROJECT_ROOT / "neural_network" / "bin" / "data"


def parse_date_ranges(dates_str: str) -> Optional[List[Tuple[date, date]]]:
    """Parse date ranges string 'YYYY-MM-DD:YYYY-MM-DD,YYYY-MM-DD:YYYY-MM-DD'."""
    if not dates_str:
        return None

    ranges = []
    for part in dates_str.split(','):
        part = part.strip()
        if ':' not in part:
            raise ValueError(f"Invalid date range: {part}")
        start_str, end_str = part.split(':')
        start = datetime.strptime(start_str.strip(), '%Y-%m-%d').date()
        end = datetime.strptime(end_str.strip(), '%Y-%m-%d').date()
        if start > end:
            raise ValueError(f"Start date must be before end date: {part}")
        ranges.append((start, end))

    return sorted(ranges)


def parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    """Parse bbox string 'lat_min,lat_max,lon_min,lon_max'."""
    parts = [float(x.strip()) for x in bbox_str.split(',')]
    if len(parts) != 4:
        raise ValueError("bbox must have 4 values: lat_min,lat_max,lon_min,lon_max")
    lat_min, lat_max, lon_min, lon_max = parts
    if lat_min >= lat_max or lon_min >= lon_max:
        raise ValueError("bbox min must be less than max")
    return lat_min, lat_max, lon_min, lon_max


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build PKL dataset for neural network training"
    )

    # Bounding box
    parser.add_argument(
        "--bbox",
        default=os.environ.get("TRAINING_BBOX"),
        help="Bounding box: lat_min,lat_max,lon_min,lon_max (default: from TRAINING_BBOX env)"
    )

    # Date ranges
    parser.add_argument(
        "--dates",
        default=os.environ.get("TRAINING_DATES"),
        help="Date ranges: YYYY-MM-DD:YYYY-MM-DD,YYYY-MM-DD:YYYY-MM-DD (default: from TRAINING_DATES env)"
    )

    # Paths
    parser.add_argument(
        "--gfs-dir",
        default=str(DEFAULT_GFS_DIR),
        help=f"Path to GFS GRIB files (default: {DEFAULT_GFS_DIR})"
    )
    parser.add_argument(
        "--elevation-dir",
        default=str(DEFAULT_ELEVATION_DIR),
        help=f"Path to elevation tiles (default: {DEFAULT_ELEVATION_DIR})"
    )
    parser.add_argument(
        "--flights-dir",
        default=str(DEFAULT_FLIGHTS_DIR),
        help=f"Path to xContest JSON files (default: {DEFAULT_FLIGHTS_DIR})"
    )
    parser.add_argument(
        "--out-dir",
        default=str(DEFAULT_OUT_DIR),
        help=f"Output directory for PKL files (default: {DEFAULT_OUT_DIR})"
    )

    # Flight processing
    parser.add_argument(
        "--no-flights",
        action="store_true",
        help="Skip flight processing (only meteo data)"
    )
    parser.add_argument(
        "--min-flights",
        type=int,
        default=int(os.environ.get("MIN_FLIGHTS_PER_SPOT", "200")),
        help="Minimum flights per spot (default: 200)"
    )
    parser.add_argument(
        "--cluster-distance",
        type=float,
        default=float(os.environ.get("SPOT_CLUSTER_DISTANCE_KM", "0")),
        help="Spot clustering radius in km (0 = disabled, default: from SPOT_CLUSTER_DISTANCE_KM env)"
    )

    # Performance
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("BUILD_WORKERS", multiprocessing.cpu_count())),
        help=f"Number of worker processes (default: CPU count or BUILD_WORKERS env)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force rebuild even if PKL files exist"
    )
    parser.add_argument(
        "--queue-size",
        type=int,
        default=int(os.environ.get("BUILD_QUEUE_SIZE", "3")),
        help="Size of GRIB job queue (default: 3 or BUILD_QUEUE_SIZE env)"
    )

    args = parser.parse_args()

    # Validate bbox
    if not args.bbox:
        print("ERROR: --bbox is required or set TRAINING_BBOX in .env file", file=sys.stderr)
        return 1

    try:
        bbox = parse_bbox(args.bbox)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Parse date ranges
    try:
        date_ranges = parse_date_ranges(args.dates) if args.dates else None
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Convert paths
    gfs_dir = Path(args.gfs_dir)
    elevation_dir = Path(args.elevation_dir)
    flights_dir = Path(args.flights_dir)
    out_dir = Path(args.out_dir)

    # Ensure output directory exists
    out_dir.mkdir(parents=True, exist_ok=True)

    # Print configuration
    print("=== Paraglidable Dataset Builder ===")
    print(f"  Bbox: {bbox}")
    print(f"  Output: {out_dir}")
    print(f"  Workers: {args.workers}")

    if date_ranges:
        print(f"  Date ranges: {len(date_ranges)}")
        for i, (start, end) in enumerate(date_ranges):
            print(f"    {i+1}. {start} - {end}")

    if args.no_flights:
        print("  Flights: SKIPPED (--no-flights)")
    else:
        print(f"  Flights dir: {flights_dir}")
        print(f"  Min flights per spot: {args.min_flights}")
        if args.cluster_distance > 0:
            print(f"  Cluster distance: {args.cluster_distance} km")

    try:
        # Phase 1: Build cells
        phase1 = BuildCellsPhase(
            bbox=bbox,
            gfs_dir=gfs_dir,
            out_dir=out_dir,
            force=args.force
        )
        cells_latlon, _ = phase1.execute()

        # Phase 2: Build meteo
        phase2 = BuildMeteoPhase(
            bbox=bbox,
            gfs_dir=gfs_dir,
            cells_latlon=cells_latlon,
            out_dir=out_dir,
            training_dates=args.dates,
            num_workers=args.workers,
            force=args.force,
            queue_size=args.queue_size
        )
        meteo_days = phase2.execute()

        # Phase 3: Build flights (skip if --no-flights)
        if not args.no_flights:
            phase3 = BuildFlightsPhase(
                flights_dir=flights_dir,
                out_dir=out_dir,
                cells_latlon=cells_latlon,
                meteo_days=meteo_days,
                bbox=bbox,
                date_ranges=date_ranges,
                min_flights=args.min_flights,
                cluster_distance_km=args.cluster_distance if args.cluster_distance > 0 else None
            )
            phase3.execute()
        else:
            print("\n=== Phase 3: SKIPPED (--no-flights) ===")
            # Create empty flights_by_cell_day.pkl
            import pickle
            nb_cells = len(cells_latlon)
            nb_days = len(meteo_days)
            empty = [[] for _ in range(nb_days * nb_cells)]
            with open(out_dir / "flights_by_cell_day.pkl", 'wb') as f:
                pickle.dump(empty, f)
            print("  Created empty flights_by_cell_day.pkl")

        # Phase 4: Build terrain
        phase4 = BuildTerrainPhase(
            elevation_dir=elevation_dir,
            out_dir=out_dir,
            cells_latlon=cells_latlon,
            force=args.force
        )
        phase4.execute()

        print("\n=== Dataset built successfully! ===")
        print("\nNext steps:")
        print("  python neural_network/train.py")

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
