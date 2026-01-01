"""
Phase classes for dataset building pipeline.

Four phases:
1. BuildCellsPhase - Generate 1x1 degree cells and map to GRIB grid
2. BuildMeteoPhase - Scan GFS data and extract weather parameters
3. BuildFlightsPhase - Process xContest flights and create spot PKL files
4. BuildTerrainPhase - Extract mountainess data from elevation tiles
"""

import json
import math
import os
import pickle
import queue as Queue
import signal
import struct
import tempfile
import threading
import time
import multiprocessing
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import tqdm

# Add neural_network to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "neural_network"))

from inc.bin_obj import BinObj
from inc.tiles_maths import TilesMaths

# Metadata file name
METADATA_FILE = "dataset_config.json"


def load_metadata(out_dir: Path) -> Dict[str, Any]:
    """Load dataset metadata from JSON file."""
    metadata_path = out_dir / METADATA_FILE
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_metadata(out_dir: Path, metadata: Dict[str, Any]) -> None:
    """Save dataset metadata to JSON file."""
    metadata_path = out_dir / METADATA_FILE
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def check_config_match(current: Dict[str, Any], saved: Dict[str, Any]) -> bool:
    """Check if current configuration matches saved metadata."""
    if not current:
        return True  # No config to check
    if not saved:
        return False  # Have config but no saved data

    for key, value in current.items():
        if key not in saved:
            return False
        if isinstance(value, list):
            if saved[key] != value:
                return False
        elif isinstance(value, (int, float, str, bool)):
            if saved[key] != value:
                return False
        elif value is None:
            # None matches anything or missing
            continue
        else:
            # Unknown type, try direct comparison
            if saved.get(key) != value:
                return False
    return True

try:
    from inc.grib_reader import GribReader
except ImportError:
    GribReader = None

try:
    import pygrib
except ImportError:
    pygrib = None

# Local imports
from .grib_workers import file_reader, file_processor, assemble_day_results
from .flight_processor import (
    FlightIndexer, load_flights_from_json, process_flights,
    filter_spots_by_flights, create_spots_pkls,
)
from elevation_reader import ElevationReader


class BuildCellsPhase:
    """Phase 1: Build cell lists from bbox."""

    def __init__(self, bbox: Tuple[float, float, float, float],
                 gfs_dir: Path,
                 out_dir: Path,
                 force: bool = False):
        """
        Args:
            bbox: (lat_min, lat_max, lon_min, lon_max)
            gfs_dir: Path to GFS GRIB files
            out_dir: Output directory for PKL files
            force: Force rebuild even if PKL files exist
        """
        self.bbox = bbox
        self.bbox_list = list(bbox)  # For metadata comparison
        self.gfs_dir = gfs_dir
        self.out_dir = out_dir
        self.force = force

    def execute(self) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]:
        """
        Execute phase 1.

        Returns:
            (cells_latlon, cells_grib) - List of (lat, lon) and list of (row, col)
        """
        print("\n=== Phase 1: Building cells ===")

        # Check if PKL files already exist with matching config
        if not self.force and self._check_existing_pkl():
            saved_metadata = load_metadata(self.out_dir)
            current_config = {"bbox": self.bbox_list}

            if check_config_match(current_config, saved_metadata):
                print("  Found existing cells PKL files with matching bbox, skipping")
                print("  Use --force to rebuild")
                try:
                    with open(self.out_dir / "sorted_cells_latlon.pkl", 'rb') as f:
                        cells_latlon = pickle.load(f, encoding='latin1')
                    with open(self.out_dir / "sorted_cells.pkl", 'rb') as f:
                        cells_grib = pickle.load(f, encoding='latin1')
                    return cells_latlon, cells_grib
                except Exception as e:
                    print(f"  WARNING: Could not load existing PKL: {e}")
                    print("  Rebuilding...")
            else:
                print(f"  Config mismatch: saved bbox = {saved_metadata.get('bbox')}")
                print("  Rebuilding with new bbox...")

        lat_min, lat_max, lon_min, lon_max = self.bbox

        # Generate 1x1 degree cells
        cells_latlon = []
        for lat in range(int(lat_min), int(lat_max) + 1):
            for lon in range(int(lon_min), int(lon_max) + 1):
                cells_latlon.append((float(lat), float(lon)))

        nb_cells = len(cells_latlon)
        print(f"  Generated {nb_cells} cells")
        print(f"  Lat: {lat_min}°..{lat_max}°, Lon: {lon_min}°..{lon_max}°")

        # Save sorted_cells_latlon.pkl
        self._save_pkl("sorted_cells_latlon", cells_latlon)

        # Map to GRIB grid
        print("  Mapping cells to GRIB grid...")
        cells_grib = self._map_cells_to_grib(cells_latlon)

        if cells_grib:
            self._save_pkl("sorted_cells", cells_grib)
        else:
            print("  WARNING: Could not map to GRIB grid")
            cells_grib = [(0, 0)] * nb_cells
            self._save_pkl("sorted_cells", cells_grib)

        # Save metadata
        save_metadata(self.out_dir, {"bbox": self.bbox_list})

        return cells_latlon, cells_grib

    def _save_pkl(self, name: str, data) -> None:
        """Save data to PKL file."""
        print(f"  Saving {name}.pkl...", end=" ")
        BinObj.save(data, name, path=str(self.out_dir))
        print("OK")

    def _map_cells_to_grib(self, cells_latlon: List[Tuple[float, float]]) -> Optional[List[Tuple[int, int]]]:
        """Map cell coordinates to GRIB grid indices."""
        if not GribReader or not pygrib:
            return None

        sample_grib = self._find_sample_grib()
        if not sample_grib:
            return None

        try:
            grb_reader = GribReader(sample_grib)
            _, lats, lons = grb_reader.getInfos()

            cells_grib = []
            for lat, lon in cells_latlon:
                row = GribReader.findClosest(lat, lats, 0)
                col = GribReader.findClosest(lon, lons, 1)
                cells_grib.append((int(row), int(col)))

            return cells_grib

        except Exception as e:
            print(f"  WARNING: Failed to map GRIB indices: {e}")
            return None

    def _find_sample_grib(self) -> Optional[str]:
        """Find first available GRIB file."""
        if not self.gfs_dir.exists():
            return None

        for root, _, files in os.walk(self.gfs_dir):
            for file in files:
                if file.endswith(('.grb', '.grb2', '.grib', '.grib2')):
                    return os.path.join(root, file)

        return None

    def _check_existing_pkl(self) -> bool:
        """Check if cells PKL files exist."""
        return ((self.out_dir / "sorted_cells_latlon.pkl").exists() and
                (self.out_dir / "sorted_cells.pkl").exists())


class BuildMeteoPhase:
    """Phase 2: Scan GFS data and extract weather parameters."""

    def __init__(self, bbox: Tuple[float, float, float, float],
                 gfs_dir: Path,
                 cells_latlon: List[Tuple[float, float]],
                 out_dir: Path,
                 training_dates: Optional[str] = None,
                 num_workers: int = 4,
                 force: bool = False,
                 queue_size: int = 3):
        """
        Args:
            bbox: Bounding box
            gfs_dir: Path to GFS GRIB files
            cells_latlon: List of cell coordinates
            out_dir: Output directory
            training_dates: Date ranges string (YYYY-MM-DD:YYYY-MM-DD,...)
            num_workers: Number of worker processes
            force: Force rebuild even if PKL files exist
            queue_size: Size of GRIB job queue
        """
        self.bbox = bbox
        self.gfs_dir = gfs_dir
        self.cells_latlon = cells_latlon
        self.out_dir = out_dir
        self.training_dates = training_dates
        self.num_workers = num_workers
        self.force = force
        self.queue_size = queue_size

        self.meteo_days = []

    def execute(self) -> List[date]:
        """
        Execute phase 2.

        Returns:
            List of meteo days
        """
        print("\n=== Phase 2: Building meteo data ===")

        # Check if PKL files already exist with matching config
        if not self.force and self._check_existing_pkl():
            saved_metadata = load_metadata(self.out_dir)
            current_config = {"training_dates": self.training_dates} if self.training_dates else {}

            if check_config_match(current_config, saved_metadata):
                print("  Found existing PKL files with matching dates, skipping")
                print("  Use --force to rebuild")
                # Load existing meteo_days
                try:
                    with open(self.out_dir / "meteo_days.pkl", 'rb') as f:
                        self.meteo_days = pickle.load(f, encoding='latin1')
                    return self.meteo_days
                except Exception as e:
                    print(f"  WARNING: Could not load existing PKL: {e}")
                    print("  Rebuilding...")
            else:
                print(f"  Config mismatch: saved dates = {saved_metadata.get('training_dates')}")
                print("  Rebuilding with new dates...")

        if not self.gfs_dir.exists():
            print(f"  ERROR: GFS directory not found: {self.gfs_dir}")
            self._suggest_download()
            raise SystemExit(1)

        # Scan for available days
        self.meteo_days = self._scan_meteo_days()
        self._save_pkl("meteo_days", self.meteo_days)

        # Build meteo_params
        meteo_params = self._build_meteo_params()
        self._save_pkl("meteo_params", meteo_params)

        # Extract meteo content
        self._build_meteo_content()

        # Save metadata
        if self.training_dates:
            save_metadata(self.out_dir, {
                "bbox": list(self.bbox),
                "training_dates": self.training_dates
            })

        return self.meteo_days

    def _check_existing_pkl(self) -> bool:
        """Check if all required PKL files exist."""
        required = ["meteo_days.pkl", "meteo_params.pkl", "meteo_content_by_cell_day.pkl"]
        for name in required:
            if not (self.out_dir / name).exists():
                return False
        return True

    def _save_pkl(self, name: str, data) -> None:
        """Save data to PKL file."""
        BinObj.save(data, name, path=str(self.out_dir))

    def _parse_training_dates(self) -> Optional[List[Tuple[date, date]]]:
        """Parse TRAINING_DATES string."""
        if not self.training_dates:
            return None

        ranges = []
        for part in self.training_dates.split(','):
            part = part.strip()
            if ':' not in part:
                continue
            start_str, end_str = part.split(':')
            start = datetime.strptime(start_str.strip(), '%Y-%m-%d').date()
            end = datetime.strptime(end_str.strip(), '%Y-%m-%d').date()
            ranges.append((start, end))
        return sorted(ranges)

    def _is_date_in_ranges(self, target_date: date, ranges: List[Tuple[date, date]]) -> bool:
        """Check if date falls within any range."""
        for start, end in ranges:
            if start <= target_date <= end:
                return True
        return False

    def _scan_meteo_days(self) -> List[date]:
        """Scan GFS directory for available days with complete data."""
        date_ranges = self._parse_training_dates()

        all_meteo_days = {}

        for month_dir in sorted(os.listdir(self.gfs_dir)):
            month_path = self.gfs_dir / month_dir
            if not month_path.is_dir():
                continue

            files_by_date = {}
            for filename in os.listdir(month_path):
                if filename.startswith('gfsanl_3_') and filename.endswith('.grb2'):
                    parts = filename.split('_')
                    if len(parts) >= 4:
                        date_str = parts[2]
                        hour_str = parts[3][:2]

                        if date_str not in files_by_date:
                            files_by_date[date_str] = set()
                        files_by_date[date_str].add(hour_str)

            for date_str, hours in sorted(files_by_date.items()):
                if '06' in hours and '12' in hours and '18' in hours:
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                    day_date = date(year, month, day)
                    all_meteo_days[day_date] = True

        print(f"  Found {len(all_meteo_days)} days with complete GFS data")

        # Filter by training dates
        if date_ranges:
            meteo_days = [d for d in all_meteo_days if self._is_date_in_ranges(d, date_ranges)]
            print(f"  Filtered to {len(meteo_days)} days (TRAINING_DATES)")

            # Check for missing dates
            missing = []
            for start, end in date_ranges:
                current = start
                while current <= end:
                    if current not in all_meteo_days:
                        missing.append(current)
                    current += timedelta(days=1)

            if missing:
                print(f"  WARNING: {len(missing)} days from TRAINING_DATES are missing!")
        else:
            meteo_days = sorted(all_meteo_days.keys())

        return meteo_days

    def _build_meteo_params(self) -> List:
        """Build list of weather parameters (195 total: 65 params x 3 hours)."""
        meteo_params = []

        for hour in [6, 12, 18]:
            # Entire atmosphere parameters
            for param_name, levels in [
                ('Precipitable water', [[('entireAtmosphere', 0), ('unknown', 0)]]),
                ('Cloud water', [[('entireAtmosphere', 0), ('unknown', 0)]]),
            ]:
                for level_list in levels:
                    meteo_params.append((hour, param_name, level_list))

            # Pressure level parameters
            pressure_levels = [1000, 900, 800, 700, 600, 500, 400, 300, 200]
            for param_name in [
                'Vertical velocity', 'Geopotential Height', 'Absolute vorticity',
                'Temperature', 'Relative humidity', 'U component of wind', 'V component of wind'
            ]:
                for level in pressure_levels:
                    meteo_params.append((hour, param_name, [[('isobaricInhPa', level)]]))

        print(f"  Total parameters: {len(meteo_params)}")
        return meteo_params

    def _build_meteo_content(self) -> None:
        """Extract weather data from GRIB files using multiprocessing."""
        print("\n  Extracting weather data...")

        if not self.meteo_days or not GribReader:
            print("  WARNING: No meteo days or GRIB reader. Creating zeros matrix.")
            matrix = np.zeros((0, 195), dtype=np.float32)
            self._save_pkl("meteo_content_by_cell_day", matrix)
            return

        # Build GRIB params (65 without hour dimension)
        grib_params = []
        for param in ['Precipitable water', 'Cloud water']:
            grib_params.append((param, [('entireAtmosphere', 0), ('atmosphereSingleLayer', 0), ('unknown', 0)]))

        pressure_levels = [1000, 900, 800, 700, 600, 500, 400, 300, 200]
        for param in ['Vertical velocity', 'Geopotential Height', 'Absolute vorticity',
                      'Temperature', 'Relative humidity', 'U component of wind', 'V component of wind']:
            for level in pressure_levels:
                grib_params.append((param, [('isobaricInhPa', level)]))

        if len(grib_params) != 65:
            print(f"  ERROR: Expected 65 parameters, got {len(grib_params)}")
            return

        # Multiprocessing setup
        total_days = len(self.meteo_days)
        total_files = total_days * 3
        num_cells = len(self.cells_latlon)

        print(f"  Processing {total_days} days, {total_files} files")
        print(f"  Using {self.num_workers} workers")

        # Queues
        file_queue = Queue.Queue(maxsize=0)
        job_queue = multiprocessing.Queue(maxsize=self.queue_size)
        hourly_queue = multiprocessing.Queue(maxsize=1000)
        results_queue = Queue.Queue()

        stop_event = threading.Event()

        # Fill file queue
        for day_date in self.meteo_days:
            for hour in [6, 12, 18]:
                file_queue.put((day_date, hour))

        # Start reader thread
        reader = threading.Thread(
            target=file_reader,
            args=(file_queue, job_queue, str(self.gfs_dir), stop_event),
            daemon=True
        )
        reader.start()

        # Start worker processes
        workers = []
        for i in range(self.num_workers):
            p = multiprocessing.Process(
                target=file_processor,
                args=(job_queue, hourly_queue, grib_params, self.cells_latlon),
                daemon=True
            )
            p.start()
            workers.append(p)

        # Start assembler thread
        assembler = threading.Thread(
            target=assemble_day_results,
            args=(hourly_queue, results_queue, len(grib_params), num_cells),
            daemon=True
        )
        assembler.start()

        # Collect results
        all_data = []
        completed = 0
        start_time = time.time()

        try:
            while completed < total_days:
                try:
                    day_date, day_data = results_queue.get(timeout=1.0)
                    all_data.extend(day_data)
                    completed += 1

                    # Progress update every 15 seconds
                    if time.time() - start_time >= 15.0 or completed == total_days:
                        elapsed = time.time() - start_time
                        speed = completed / elapsed if elapsed > 0 else 0
                        if speed > 0:
                            eta_seconds = (total_days - completed) / speed
                            eta_str = f"{eta_seconds/60:.1f}min"
                        else:
                            eta_str = "?"

                        # Get queue sizes for diagnostics
                        try:
                            q_jobs = job_queue.qsize()
                            q_hourly = hourly_queue.qsize()
                            q_results = results_queue.qsize()
                        except NotImplementedError:
                            q_jobs = -1
                            q_hourly = -1
                            q_results = -1

                        print(f"  [{completed}/{total_days}] {completed/total_days*100:.1f}% | "
                              f"Speed: {speed:.2f} days/s | ETA: {eta_str} | "
                              f"Q: Jobs={q_jobs}/3 Hourly={q_hourly} Results={q_results}")
                        start_time = time.time()

                except Queue.Empty:
                    if not any(t.is_alive() for t in [reader] + workers + [assembler]):
                        print("  ERROR: All workers died!")
                        break

        except KeyboardInterrupt:
            print("\n  Interrupted!")
            stop_event.set()

            # Drain queues
            for q in [job_queue, hourly_queue, file_queue]:
                try:
                    while not q.empty():
                        q.get_nowait()
                except:
                    pass

            # Terminate workers
            for p in workers:
                p.terminate()
            raise SystemExit(130)

        # Normal cleanup
        for _ in range(self.num_workers):
            job_queue.put((None, None, None))
        for p in workers:
            p.join()

        hourly_queue.put((None, None, None))
        assembler.join()

        # Save results
        meteo_content = np.array(all_data, dtype=np.float32)
        print(f"  Matrix shape: {meteo_content.shape}")

        self._save_pkl("meteo_content_by_cell_day", meteo_content)

    def _suggest_download(self) -> None:
        """Suggest download commands."""
        print("\n  To download GFS data:")
        print("    python scripts/download_GFS.py --start-date YYYY-MM-DD --end-date YYYY-MM-DD")


class BuildFlightsPhase:
    """Phase 3: Process xContest flights and create spot PKL files."""

    def __init__(self, flights_dir: Path,
                 out_dir: Path,
                 cells_latlon: List[Tuple[float, float]],
                 meteo_days: List[date],
                 bbox: Optional[Tuple[float, float, float, float]] = None,
                 date_ranges: Optional[List[Tuple[date, date]]] = None,
                 min_flights: int = 200,
                 cluster_distance_km: Optional[float] = None):
        """
        Args:
            flights_dir: Path to xContest JSON files
            out_dir: Output directory for PKL files
            cells_latlon: List of cell coordinates
            meteo_days: List of meteo days
            bbox: Optional bbox filter
            date_ranges: Optional date range filters
            min_flights: Minimum flights per spot
            cluster_distance_km: Optional clustering radius
        """
        self.flights_dir = flights_dir
        self.out_dir = out_dir
        self.cells_latlon = cells_latlon
        self.meteo_days = meteo_days
        self.bbox = bbox
        self.date_ranges = date_ranges
        self.min_flights = min_flights
        self.cluster_distance_km = cluster_distance_km

    def execute(self) -> None:
        """Execute phase 3."""
        print("\n=== Phase 3: Processing flights ===")

        # Load flights from JSON
        print("\n  Loading flights from JSON...")
        flights = load_flights_from_json(self.flights_dir)
        if not flights:
            print("  WARNING: No flights loaded. Skipping flight processing.")
            return

        # Create indexer
        indexer = FlightIndexer(self.cells_latlon, self.meteo_days, self.date_ranges)

        # Create elevation reader
        elev_reader = ElevationReader()

        # Process flights
        print("\n  Processing flight data...")
        result = process_flights(
            flights,
            indexer,
            elev_reader,
            bbox=self.bbox,
            date_ranges=self.date_ranges,
            cluster_distance_km=self.cluster_distance_km
        )

        print(f"  Processed flights: {len(result['flights'])}")
        print(f"  Created spots: {len(result['spots'])}")

        # Filter spots by minimum flights
        print("\n  Filtering spots by minimum flights...")
        valid_spots = filter_spots_by_flights(
            result['flights'],
            min_flights=self.min_flights,
            bbox=self.bbox,
            date_ranges=self.date_ranges
        )

        if not valid_spots:
            print("  WARNING: No valid spots found. Skipping PKL generation.")
            return

        # Create PKL files
        print("\n  Creating spot PKL files...")
        nb_cells = len(self.cells_latlon)
        nb_days = len(self.meteo_days)

        create_spots_pkls(result['flights'], valid_spots, self.out_dir, nb_cells, nb_days)


class BuildTerrainPhase:
    """Phase 4: Extract mountainess data from elevation tiles."""

    def __init__(self, elevation_dir: Path,
                 out_dir: Path,
                 cells_latlon: List[Tuple[float, float]],
                 force: bool = False):
        """
        Args:
            elevation_dir: Path to elevation tiles
            out_dir: Output directory for PKL files
            cells_latlon: List of cell coordinates
            force: Force rebuild even if PKL files exist
        """
        self.elevation_dir = elevation_dir
        self.out_dir = out_dir
        self.cells_latlon = cells_latlon
        self.nb_cells = len(cells_latlon)  # For metadata comparison
        self.force = force

    def execute(self) -> None:
        """Execute phase 4."""
        print("\n=== Phase 4: Building mountainess_by_cell_alt ===")

        # Check if PKL file already exists with matching config
        if not self.force and (self.out_dir / "mountainess_by_cell_alt.pkl").exists():
            saved_metadata = load_metadata(self.out_dir)
            current_config = {"nb_cells": self.nb_cells}

            if check_config_match(current_config, saved_metadata):
                print("  Found existing mountainess PKL file with matching cells, skipping")
                print("  Use --force to rebuild")
                return
            else:
                print(f"  Config mismatch: saved nb_cells = {saved_metadata.get('nb_cells')}, current = {self.nb_cells}")
                print("  Rebuilding...")

        mountainess_by_cell_alt = []

        for lat, lon in tqdm.tqdm(self.cells_latlon, desc="  Processing cells"):
            mountainess = self._get_mountainess(lat, lon)
            mountainess_by_cell_alt.append([mountainess] * 5)

        self._save_pkl("mountainess_by_cell_alt", mountainess_by_cell_alt)

        # Save metadata
        save_metadata(self.out_dir, {"nb_cells": self.nb_cells})

    def _save_pkl(self, name: str, data) -> None:
        """Save data to PKL file."""
        BinObj.save(data, name, path=str(self.out_dir))

    def _get_mountainess(self, lat: float, lon: float) -> float:
        """Get mountainess value for coordinates."""
        zoom = 7
        try:
            coords = TilesMaths.LatLonToTileCoords(zoom, lat, lon)
            filepath = self.elevation_dir / str(zoom) / str(coords['tx']) / f"{coords['ty']}.mountainess"

            if not filepath.exists():
                return 0.5

            with open(filepath, 'rb') as f:
                content = f.read(256 * 256)

            byte_idx = coords['x'] * 256 + coords['y']
            if byte_idx >= len(content):
                return 0.5

            value = struct.unpack('B', content[byte_idx:byte_idx+1])[0]
            return float(value) / 255.0

        except Exception:
            return 0.5
