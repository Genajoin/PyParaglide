"""
Phase 2: Scan GFS data and extract weather parameters.

Key fix: Accepts multiple date ranges and accumulates ALL days before saving.
This fixes the sequential overwrite bug where each date range was overwriting previous data.
"""

import os
import pickle
import queue as Queue
import signal
import threading
import time
import multiprocessing
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

from pyparaglide.preprocessing.utils.bin_obj import BinObj
from pyparaglide.preprocessing.utils.metadata import load_metadata, save_metadata, check_config_match
from pyparaglide.preprocessing.workers.grib_reader import GribReader
from pyparaglide.preprocessing.workers.grib_workers import file_reader, file_processor, assemble_day_results


class BuildMeteoPhase:
    """Phase 2: Scan GFS data and extract weather parameters."""

    def __init__(self, bbox: Tuple[float, float, float, float],
                 gfs_dir: Path,
                 cells_latlon: List[Tuple[float, float]],
                 out_dir: Path,
                 date_ranges: Optional[List[Tuple[date, date]]] = None,
                 num_workers: int = 4,
                 force: bool = False,
                 queue_size: int = 3):
        """
        Args:
            bbox: Bounding box
            gfs_dir: Path to GFS GRIB files
            cells_latlon: List of cell coordinates
            out_dir: Output directory
            date_ranges: List of (start_date, end_date) tuples - CRITICAL: All ranges accumulated
            num_workers: Number of worker processes
            force: Force rebuild even if PKL files exist
            queue_size: Size of GRIB job queue
        """
        self.bbox = bbox
        self.gfs_dir = gfs_dir
        self.cells_latlon = cells_latlon
        self.out_dir = out_dir
        self.date_ranges = date_ranges  # List of tuples, NOT a single range!
        self.num_workers = num_workers
        self.force = force
        self.queue_size = queue_size

        self.meteo_days = []

    def execute(self) -> List[date]:
        """
        Execute phase 2.

        Returns:
            List of meteo days (accumulated from ALL date ranges)
        """
        print("\n=== Phase 2: Building meteo data ===")

        # Build training dates string from date_ranges for metadata
        training_dates_str = self._date_ranges_to_string()

        # Check if PKL files already exist with matching config
        if not self.force and self._check_existing_pkl():
            saved_metadata = load_metadata(self.out_dir)
            current_config = {"training_dates": training_dates_str} if training_dates_str else {}

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

        # Scan for available days - CRITICAL: Accumulates ALL date ranges
        self.meteo_days = self._scan_meteo_days()
        self._save_pkl("meteo_days", self.meteo_days)

        # Build meteo_params
        meteo_params = self._build_meteo_params()
        self._save_pkl("meteo_params", meteo_params)

        # Extract meteo content
        self._build_meteo_content()

        # Save metadata
        if training_dates_str:
            save_metadata(self.out_dir, {
                "bbox": list(self.bbox),
                "training_dates": training_dates_str
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

    def _date_ranges_to_string(self) -> Optional[str]:
        """Convert date_ranges list to string format for metadata."""
        if not self.date_ranges:
            return None
        return ','.join(f"{start.strftime('%Y-%m-%d')}:{end.strftime('%Y-%m-%d')}"
                       for start, end in self.date_ranges)

    def _is_date_in_ranges(self, target_date: date, ranges: List[Tuple[date, date]]) -> bool:
        """Check if date falls within any range."""
        if not ranges:
            return True
        for start, end in ranges:
            if start <= target_date <= end:
                return True
        return False

    def _scan_meteo_days(self) -> List[date]:
        """
        Scan GFS directory for available days with complete data.

        CRITICAL FIX: This method now accumulates days from ALL date ranges
        instead of just the last one. The loop below iterates through ALL
        date_ranges and extends the meteo_days list with matching days.
        """
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

        # Filter by training dates - CRITICAL: Check ALL date ranges
        if self.date_ranges:
            meteo_days = [d for d in all_meteo_days if self._is_date_in_ranges(d, self.date_ranges)]
            print(f"  Filtered to {len(meteo_days)} days (TRAINING_DATES)")

            # Check for missing dates
            missing = []
            for start, end in self.date_ranges:
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

        if not self.meteo_days or GribReader is None:
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

        # Fill file queue - CRITICAL: ALL meteo_days are processed here
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
        last_update_time = start_time

        try:
            while completed < total_days:
                try:
                    day_date, day_data = results_queue.get(timeout=1.0)
                    all_data.extend(day_data)
                    completed += 1

                    # Progress update every 15 seconds
                    current_time = time.time()
                    if current_time - last_update_time >= 15.0 or completed == total_days:
                        elapsed = current_time - start_time
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
                        last_update_time = current_time

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

        # Save results - CRITICAL: All accumulated data saved in ONE file
        meteo_content = np.array(all_data, dtype=np.float32)
        print(f"  Matrix shape: {meteo_content.shape}")

        self._save_pkl("meteo_content_by_cell_day", meteo_content)

    def _suggest_download(self) -> None:
        """Suggest download commands."""
        print("\n  To download GFS data:")
        print("    pyparaglide download --start YYYY-MM-DD --end YYYY-MM-DD")
