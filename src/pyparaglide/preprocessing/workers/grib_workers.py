"""
Worker functions for multiprocessing GRIB file processing.

3-stage hybrid pipeline:
- Reader Thread (HDD -> RAM disk)
- Worker Processes (RAM -> CPU)
- Assembler Thread (hourly results -> daily records)

Original source: scripts/build_dataset/grib_workers.py
"""

import gc
import os
import queue as Queue
import signal
import tempfile
import uuid
from typing import List, Tuple, Optional

import numpy as np


def file_reader(file_queue, job_queue, gfs_dir, stop_event, include_grib_path=False):
    """
    Reader Thread (Main Process): SEQUENTIAL HDD I/O.
    Reads GRIB from HDD, writes copy to /dev/shm (RAM), puts PATH into job_queue.

    Args:
        file_queue: Queue with (day_date, hour) tuples
        job_queue: Queue for (day_date, hour, temp_path, grib_path) results
        gfs_dir: Path to GFS GRIB files directory
        stop_event: Threading event to stop processing
        include_grib_path: If True, include original GRIB path in job_queue (for caching)
    """
    temp_dir = '/dev/shm' if os.path.exists('/dev/shm') else tempfile.gettempdir()

    while not stop_event.is_set():
        try:
            day_date, hour = file_queue.get(timeout=1)
        except Queue.Empty:
            break

        if day_date is None:
            break

        grb_path = os.path.join(
            gfs_dir,
            day_date.strftime('%Y-%m'),
            f"gfsanl_3_{day_date.strftime('%Y%m%d')}_{hour:02d}00_000.grb2"
        )

        if not os.path.exists(grb_path):
            if include_grib_path:
                job_queue.put((day_date, hour, None, grb_path))  # Missing file
            else:
                job_queue.put((day_date, hour, None))  # Missing file
            continue

        try:
            # 1. Read ENTIRE file from HDD (Sequential I/O)
            with open(grb_path, 'rb') as f:
                data = f.read()

            # 2. Write to RAM disk (Fast I/O) for workers
            temp_path = os.path.join(temp_dir, f"grib_{uuid.uuid4()}.grb2")
            with open(temp_path, 'wb') as f:
                f.write(data)

            # 3. Pass PATH to workers (avoid pickling 500MB data)
            if include_grib_path:
                job_queue.put((day_date, hour, temp_path, grb_path))
            else:
                job_queue.put((day_date, hour, temp_path))

        except Exception:
            if include_grib_path:
                job_queue.put((day_date, hour, None, grb_path))
            else:
                job_queue.put((day_date, hour, None))


def file_processor(job_queue, hourly_queue, grib_params, cells_latlon, cache_dir=None):
    """
    Worker Process (Separate CPU Core): CPU INTENSIVE.
    Reads from RAM disk, parses GRIB, extracts values.

    Args:
        job_queue: Queue with (day_date, hour, temp_path) tuples
        hourly_queue: Queue for (day_date, hour, values) results
        grib_params: List of 69 GRIB parameters to extract (65 base + 4 thermo)
        cells_latlon: List of (lat, lon) cell coordinates
        cache_dir: Path to GRIB cache directory (None = disable cache)
    """
    # Re-import inside process
    from pyparaglide.preprocessing.workers.grib_reader import InMemoryGribReader
    from pyparaglide.preprocessing.cache import GribCache

    if InMemoryGribReader is None:
        # No GRIB reader available, just consume the queue
        while True:
            try:
                item = job_queue.get()
            except Exception:
                break
            # Handle both 3-tuple and 4-tuple formats
            if len(item) == 3:
                day_date, hour, temp_path = item
                grib_path = None
            elif len(item) == 4:
                day_date, hour, temp_path, grib_path = item
            else:
                break

            if day_date is None:
                break
            hourly_queue.put((day_date, hour, None))
            if temp_path and os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
        return

    # Ignore SIGINT in workers, let main process handle cleanup
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # Initialize cache if enabled
    cache = GribCache(cache_dir) if cache_dir else None

    while True:
        try:
            item = job_queue.get()
        except Exception:
            break

        # Handle both 3-tuple and 4-tuple formats
        if len(item) == 3:
            day_date, hour, temp_path = item
            grib_path = None
        elif len(item) == 4:
            day_date, hour, temp_path, grib_path = item
        else:
            break

        if day_date is None:  # Sentinel
            break

        if temp_path is None:  # Missing file
            hourly_queue.put((day_date, hour, None))
            continue

        # Check per-cell cache if enabled and grib_path is provided
        if cache and grib_path:
            try:
                # Check which cells have per-cell cache
                _, missing_cells = cache.has_cell_cache(grib_path, cells_latlon)

                # Only load from cache if ALL cells are cached
                if not missing_cells:
                    # Load all cells from per-cell cache
                    values = cache.load_cells_batch(grib_path, cells_latlon, flatten=True).tolist()
                    hourly_queue.put((day_date, hour, values))
                    # Cleanup temp file
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except:
                            pass
                    continue
            except Exception:
                # Cache miss or error, fall through to GRIB parsing
                pass

        grb_reader = None
        try:
            # Parse with pygrib (CPU intensive, Parallel)
            grb_reader = InMemoryGribReader(temp_path)
            values = grb_reader.get_values(grib_params, cells_latlon)

            # Validate data length
            expected_len = len(grib_params) * len(cells_latlon)
            if values is None:
                print(f"[WARNING] {day_date} {hour}:00 - GRIB reader returned None")
                hourly_queue.put((day_date, hour, None))
            elif len(values) != expected_len:
                actual = len(values)
                missing = expected_len - actual
                print(f"[WARNING] {day_date} {hour}:00 - Got {actual}/{expected_len} values ({missing} missing)")
                # Still send what we have instead of None!
                hourly_queue.put((day_date, hour, values))
            else:
                hourly_queue.put((day_date, hour, values))

                # Save to per-cell cache if enabled and grib_path is provided
                if cache and grib_path and values is not None:
                    try:
                        # Reshape flat values to (nb_cells, 69) for cache
                        cached_values = np.array(values, dtype=np.float32).reshape(len(cells_latlon), len(grib_params))

                        # Save each cell individually (for bbox-independent caching)
                        for cell_idx, (cell_lat, cell_lon) in enumerate(cells_latlon):
                            cell_values = cached_values[cell_idx:cell_idx+1, :]  # Shape (1, 69)
                            cache.save_cell(grib_path, cell_lat, cell_lon, cell_values)
                    except Exception as cache_err:
                        # Don't fail if cache save fails
                        print(f"[WARNING] Failed to cache {day_date} {hour}:00: {cache_err}")

        except Exception as e:
            print(f"[ERROR] {day_date} {hour}:00 - {type(e).__name__}: {str(e)[:100]}")
            hourly_queue.put((day_date, hour, None))
        finally:
            if grb_reader:
                del grb_reader
            # Cleanup temp file from RAM disk
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except:
                    pass
            gc.collect()


def assemble_day_results(hourly_queue, day_results_queue, num_params, num_cells):
    """
    Collect hourly results and assemble into day records.

    Args:
        hourly_queue: Queue with (day_date, hour, values) tuples
        day_results_queue: Queue where completed day_data is placed
        num_params: Number of parameters per hour (69: 65 base + 4 thermo)
        num_cells: Number of cells
    """
    hours_collected = {}  # (day_date, hour) -> values list
    days_seen = set()  # Track which days we've seen at least once

    while True:
        try:
            day_date, hour, values = hourly_queue.get(timeout=1)
        except Queue.Empty:
            continue

        if day_date is None:  # Sentinel to stop
            break

        days_seen.add(day_date)

        # Store hourly values (None = missing file)
        if values is None:
            hours_collected[(day_date, hour)] = [0.0] * (num_params * num_cells)
        else:
            # Validate that values has expected length; if not, pad/truncate
            expected_len = num_params * num_cells
            if len(values) < expected_len:
                # Pad with zeros if too short
                values = list(values) + [0.0] * (expected_len - len(values))
            elif len(values) > expected_len:
                # Truncate if too long (shouldn't happen, but safety)
                values = values[:expected_len]
            hours_collected[(day_date, hour)] = values

        # Check if we have all 3 hours for this day
        if (day_date, 6) in hours_collected and \
           (day_date, 12) in hours_collected and \
           (day_date, 18) in hours_collected:

            # Assemble day data: [cell1_values, cell2_values, ...]
            # where cell_values = [hour6_param1..69, hour12_param1..69, hour18_param1..69]
            day_data = []
            for cell_idx in range(num_cells):
                cell_values = []
                for hour in [6, 12, 18]:
                    start = cell_idx * num_params
                    end = start + num_params
                    hour_data = hours_collected[(day_date, hour)]
                    # Ensure hour_data has enough elements
                    if len(hour_data) >= end:
                        cell_values.extend(hour_data[start:end])
                    else:
                        # Pad with zeros if hour_data is too short
                        cell_values.extend(hour_data[start:] + [0.0] * (end - len(hour_data)))
                day_data.append(cell_values)

            day_results_queue.put((day_date, day_data))

            # Cleanup hourly data for this day (free memory)
            for h in [6, 12, 18]:
                del hours_collected[(day_date, h)]

    # Handle any incomplete days (missing hours) - create zero-filled data
    if hours_collected:
        import sys
        print(f"[WARNING] Assembler: {len(hours_collected)//3} day(s) incomplete due to missing hours", file=sys.stderr)
        for day_date in set(d for d, _ in hours_collected.keys()):
            # Create zero-filled day data for incomplete days
            day_data = [[0.0] * (num_params * 3) for _ in range(num_cells)]
            day_results_queue.put((day_date, day_data))
