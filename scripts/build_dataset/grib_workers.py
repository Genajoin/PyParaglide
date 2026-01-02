"""
Worker functions for multiprocessing GRIB file processing.

3-stage hybrid pipeline:
- Reader Thread (HDD -> RAM disk)
- Worker Processes (RAM -> CPU)
- Assembler Thread (hourly results -> daily records)
"""

import gc
import os
import queue as Queue
import signal
import tempfile
import uuid
from typing import List, Tuple, Optional

import numpy as np


def file_reader(file_queue, job_queue, gfs_dir, stop_event):
    """
    Reader Thread (Main Process): SEQUENTIAL HDD I/O.
    Reads GRIB from HDD, writes copy to /dev/shm (RAM), puts PATH into job_queue.
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
            job_queue.put((day_date, hour, temp_path))

        except Exception as e:
            job_queue.put((day_date, hour, None))


def file_processor(job_queue, hourly_queue, grib_params, cells_latlon):
    """
    Worker Process (Separate CPU Core): CPU INTENSIVE.
    Reads from RAM disk, parses GRIB, extracts values.
    """
    # Re-import inside process
    from neural_network.inc.grib_reader import InMemoryGribReader

    # Ignore SIGINT in workers, let main process handle cleanup
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    while True:
        try:
            day_date, hour, temp_path = job_queue.get()
        except Exception:
            break

        if day_date is None:  # Sentinel
            break

        if temp_path is None:  # Missing file
            hourly_queue.put((day_date, hour, None))
            continue

        grb_reader = None
        try:
            # Parse with pygrib (CPU intensive, Parallel)
            grb_reader = InMemoryGribReader(temp_path)
            values = grb_reader.getValues(grib_params, cells_latlon)

            # Validate data length
            expected_len = len(grib_params) * len(cells_latlon)
            if values is None or len(values) != expected_len:
                hourly_queue.put((day_date, hour, None))
            else:
                hourly_queue.put((day_date, hour, values))

        except Exception as e:
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
        num_params: Number of parameters (65)
        num_cells: Number of cells
    """
    hours_collected = {}  # (day_date, hour) -> values list

    while True:
        try:
            day_date, hour, values = hourly_queue.get(timeout=1)
        except Queue.Empty:
            continue

        if day_date is None:  # Sentinel to stop
            break

        # Store hourly values (None = missing file)
        if values is None:
            hours_collected[(day_date, hour)] = [0.0] * (num_params * num_cells)
        else:
            hours_collected[(day_date, hour)] = values

        # Check if we have all 3 hours for this day
        if (day_date, 6) in hours_collected and \
           (day_date, 12) in hours_collected and \
           (day_date, 18) in hours_collected:

            # Assemble day data: [cell1_values, cell2_values, ...]
            # where cell_values = [hour6_param1..65, hour12_param1..65, hour18_param1..65]
            day_data = []
            for cell_idx in range(num_cells):
                cell_values = []
                for hour in [6, 12, 18]:
                    start = cell_idx * num_params
                    end = start + num_params
                    cell_values.extend(hours_collected[(day_date, hour)][start:end])
                day_data.append(cell_values)

            day_results_queue.put((day_date, day_data))

            # Cleanup hourly data for this day (free memory)
            for h in [6, 12, 18]:
                del hours_collected[(day_date, h)]
