"""
Multiprocessing workers for GRIB file processing.

This module provides worker functions for parallel GRIB processing:
- GribReader: Wrapper for neural_network GRIB reader
- file_reader: Thread for reading GRIB files from HDD
- file_processor: Process for CPU-intensive GRIB parsing
- assemble_day_results: Thread for assembling hourly results into daily records
"""

# Try to import from neural_network, but make it optional
try:
    from pyparaglide.preprocessing.workers.grib_reader import GribReader, InMemoryGribReader
except ImportError:
    GribReader = None
    InMemoryGribReader = None

from pyparaglide.preprocessing.workers.grib_workers import (
    file_reader,
    file_processor,
    assemble_day_results,
)

__all__ = [
    "GribReader",
    "InMemoryGribReader",
    "file_reader",
    "file_processor",
    "assemble_day_results",
]
