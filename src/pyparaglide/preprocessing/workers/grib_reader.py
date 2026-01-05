"""
GRIB file reader wrapper for PyParaglide.

This module exports the GRIB reader classes from the inference module.
The classes are now implemented in src/pyparaglide/inference/grib_reader_v2.py
with proper type hints, logging, and error handling.

Classes:
    GribReader: Standard GRIB reader with pygrib index
    InMemoryGribReader: Loads entire GRIB file into memory for fast access

Both classes support extracting meteorological data from GFS GRIB files
for specific parameters and geographic locations.
"""

from pyparaglide.inference.grib_reader_v2 import GribReader, InMemoryGribReader, PYGRIB_AVAILABLE

__all__ = ["GribReader", "InMemoryGribReader", "PYGRIB_AVAILABLE"]
