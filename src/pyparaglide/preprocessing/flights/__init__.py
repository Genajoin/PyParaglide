"""
Flight data processing and elevation reading.

This module provides:
- ElevationReader: Reading elevation and mountainess from tiles
- GeoTiffReader: Reading elevation and mountainess from GeoTIFF files
- Flight processing: xContest JSON processing, spot clustering, PKL generation
"""

from .elevation_reader import ElevationReader
from .flight_processor import (
    FlightIndexer,
    load_flights_from_json,
    process_flights,
    filter_spots_by_flights,
    create_spots_pkls,
)
from .geotiff_reader import GeoTiffReader

__all__ = [
    "ElevationReader",
    "GeoTiffReader",
    "FlightIndexer",
    "load_flights_from_json",
    "process_flights",
    "filter_spots_by_flights",
    "create_spots_pkls",
]
