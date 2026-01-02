"""
Flight data processing and elevation reading.

This module provides:
- ElevationReader: Reading elevation and mountainess from tiles
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

__all__ = [
    "ElevationReader",
    "FlightIndexer",
    "load_flights_from_json",
    "process_flights",
    "filter_spots_by_flights",
    "create_spots_pkls",
]
