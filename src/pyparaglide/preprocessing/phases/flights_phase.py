"""
Phase 3: Process xContest flights and create spot PKL files.
"""

from datetime import date
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from pyparaglide.preprocessing.flights.flight_processor import (
    FlightIndexer,
    load_flights_from_json,
    process_flights,
    filter_spots_by_flights,
    create_spots_pkls,
)
from pyparaglide.preprocessing.flights.elevation_reader import ElevationReader


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
