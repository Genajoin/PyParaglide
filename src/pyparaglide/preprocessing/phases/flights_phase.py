"""
Phase 3: Process xContest flights and create flight PKL files.
"""

from datetime import date
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

from pyparaglide.preprocessing.flights.flight_processor import (
    FlightIndexer,
    load_flights_from_json,
    process_flights,
)
from pyparaglide.preprocessing.flights.elevation_reader import ElevationReader


class BuildFlightsPhase:
    """Phase 3: Process xContest flights and create flight PKL files."""

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
            min_flights: Minimum flights per spot (unused, kept for compatibility)
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

        # Save flights_by_cell_day.pkl
        import pickle
        import numpy as np

        nb_cells = len(self.cells_latlon)
        nb_days = len(self.meteo_days)

        flights_by_cell_day = np.zeros((nb_cells * nb_days,), dtype=object)
        for i in range(len(flights_by_cell_day)):
            flights_by_cell_day[i] = []

        # Populate flights_by_cell_day from processed results
        for flight in result['flights']:
            # flight is a dict with 'cell_index', 'day_index', and flight data
            cell_id = flight['cell_index']
            day_idx = flight['day_index']
            linear_idx = day_idx * nb_cells + cell_id
            # Convert dict to tuple format: (datetime, (score, lat, lon))
            flight_tuple = (
                flight['datetime'],
                (float(flight['score']) if flight['score'] else 0.0, flight['lat'], flight['lon'])
            )
            flights_by_cell_day[linear_idx].append(flight_tuple)

        with open(self.out_dir / "flights_by_cell_day.pkl", 'wb') as f:
            pickle.dump(flights_by_cell_day, f)

        print(f"  Created flights_by_cell_day.pkl")

        # NEW: Compute cell statistics
        print("\n  Computing cell statistics...")
        cell_flight_counts = np.zeros(nb_cells, dtype=np.int32)
        for linear_idx in range(len(flights_by_cell_day)):
            cell_id = linear_idx % nb_cells
            cell_flight_counts[cell_id] += len(flights_by_cell_day[linear_idx])

        # Save statistics for filtering pass
        stats_dict = {
            'total_flights_per_cell': cell_flight_counts.tolist(),
            'nb_cells_original': nb_cells,
            'nb_days': nb_days,
        }

        with open(self.out_dir / "cell_statistics.pkl", 'wb') as f:
            pickle.dump(stats_dict, f)

        print(f"\n  Cell statistics:")
        print(f"    Total cells: {nb_cells}")
        print(f"    Max flights/cell: {cell_flight_counts.max()}")
        print(f"    Min flights/cell: {cell_flight_counts.min()}")
        print(f"    Mean flights/cell: {cell_flight_counts.mean():.1f}")
