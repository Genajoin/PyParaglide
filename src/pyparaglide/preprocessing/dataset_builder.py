"""
Dataset builder for PyParaglide.

Builds PKL files from GFS GRIB data and flight records.

This is the full implementation using the 4-phase pipeline:
1. BuildCellsPhase - Generate 1x1 degree cells and map to GRIB grid
2. BuildMeteoPhase - Scan GFS data and extract weather parameters
3. BuildFlightsPhase - Process xContest flights and create spot PKL files
4. BuildTerrainPhase - Extract mountainess data from elevation tiles

CRITICAL FIX: build_all() method accepts multiple date ranges and accumulates
ALL data before saving, fixing the sequential overwrite bug.
"""

import datetime as dt
from datetime import date
from pathlib import Path
from typing import Any, List, Tuple, Optional

from pyparaglide.preprocessing.phases import (
    BuildCellsPhase,
    BuildMeteoPhase,
    BuildFlightsPhase,
    BuildTerrainPhase,
)


class DatasetBuilder:
    """
    Build training dataset from GFS GRIB files and flight data.

    This class orchestrates the 4-phase dataset building pipeline.
    """

    def __init__(
        self,
        gfs_dir: Path | str,
        flights_dir: Path | str,
        output_dir: Path | str,
        bbox: tuple[float, float, float, float] | None = None,
        elevation_dir: Path | str | None = None,
    ):
        """
        Initialize dataset builder.

        Args:
            gfs_dir: Directory containing GFS GRIB files
            flights_dir: Directory containing xContest JSON files
            output_dir: Output directory for PKL files
            bbox: Bounding box (lat_min, lat_max, lon_min, lon_max)
            elevation_dir: Directory containing elevation tiles
        """
        self.gfs_dir = Path(gfs_dir)
        self.flights_dir = Path(flights_dir)
        self.output_dir = Path(output_dir)
        self.bbox = bbox or (45.0, 47.0, 13.0, 15.0)  # Alps region
        self.elevation_dir = Path(elevation_dir) if elevation_dir else None

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def build(
        self,
        start_date: dt.date | str,
        end_date: dt.date | str,
        min_flights_per_spot: int = 200,
        include_flights: bool = True,
    ) -> dict[str, Any]:
        """
        Build PKL dataset for a single date range.

        DEPRECATED: This method is kept for backward compatibility.
        Use build_all() instead to avoid sequential overwrite issues.

        Args:
            start_date: Start date
            end_date: End date
            min_flights_per_spot: Minimum flights required per spot
            include_flights: Whether to include flight data processing

        Returns:
            Dictionary with build statistics
        """
        if isinstance(start_date, str):
            start_date = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
        if isinstance(end_date, str):
            end_date = dt.datetime.strptime(end_date, "%Y-%m-%d").date()

        return self.build_all(
            date_ranges=[(start_date, end_date)],
            min_flights_per_spot=min_flights_per_spot,
            include_flights=include_flights,
        )

    def build_all(
        self,
        date_ranges: List[Tuple[date, date]],
        min_flights_per_spot: int = 200,
        include_flights: bool = True,
        cluster_distance_km: Optional[float] = None,
        num_workers: int = 4,
        force: bool = False,
    ) -> dict[str, Any]:
        """
        Build PKL dataset for multiple date ranges.

        CRITICAL FIX: This method accumulates data from ALL date ranges
        before saving, fixing the sequential overwrite bug.

        Args:
            date_ranges: List of (start_date, end_date) tuples
            min_flights_per_spot: Minimum flights required per spot
            include_flights: Whether to include flight data processing
            cluster_distance_km: Spot clustering radius in km
            num_workers: Number of multiprocessing workers for GRIB processing
            force: Force rebuild even if PKL files exist

        Returns:
            Dictionary with build statistics
        """
        print(f"Building dataset:")
        print(f"  Date ranges: {len(date_ranges)} range(s)")
        for start, end in date_ranges:
            print(f"    {start} to {end}")
        print(f"  Bbox: {self.bbox}")
        print(f"  Output: {self.output_dir}")
        print(f"  Workers: {num_workers}")
        print()

        # Phase 1: Build cells
        phase1 = BuildCellsPhase(
            bbox=self.bbox,
            gfs_dir=self.gfs_dir,
            out_dir=self.output_dir,
            force=force,
        )
        cells_latlon, cells_grib = phase1.execute()

        # Phase 2: Build meteo data - CRITICAL: Pass ALL date_ranges at once
        phase2 = BuildMeteoPhase(
            bbox=self.bbox,
            gfs_dir=self.gfs_dir,
            cells_latlon=cells_latlon,
            out_dir=self.output_dir,
            date_ranges=date_ranges,  # ALL ranges, accumulated
            num_workers=num_workers,
            force=force,
        )
        meteo_days = phase2.execute()

        # Phase 3: Process flights
        spots_count = 0
        if include_flights:
            phase3 = BuildFlightsPhase(
                flights_dir=self.flights_dir,
                out_dir=self.output_dir,
                cells_latlon=cells_latlon,
                meteo_days=meteo_days,
                bbox=self.bbox,
                date_ranges=date_ranges,  # ALL ranges, accumulated
                min_flights=min_flights_per_spot,
                cluster_distance_km=cluster_distance_km,
            )
            phase3.execute()
            # Count spots from spots.pkl
            import pickle
            try:
                with open(self.output_dir / "spots.pkl", 'rb') as f:
                    spots = pickle.load(f)
                    spots_count = len(spots)
            except:
                spots_count = 0
        else:
            print("\n=== Phase 3: Skipping flight processing (--no-flights) ===")
            # Create empty flights PKL
            import pickle
            import numpy as np
            nb_cells = len(cells_latlon)
            nb_days = len(meteo_days)
            flights_by_cell_day = np.zeros((nb_cells * nb_days,), dtype=object)
            for i in range(len(flights_by_cell_day)):
                flights_by_cell_day[i] = []
            with open(self.output_dir / "flights_by_cell_day.pkl", 'wb') as f:
                pickle.dump(flights_by_cell_day, f)

        # Phase 4: Build terrain data
        if self.elevation_dir:
            phase4 = BuildTerrainPhase(
                elevation_dir=self.elevation_dir,
                out_dir=self.output_dir,
                cells_latlon=cells_latlon,
                force=force,
            )
            phase4.execute()
        else:
            print("\n=== Phase 4: Skipping terrain (no elevation dir) ===")
            # Create default mountainess
            import pickle
            import numpy as np
            nb_cells = len(cells_latlon)
            mountainess_by_cell_alt = np.zeros((nb_cells, 5), dtype=np.float32)
            with open(self.output_dir / "mountainess_by_cell_alt.pkl", 'wb') as f:
                pickle.dump(mountainess_by_cell_alt, f)

        print()
        print("Dataset build complete!")
        print(f"  Cells: {len(cells_latlon)}")
        print(f"  Spots: {spots_count}")
        print(f"  Days: {len(meteo_days)}")

        return {
            "cells": len(cells_latlon),
            "spots": spots_count,
            "days": len(meteo_days),
        }
