"""
Dataset builder for PyParaglide.

Builds PKL files from GFS GRIB data and flight records.

This is the full implementation using the 5-phase pipeline:
1. BuildCellsPhase - Generate 1x1 degree cells and map to GRIB grid
2. BuildMeteoPhase - Scan GFS data and extract weather parameters
3. BuildFlightsPhase - Process xContest flights and create flight PKL files
4. BuildTerrainPhase - Extract mountainess data from elevation tiles
3.5. FilterCellsPhase - Filter data-sparse cells and reindex all PKL files

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
    FilterCellsPhase,
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
        min_flights_per_cell: int = 0,
        include_flights: bool = True,
        cluster_distance_km: Optional[float] = None,
        num_workers: int = 4,
        force: bool = False,
        use_cache: bool = True,
        rebuild_cache: bool = False,
    ) -> dict[str, Any]:
        """
        Build PKL dataset for multiple date ranges.

        CRITICAL FIX: This method accumulates data from ALL date ranges
        before saving, fixing the sequential overwrite bug.

        Args:
            date_ranges: List of (start_date, end_date) tuples
            min_flights_per_spot: Minimum flights required per spot
            min_flights_per_cell: Minimum flights per cell for training (0 = no filtering)
            include_flights: Whether to include flight data processing
            cluster_distance_km: Spot clustering radius in km
            num_workers: Number of multiprocessing workers for GRIB processing
            force: Force rebuild even if PKL files exist
            use_cache: Enable GRIB file caching (default: True)
            rebuild_cache: Force rebuild of GRIB cache

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
        if use_cache:
            print(f"  GRIB cache: enabled")
        else:
            print(f"  GRIB cache: disabled")
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
            use_cache=use_cache,
            rebuild_cache=rebuild_cache,
        )
        meteo_days = phase2.execute()

        # Phase 3: Process flights
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

        # Phase 4: Build terrain data (BEFORE Phase 3.5 to allow filtering)
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

        # Phase 3.5: Filter data-sparse cells (AFTER Phase 4)
        if min_flights_per_cell > 0 and include_flights:
            phase3_5 = FilterCellsPhase(
                out_dir=self.output_dir,
                min_flights_per_cell=min_flights_per_cell,
            )
            filter_result = phase3_5.execute()

            # Update cells_latlon after filtering
            cells_latlon = [cells_latlon[i] for i in filter_result['cells_kept']]

            print(f"\n  Filtered: {filter_result['nb_cells_after']}/{filter_result['nb_cells_before']} cells")
        else:
            if min_flights_per_cell == 0:
                print("\n=== Phase 3.5: Skipping cell filtering (min_flights_per_cell=0) ===")
            elif not include_flights:
                print("\n=== Phase 3.5: Skipping cell filtering (no flight data) ===")

        print()
        print("Dataset build complete!")
        print(f"  Cells: {len(cells_latlon)}")
        print(f"  Days: {len(meteo_days)}")

        return {
            "cells": len(cells_latlon),
            "days": len(meteo_days),
        }
