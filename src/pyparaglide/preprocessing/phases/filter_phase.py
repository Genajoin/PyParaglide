"""
Phase 3.5: Filter data-sparse cells and reindex PKL files.

This phase removes cells with insufficient training data to reduce label noise
and improve model generalization.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np


def load_pkl(filename: str, pkl_dir: Path) -> Any:
    """Load PKL file."""
    with open(pkl_dir / filename, 'rb') as f:
        return pickle.load(f)


def save_pkl(filename: str, data: Any, pkl_dir: Path) -> None:
    """Save PKL file."""
    with open(pkl_dir / filename, 'wb') as f:
        pickle.dump(data, f)


class FilterCellsPhase:
    """Phase 3.5: Filter data-sparse cells and reindex all PKL files."""

    def __init__(self, out_dir: Path, min_flights_per_cell: int):
        """
        Initialize filter phase.

        Args:
            out_dir: Output directory containing PKL files
            min_flights_per_cell: Minimum number of flights required per cell
        """
        self.out_dir = out_dir
        self.min_flights_per_cell = min_flights_per_cell

    def execute(self) -> dict[str, Any]:
        """
        Filter cells with < min_flights_per_cell flights.

        Returns:
            {
                'cells_kept': list[int],  # Original indices of kept cells
                'nb_cells_before': int,
                'nb_cells_after': int,
                'cells_filtered_count': int,
            }

        Raises:
            FileNotFoundError: If cell_statistics.pkl doesn't exist
            ValueError: If min_flights_per_cell is negative
        """
        print("\n=== Phase 3.5: Filtering data-sparse cells ===")

        # Validate threshold
        if self.min_flights_per_cell < 0:
            raise ValueError(
                f"min_flights_per_cell must be >= 0, got {self.min_flights_per_cell}"
            )

        # Load cell statistics
        stats_path = self.out_dir / "cell_statistics.pkl"
        if not stats_path.exists():
            print(f"  ERROR: cell_statistics.pkl not found!")
            print(f"  Run 'pyparaglide build-dataset' to generate it.")
            raise FileNotFoundError(stats_path)

        with open(stats_path, 'rb') as f:
            stats = pickle.load(f)

        cell_flight_counts = np.array(stats['total_flights_per_cell'])
        nb_cells_old = stats['nb_cells_original']
        nb_days = stats['nb_days']

        print(f"  Threshold: {self.min_flights_per_cell} flights/cell")

        # Identify cells to keep
        cells_to_keep = np.where(cell_flight_counts >= self.min_flights_per_cell)[0]
        cells_filtered = np.where(cell_flight_counts < self.min_flights_per_cell)[0]

        # Edge case: all cells filtered
        if len(cells_to_keep) == 0:
            print(f"\n  WARNING: All {nb_cells_old} cells have < {self.min_flights_per_cell} flights!")
            print(f"  Consider lowering --min-cells threshold.")
            print(f"  Proceeding with all cells (no filtering applied).")
            cells_to_keep = np.arange(nb_cells_old)
            cells_filtered = np.array([], dtype=np.int32)

        print(f"\n  Filtering: {len(cells_to_keep)}/{nb_cells_old} cells kept")
        if len(cells_filtered) > 0:
            print(f"  Removed {len(cells_filtered)} cells with < {self.min_flights_per_cell} flights")

        # Reindex PKL files
        print(f"\n  Reindexing PKL files:")

        # 1. sorted_cells_latlon.pkl
        cells_latlon_old = load_pkl("sorted_cells_latlon.pkl", self.out_dir)
        cells_latlon_new = [cells_latlon_old[i] for i in cells_to_keep]
        save_pkl("sorted_cells_latlon.pkl", cells_latlon_new, self.out_dir)
        print(f"    ✓ sorted_cells_latlon.pkl: {len(cells_latlon_old)} → {len(cells_latlon_new)} cells")

        # 2. sorted_cells.pkl
        cells_grib_old = load_pkl("sorted_cells.pkl", self.out_dir)
        cells_grib_new = [cells_grib_old[i] for i in cells_to_keep]
        save_pkl("sorted_cells.pkl", cells_grib_new, self.out_dir)
        print(f"    ✓ sorted_cells.pkl: {len(cells_grib_old)} → {len(cells_grib_new)} cells")

        # 3. meteo_content_by_cell_day.pkl
        meteo_old = load_pkl("meteo_content_by_cell_day.pkl", self.out_dir)
        old_shape = meteo_old.shape

        # Reshape → Filter cells → Flatten
        meteo_reshaped = meteo_old.reshape(nb_days, nb_cells_old, -1)
        meteo_filtered = meteo_reshaped[:, cells_to_keep, :]
        meteo_new = meteo_filtered.reshape(-1, meteo_filtered.shape[2])

        save_pkl("meteo_content_by_cell_day.pkl", meteo_new, self.out_dir)
        print(f"    ✓ meteo_content_by_cell_day.pkl: {old_shape} → {meteo_new.shape}")

        # 4. flights_by_cell_day.pkl
        flights_old = load_pkl("flights_by_cell_day.pkl", self.out_dir)
        old_shape = flights_old.shape

        # Reshape → Filter → Flatten
        flights_reshaped = flights_old.reshape(nb_days, nb_cells_old)
        flights_filtered = flights_reshaped[:, cells_to_keep]
        flights_new = flights_filtered.flatten()

        save_pkl("flights_by_cell_day.pkl", flights_new, self.out_dir)
        print(f"    ✓ flights_by_cell_day.pkl: {old_shape} → {flights_new.shape}")

        # 5. mountainess_by_cell_alt.pkl (if exists)
        mountainess_path = self.out_dir / "mountainess_by_cell_alt.pkl"
        if mountainess_path.exists():
            mountainess_old = load_pkl("mountainess_by_cell_alt.pkl", self.out_dir)

            # Check if already filtered (from previous run)
            if mountainess_old.shape[0] == nb_cells_old:
                old_shape = mountainess_old.shape
                # Filter rows (cells dimension)
                mountainess_new = mountainess_old[cells_to_keep, :]
                save_pkl("mountainess_by_cell_alt.pkl", mountainess_new, self.out_dir)
                print(f"    ✓ mountainess_by_cell_alt.pkl: {old_shape} → {mountainess_new.shape}")
            else:
                print(f"    ⚠ mountainess_by_cell_alt.pkl: already filtered or mismatched, skipping")
        else:
            print(f"    ⚠ mountainess_by_cell_alt.pkl: not found, will be created in Phase 4")

        print(f"\n  Filtered dataset saved")

        return {
            'cells_kept': cells_to_keep.tolist(),
            'nb_cells_before': nb_cells_old,
            'nb_cells_after': len(cells_to_keep),
            'cells_filtered_count': len(cells_filtered),
        }
