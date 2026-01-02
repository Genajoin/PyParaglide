"""
Phase 1: Build cell lists from bbox.
"""

import os
import pickle
from pathlib import Path
from typing import List, Tuple, Optional

from pyparaglide.preprocessing.utils.bin_obj import BinObj
from pyparaglide.preprocessing.utils.metadata import load_metadata, save_metadata, check_config_match
from pyparaglide.preprocessing.workers.grib_reader import GribReader


class BuildCellsPhase:
    """Phase 1: Build cell lists from bbox."""

    def __init__(self, bbox: Tuple[float, float, float, float],
                 gfs_dir: Path,
                 out_dir: Path,
                 force: bool = False):
        """
        Args:
            bbox: (lat_min, lat_max, lon_min, lon_max)
            gfs_dir: Path to GFS GRIB files
            out_dir: Output directory for PKL files
            force: Force rebuild even if PKL files exist
        """
        self.bbox = bbox
        self.bbox_list = list(bbox)  # For metadata comparison
        self.gfs_dir = gfs_dir
        self.out_dir = out_dir
        self.force = force

    def execute(self) -> Tuple[List[Tuple[float, float]], List[Tuple[int, int]]]:
        """
        Execute phase 1.

        Returns:
            (cells_latlon, cells_grib) - List of (lat, lon) and list of (row, col)
        """
        print("\n=== Phase 1: Building cells ===")

        # Check if PKL files already exist with matching config
        if not self.force and self._check_existing_pkl():
            saved_metadata = load_metadata(self.out_dir)
            current_config = {"bbox": self.bbox_list}

            if check_config_match(current_config, saved_metadata):
                print("  Found existing cells PKL files with matching bbox, skipping")
                print("  Use --force to rebuild")
                try:
                    with open(self.out_dir / "sorted_cells_latlon.pkl", 'rb') as f:
                        cells_latlon = pickle.load(f, encoding='latin1')
                    with open(self.out_dir / "sorted_cells.pkl", 'rb') as f:
                        cells_grib = pickle.load(f, encoding='latin1')
                    return cells_latlon, cells_grib
                except Exception as e:
                    print(f"  WARNING: Could not load existing PKL: {e}")
                    print("  Rebuilding...")
            else:
                print(f"  Config mismatch: saved bbox = {saved_metadata.get('bbox')}")
                print("  Rebuilding with new bbox...")

        lat_min, lat_max, lon_min, lon_max = self.bbox

        # Generate 1x1 degree cells
        cells_latlon = []
        for lat in range(int(lat_min), int(lat_max) + 1):
            for lon in range(int(lon_min), int(lon_max) + 1):
                cells_latlon.append((float(lat), float(lon)))

        nb_cells = len(cells_latlon)
        print(f"  Generated {nb_cells} cells")
        print(f"  Lat: {lat_min}°..{lat_max}°, Lon: {lon_min}°..{lon_max}°")

        # Save sorted_cells_latlon.pkl
        self._save_pkl("sorted_cells_latlon", cells_latlon)

        # Map to GRIB grid
        print("  Mapping cells to GRIB grid...")
        cells_grib = self._map_cells_to_grib(cells_latlon)

        if cells_grib:
            self._save_pkl("sorted_cells", cells_grib)
        else:
            print("  WARNING: Could not map to GRIB grid")
            cells_grib = [(0, 0)] * nb_cells
            self._save_pkl("sorted_cells", cells_grib)

        # Save metadata
        save_metadata(self.out_dir, {"bbox": self.bbox_list})

        return cells_latlon, cells_grib

    def _save_pkl(self, name: str, data) -> None:
        """Save data to PKL file."""
        print(f"  Saving {name}.pkl...", end=" ")
        BinObj.save(data, name, path=str(self.out_dir))
        print("OK")

    def _map_cells_to_grib(self, cells_latlon: List[Tuple[float, float]]) -> Optional[List[Tuple[int, int]]]:
        """Map cell coordinates to GRIB grid indices."""
        if GribReader is None:
            return None

        sample_grib = self._find_sample_grib()
        if not sample_grib:
            return None

        try:
            grb_reader = GribReader(sample_grib)
            _, lats, lons = grb_reader.getInfos()

            cells_grib = []
            for lat, lon in cells_latlon:
                row = GribReader.findClosest(lat, lats, 0)
                col = GribReader.findClosest(lon, lons, 1)
                cells_grib.append((int(row), int(col)))

            return cells_grib

        except Exception as e:
            print(f"  WARNING: Failed to map GRIB indices: {e}")
            return None

    def _find_sample_grib(self) -> Optional[str]:
        """Find first available GRIB file."""
        if not self.gfs_dir.exists():
            return None

        for root, _, files in os.walk(self.gfs_dir):
            for file in files:
                if file.endswith(('.grb', '.grb2', '.grib', '.grib2')):
                    return os.path.join(root, file)

        return None

    def _check_existing_pkl(self) -> bool:
        """Check if cells PKL files exist."""
        return ((self.out_dir / "sorted_cells_latlon.pkl").exists() and
                (self.out_dir / "sorted_cells.pkl").exists())
