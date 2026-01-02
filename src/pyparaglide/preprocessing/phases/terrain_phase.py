"""
Phase 4: Extract mountainess data from elevation tiles.
"""

import struct
from datetime import date
from pathlib import Path
from typing import List, Tuple

import tqdm

from pyparaglide.preprocessing.utils.bin_obj import BinObj
from pyparaglide.preprocessing.utils.metadata import load_metadata, save_metadata, check_config_match
from pyparaglide.preprocessing.utils.tiles_maths import TilesMaths


class BuildTerrainPhase:
    """Phase 4: Extract mountainess data from elevation tiles."""

    def __init__(self, elevation_dir: Path,
                 out_dir: Path,
                 cells_latlon: List[Tuple[float, float]],
                 force: bool = False):
        """
        Args:
            elevation_dir: Path to elevation tiles
            out_dir: Output directory for PKL files
            cells_latlon: List of cell coordinates
            force: Force rebuild even if PKL files exist
        """
        self.elevation_dir = elevation_dir
        self.out_dir = out_dir
        self.cells_latlon = cells_latlon
        self.nb_cells = len(cells_latlon)  # For metadata comparison
        self.force = force

    def execute(self) -> None:
        """Execute phase 4."""
        print("\n=== Phase 4: Building mountainess_by_cell_alt ===")

        # Check if PKL file already exists with matching config
        if not self.force and (self.out_dir / "mountainess_by_cell_alt.pkl").exists():
            saved_metadata = load_metadata(self.out_dir)
            current_config = {"nb_cells": self.nb_cells}

            if check_config_match(current_config, saved_metadata):
                print("  Found existing mountainess PKL file with matching cells, skipping")
                print("  Use --force to rebuild")
                return
            else:
                print(f"  Config mismatch: saved nb_cells = {saved_metadata.get('nb_cells')}, current = {self.nb_cells}")
                print("  Rebuilding...")

        mountainess_by_cell_alt = []

        for lat, lon in tqdm.tqdm(self.cells_latlon, desc="  Processing cells"):
            mountainess = self._get_mountainess(lat, lon)
            mountainess_by_cell_alt.append([mountainess] * 5)

        self._save_pkl("mountainess_by_cell_alt", mountainess_by_cell_alt)

        # Save metadata
        save_metadata(self.out_dir, {"nb_cells": self.nb_cells})

    def _save_pkl(self, name: str, data) -> None:
        """Save data to PKL file."""
        BinObj.save(data, name, path=str(self.out_dir))

    def _get_mountainess(self, lat: float, lon: float) -> float:
        """Get mountainess value for coordinates."""
        zoom = 7
        try:
            coords = TilesMaths.LatLonToTileCoords(zoom, lat, lon)
            filepath = self.elevation_dir / str(zoom) / str(coords['tx']) / f"{coords['ty']}.mountainess"

            if not filepath.exists():
                return 0.5

            with open(filepath, 'rb') as f:
                content = f.read(256 * 256)

            byte_idx = coords['x'] * 256 + coords['y']
            if byte_idx >= len(content):
                return 0.5

            value = struct.unpack('B', content[byte_idx:byte_idx+1])[0]
            return float(value) / 255.0

        except Exception:
            return 0.5
