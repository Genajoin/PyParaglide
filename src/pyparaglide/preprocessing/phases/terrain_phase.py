"""
Phase 4: Extract mountainess data from elevation tiles.
"""

import struct
from datetime import date
from pathlib import Path
from typing import List, Tuple

import tqdm

from pyparaglide.config import get_settings
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

        # Check for GeoTIFF, download if missing
        geotiff_path = self.elevation_dir / "elevation.tif"
        use_geotiff = False

        if geotiff_path.exists():
            print("  Using GeoTIFF elevation data")
            use_geotiff = True
        else:
            settings = get_settings()
            if settings.elevation_auto_download:
                print("  Elevation GeoTIFF not found, downloading...")
                try:
                    from pyparaglide.downloads.elevation_downloader import ElevationDownloader

                    bbox = self._get_bbox_from_cells()
                    downloader = ElevationDownloader(
                        data_dir=self.elevation_dir,
                        bbox=bbox,
                        product=settings.elevation_source,  # SRTM1 or SRTM3
                    )
                    downloader.download()
                    use_geotiff = True
                except Exception as e:
                    print(f"  Warning: Failed to download elevation data: {e}")
                    print("  Falling back to legacy .mountainess files")
            else:
                print("  No GeoTIFF found, using legacy .mountainess files")

        # Build mountainess data
        mountainess_by_cell_alt = []

        if use_geotiff:
            # Use GeoTiffReader
            from pyparaglide.preprocessing.flights.geotiff_reader import GeoTiffReader

            reader = GeoTiffReader(self.elevation_dir)
            try:
                for lat, lon in tqdm.tqdm(self.cells_latlon, desc="  Processing cells"):
                    mountainess = reader.get_mountainess(lat, lon)
                    mountainess_by_cell_alt.append([mountainess] * 5)
            finally:
                reader.close()
        else:
            # Use legacy _get_mountainess
            for lat, lon in tqdm.tqdm(self.cells_latlon, desc="  Processing cells"):
                mountainess = self._get_mountainess(lat, lon)
                mountainess_by_cell_alt.append([mountainess] * 5)

        # Convert to numpy array for consistent shape handling
        import numpy as np
        mountainess_by_cell_alt = np.array(mountainess_by_cell_alt, dtype=np.float32)

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

    def _get_bbox_from_cells(self) -> Tuple[float, float, float, float]:
        """Calculate bounding box from cell coordinates."""
        if not self.cells_latlon:
            raise ValueError("No cells available")

        lats = [lat for lat, _ in self.cells_latlon]
        lons = [lon for _, lon in self.cells_latlon]

        # Add 0.5° padding for bbox (cover the cells)
        lat_min = min(lats) - 0.5
        lat_max = max(lats) + 0.5
        lon_min = min(lons) - 0.5
        lon_max = max(lons) + 0.5

        # Clamp to valid ranges
        lat_min = max(-90, min(90, lat_min))
        lat_max = max(-90, min(90, lat_max))
        lon_min = max(-180, min(180, lon_min))
        lon_max = max(-180, min(180, lon_max))

        return (lat_min, lat_max, lon_min, lon_max)
