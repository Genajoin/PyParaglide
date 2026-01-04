"""
Elevation data downloader using SRTM data.

Downloads SRTM elevation tiles directly with progress bars.
SRTM provides 30m (SRTM1) or 90m (SRTM3) resolution global coverage.
"""

import json
import time
import zipfile
from pathlib import Path

import requests
import tqdm
import rasterio
from rasterio.merge import merge


class ElevationDownloader:
    """
    Download SRTM elevation data for a bounding box.

    Downloads SRTM tiles directly with progress bars and merges them.
    """

    # SRTM tile download URLs
    SRTM1_URL = "https://elevation-tiles-prod.s3.amazonaws.com/srtm/v1/3.0.0/{lat}/{lon}.tif"
    SRTM3_URL = "https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_{lat}_{lon}.zip"

    def __init__(
        self,
        data_dir: Path | str,
        bbox: tuple[float, float, float, float],
        product: str = "SRTM3",  # SRTM1 (30m) or SRTM3 (90m)
        max_retries: int = 3,
    ):
        """
        Initialize elevation downloader.

        Args:
            data_dir: Directory for elevation data storage
            bbox: Bounding box (lat_min, lat_max, lon_min, lon_max)
            product: SRTM1 (30m) or SRTM3 (90m)
            max_retries: Maximum download retry attempts
        """
        self.data_dir = Path(data_dir)
        self.bbox = bbox
        self.product = product
        self.max_retries = max_retries

        # Create directories
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Output paths
        self.output_path = self.data_dir / "elevation.tif"
        self.metadata_path = self.data_dir / "metadata.json"

    def download(self) -> dict:
        """
        Download SRTM elevation data for bbox.

        Returns:
            Dict with download statistics
        """
        stats = {
            "source": self.product,
            "bbox": self.bbox,
            "tiles_requested": 0,
            "tiles_downloaded": 0,
            "tiles_failed": 0,
            "failed_tiles": [],  # List of (lat, lon) tuples
            "output_size_mb": 0,
        }

        lat_min, lat_max, lon_min, lon_max = self.bbox

        print(f"\n=== Downloading {self.product} elevation data ===")
        print(f"  BBox: {lat_min:.2f}°N to {lat_max:.2f}°N, {lon_min:.2f}°E to {lon_max:.2f}°E")
        print(f"  Output: {self.output_path}")

        # Validate and clip bbox to SRTM coverage
        self._validate_coverage()

        # Get tiles needed for this bbox
        tiles = self._get_tiles_for_bbox()

        if not tiles:
            raise RuntimeError(f"No tiles found for bbox {self.bbox}")

        stats["tiles_requested"] = len(tiles)
        print(f"  Need to download {len(tiles)} tiles")

        # Download tiles
        tile_files = []
        for lat, lon in tqdm.tqdm(tiles, desc="  Downloading tiles"):
            tile_path = self._download_tile(lat, lon)
            if tile_path:
                tile_files.append(tile_path)
                stats["tiles_downloaded"] += 1
            else:
                stats["tiles_failed"] += 1
                stats["failed_tiles"].append((lat, lon))

        if not tile_files:
            raise RuntimeError("Failed to download any tiles")

        # Merge tiles into single GeoTIFF
        print(f"\n  Merging {len(tile_files)} tiles...")
        self._merge_tiles(tile_files, self.output_path)

        # Get file size
        if self.output_path.exists():
            stats["output_size_mb"] = self.output_path.stat().st_size / (1024 * 1024)

            # Print detailed summary
            print(f"\n  Download Summary:")
            print(f"    Requested:  {stats['tiles_requested']} tiles")
            print(f"    Downloaded: {stats['tiles_downloaded']} tiles")
            print(f"    Failed:     {stats['tiles_failed']} tiles")

            if stats['tiles_failed'] > 0 and stats['failed_tiles']:
                print(f"    Failed tiles:")
                for lat, lon in stats['failed_tiles'][:5]:  # Show first 5
                    print(f"      - lat {lat:+4d}, lon {lon:+4d}")
                if len(stats['failed_tiles']) > 5:
                    print(f"      ... and {len(stats['failed_tiles']) - 5} more")

            print(f"    Output:     {stats['output_size_mb']:.1f} MB")

            # Save metadata
            self._save_metadata(stats)

        print(f"\nElevation data ready!")

        return stats

    def _get_tiles_for_bbox(self) -> list[tuple[int, int]]:
        """Get SRTM tile coordinates for bounding box.

        Returns tile coordinates based on naming convention:
        - SRTM1: tiles are 1°x1°, URL format: {lat}/{lon}.tif
        - SRTM3: tiles are 5°x5°, CGIAR naming: srtm_{col}_{row}.zip
                 col/row are grid indices: col=(lon+185)//5, row=(lat+70)//5
        """
        lat_min, lat_max, lon_min, lon_max = self.bbox

        tiles = []

        # SRTM tiles are 1° x 1° for SRTM1, 5° x 5° for SRTM3
        if self.product == "SRTM1":
            tile_size = 1
            # SRTM1 uses lat/lon in URL: {lat}/{lon}.tif
            lat_start = int(lat_min // tile_size) * tile_size
            lon_start = int(lon_min // tile_size) * tile_size

            lat = lat_start
            while lat < lat_max:
                lon = lon_start
                while lon < lon_max:
                    tiles.append((lat, lon))
                    lon += tile_size
                lat += tile_size
        else:  # SRTM3
            tile_size = 5
            # CGIAR SRTM3 naming: srtm_{lat_min}_{lon_min}.zip
            # where lat_min and lon_min are the southwest corner coordinates
            lat_start = int(lat_min // tile_size) * tile_size
            lon_start = int(lon_min // tile_size) * tile_size

            lat = lat_start
            while lat < lat_max:
                lon = lon_start
                while lon < lon_max:
                    # For CGIAR URL: (lat_min, lon_min) - southwest corner
                    tiles.append((lat, lon))
                    lon += tile_size
                lat += tile_size

        return tiles

    def _cgiar_tile_indices(self, lat_south: int, lon_west: int) -> tuple[int, int]:
        """
        Calculate CGIAR SRTM tile grid indices from geographic coordinates.

        CGIAR tiles are 5°×5° numbered in a custom grid system:
        - lon_tile = (lon_west // 5) + 37
        - lat_tile = (60 - lat_south) // 5

        Formula verified against real CGIAR tiles:
        - srtm_39_03: lon 10-15°, lat 45-50° (Alps)
        - srtm_39_13: lon 10-15°, lat -5-0° (Africa)
        - srtm_45_11: lon 40-45°, lat 5-10° (East Africa)

        Args:
            lat_south: Southern edge of 5°×5° tile (e.g., 45 for 45-50°N)
            lon_west: Western edge of 5°×5° tile (e.g., 10 for 10-15°E)

        Returns:
            (lon_tile, lat_tile) tuple of CGIAR indices

        Examples:
            Alps (45°N, 10°E):         (39, 3)  → srtm_39_03.zip
            S. America (-30°S, -70°W): (23, 18) → srtm_23_18.zip
            Equator (0°, 0°):          (37, 12) → srtm_37_12.zip
        """
        lon_tile = (lon_west // 5) + 37
        lat_tile = (60 - lat_south) // 5
        return int(lon_tile), int(lat_tile)

    def _validate_coverage(self) -> None:
        """Validate bbox is within SRTM coverage and warn if outside."""
        lat_min, lat_max, lon_min, lon_max = self.bbox

        if lat_min < -60 or lat_max > 60:
            print(f"\n[WARNING] SRTM coverage limited to ±60° latitude")
            print(f"  Your bbox: {lat_min}° to {lat_max}°")
            print(f"  Data will be clipped to valid range")

        # Clip to valid range
        self.bbox = (
            max(lat_min, -60),
            min(lat_max, 60),
            lon_min,  # Longitude has full -180 to +180 coverage
            lon_max,
        )

    def _download_tile(self, lat: int, lon: int) -> Path | None:
        """
        Download single SRTM tile.

        Args:
            lat: Tile latitude (minimum/south edge)
            lon: Tile longitude (minimum/west edge)

        Returns:
            Path to downloaded tile or None if failed
        """
        if self.product == "SRTM1":
            # SRTM1 uses AWS S3 CDN with GeoTIFF
            url = f"https://elevation-tiles-prod.s3.amazonaws.com/srtm/v1/3.0.0/{lat}/{lon}.tif"
            output_path = self.cache_dir / f"srtm1_{lat}_{lon}.tif"
            return self._download_file(url, output_path)
        else:
            # SRTM3 uses CGIAR ZIP files
            # URL format: srtm_{col}_{row}.zip (grid indices, not coordinates)
            col, row = self._cgiar_tile_indices(lat, lon)
            url = f"https://srtm.csi.cgiar.org/wp-content/uploads/files/srtm_5x5/TIFF/srtm_{col:02d}_{row:02d}.zip"
            zip_path = self.cache_dir / f"srtm_{col:02d}_{row:02d}.zip"
            tif_path = self.cache_dir / f"srtm_{col:02d}_{row:02d}.tif"

            if tif_path.exists():
                return tif_path

            # Download ZIP
            if not self._download_file(url, zip_path):
                return None

            # Extract ZIP
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    # Find .tif file inside
                    for name in zf.namelist():
                        if name.endswith('.tif'):
                            zf.extract(name, self.cache_dir)
                            # Rename to consistent name
                            extracted = self.cache_dir / name
                            extracted.rename(tif_path)
                            break

                # Clean up ZIP
                zip_path.unlink()

                return tif_path
            except Exception as e:
                print(f"    Error extracting {zip_path}: {e}")
                return None

    def _download_file(self, url: str, output_path: Path) -> Path | None:
        """
        Download file with progress bar and retry logic.

        Args:
            url: Source URL
            output_path: Destination path

        Returns:
            Path to downloaded file or None if failed
        """
        if output_path.exists():
            return output_path

        # Extract tile name for logging
        tile_name = output_path.stem

        for attempt in range(self.max_retries):
            try:
                # Stream download with progress
                with requests.get(url, stream=True, timeout=30) as r:
                    # Handle 404 separately - don't retry
                    if r.status_code == 404:
                        if attempt == 0:  # Only log once
                            print(f"    {tile_name}: not available (404)")
                        return None

                    r.raise_for_status()
                    total_size = int(r.headers.get('content-length', 0))

                    with open(output_path, 'wb') as f:
                        with tqdm.tqdm(
                            total=total_size,
                            unit='B',
                            unit_scale=True,
                            desc=f"    {tile_name}",
                            leave=False,
                        ) as pbar:
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))

                    return output_path

            except requests.exceptions.Timeout:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    print(f"    {tile_name}: timeout, retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"    {tile_name}: failed after {self.max_retries} timeouts")
                    return None

            except requests.exceptions.RequestException as e:
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"    {tile_name}: error ({e}), retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    print(f"    {tile_name}: failed after {self.max_retries} attempts: {e}")
                    return None

            except Exception as e:
                print(f"    {tile_name}: unexpected error: {e}")
                return None

        return None

    def _merge_tiles(self, tile_files: list[Path], output_path: Path) -> None:
        """
        Merge multiple GeoTIFF tiles into single file.

        Args:
            tile_files: List of tile paths
            output_path: Output merged file path
        """
        # Use rasterio.merge to merge tiles
        datasets = [rasterio.open(f) for f in tile_files]

        # Merge tiles
        merged, out_transform = merge(datasets)

        # Get output metadata from first dataset
        src = datasets[0]
        out_meta = src.meta.copy()

        # Update transform and size
        out_meta.update({
            'transform': out_transform,
            'width': merged.shape[2],
            'height': merged.shape[1],
        })

        # Write merged file
        with rasterio.open(output_path, 'w', **out_meta) as dst:
            dst.write(merged)

        # Close datasets
        for ds in datasets:
            ds.close()

    def _save_metadata(self, stats: dict) -> None:
        """Save metadata about downloaded data."""
        metadata = {
            "source": self.product,
            "bbox": list(self.bbox),
            "tiles_downloaded": stats["tiles_downloaded"],
            "output_size_mb": stats["output_size_mb"],
            "output_path": str(self.output_path),
        }

        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
