#!/usr/bin/env python3
"""
Build PKL dataset for neural network training from flights database and GFS data.

This script creates all required PKL files for neural network training:
1. sorted_cells_latlon.pkl - cell coordinates (lat, lon)
2. sorted_cells.pkl - GRIB grid indices (row, col)
3. meteo_days.pkl - list of days with weather data
4. meteo_params.pkl - list of weather parameters
5. meteo_content_by_cell_day.pkl - weather matrix [days*cells, 195]
6. flights_by_cell_day.pkl - flights data by cell and day
7. mountainess_by_cell_alt.pkl - terrain data by cell and altitude

Flight scoring:
- Uses xc_score (XC score from parse_igc_with_libs.py) as primary score metric
- XC score includes FAI triangles, free distance with multipliers (1.0-1.6)
- Falls back to distance_km if xc_score is not available
- Score threshold of 60 is used for crossability in neural network training
"""

import argparse
import math
import os
import struct
import sys
from datetime import datetime, date
from typing import List, Tuple, Optional, Dict

# Add neural_network path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'neural_network'))
sys.path.insert(0, os.path.dirname(__file__))

try:
    import numpy as np
    import pickle
    import tqdm
except ImportError as e:
    print(f"ERROR: Required module not found: {e}", file=sys.stderr)
    print("Install with: pip install numpy tqdm", file=sys.stderr)
    sys.exit(1)

try:
    import pygrib
except ImportError:
    print("WARNING: pygrib not installed. GFS processing will not work.", file=sys.stderr)
    print("Install with: pip install pygrib", file=sys.stderr)
    pygrib = None

# Import from neural_network
try:
    from inc.bin_obj import BinObj
    from inc.tiles_maths import TilesMaths
except ImportError as e:
    print(f"ERROR: Failed to import neural_network modules: {e}", file=sys.stderr)
    print("Make sure you're running from project root directory", file=sys.stderr)
    sys.exit(1)

# Optional imports for GFS processing
try:
    from inc.grib_reader import GribReader
    from inc.dataset import GfsData
    HAS_GRIB = True
except ImportError:
    HAS_GRIB = False
    GribReader = None
    GfsData = None

# Import DB connection
try:
    from igc_ingest_skygr import connect_db, Db
except ImportError:
    print("ERROR: Failed to import igc_ingest_skygr", file=sys.stderr)
    sys.exit(1)


class PKLDatasetBuilder:
    """Builder for PKL dataset files."""

    def __init__(
        self,
        bbox: Tuple[float, float, float, float],
        gfs_dir: str,
        elevation_dir: str,
        out_dir: str,
        db: Optional[Db] = None
    ):
        """
        Initialize dataset builder.

        Args:
            bbox: (lat_min, lat_max, lon_min, lon_max)
            gfs_dir: path to GFS GRIB files
            elevation_dir: path to elevation tiles (tiler/_cache/elevation)
            out_dir: output directory for PKL files
            db: database connection (optional, needed for flights data)
        """
        self.bbox = bbox
        self.gfs_dir = gfs_dir
        self.elevation_dir = elevation_dir
        self.out_dir = out_dir
        self.db = db

        # Will be populated
        self.cells_latlon = []
        self.cells_grib = []
        self.meteo_days = []
        self.nb_cells = 0
        self.nb_days = 0

        # Ensure output directory exists
        os.makedirs(out_dir, exist_ok=True)

    def save_pkl(self, name: str, data) -> None:
        """Save data to PKL file using BinObj."""
        print(f"  Saving {name}.pkl...", flush=True)
        BinObj.save(data, name, path=self.out_dir)

    def build_cells(self) -> None:
        """
        Step 1: Build cell lists from bbox.

        Creates:
        - sorted_cells_latlon.pkl: List[(lat, lon)]
        - sorted_cells.pkl: List[(row, col)] in GRIB grid
        """
        print("\n=== Step 1: Building cells ===", flush=True)

        lat_min, lat_max, lon_min, lon_max = self.bbox

        # Generate cells (1°x1° grid)
        cells_latlon = []
        for lat in range(int(lat_min), int(lat_max)):
            for lon in range(int(lon_min), int(lon_max)):
                cells_latlon.append((float(lat), float(lon)))

        self.cells_latlon = cells_latlon
        self.nb_cells = len(cells_latlon)

        print(f"  Generated {self.nb_cells} cells", flush=True)
        print(f"  Lat range: {lat_min}°..{lat_max}°", flush=True)
        print(f"  Lon range: {lon_min}°..{lon_max}°", flush=True)

        # Save sorted_cells_latlon.pkl
        self.save_pkl("sorted_cells_latlon", cells_latlon)

        # Map to GRIB indices (requires sample GRIB file)
        print("  Mapping cells to GRIB grid...", flush=True)
        cells_grib = self.map_cells_to_grib(cells_latlon)

        if cells_grib:
            self.cells_grib = cells_grib
            self.save_pkl("sorted_cells", cells_grib)
        else:
            print("  WARNING: Could not map to GRIB grid (no sample file)", flush=True)
            print("  sorted_cells.pkl will be empty", flush=True)
            self.cells_grib = [(0, 0)] * self.nb_cells  # Placeholder
            self.save_pkl("sorted_cells", self.cells_grib)

    def map_cells_to_grib(self, cells_latlon: List[Tuple[float, float]]) -> Optional[List[Tuple[int, int]]]:
        """
        Map cell coordinates to GRIB grid indices.

        Args:
            cells_latlon: List of (lat, lon) tuples

        Returns:
            List of (row, col) tuples in GRIB grid, or None if failed
        """
        if not HAS_GRIB or not pygrib:
            return None

        # Find first available GRIB file
        sample_grib = self.find_sample_grib()
        if not sample_grib:
            return None

        try:
            # Read grid structure
            grb_reader = GribReader(sample_grib)
            valid_date, lats, lons = grb_reader.getInfos()

            # Map each cell
            cells_grib = []
            for lat, lon in cells_latlon:
                row = GribReader.findClosest(lat, lats, 0)
                col = GribReader.findClosest(lon, lons, 1)
                cells_grib.append((int(row), int(col)))

            return cells_grib

        except Exception as e:
            print(f"  WARNING: Failed to map GRIB indices: {e}", flush=True)
            return None

    def find_sample_grib(self) -> Optional[str]:
        """Find first available GRIB file for grid structure."""
        if not os.path.exists(self.gfs_dir):
            return None

        # Look for GRIB files
        for root, dirs, files in os.walk(self.gfs_dir):
            for file in files:
                if file.endswith(('.grb', '.grb2', '.grib', '.grib2')):
                    return os.path.join(root, file)

        return None

    def build_meteo_days(self, date_range: Optional[Tuple[date, date]] = None) -> None:
        """
        Step 2: Scan GFS directory for available days.

        Creates:
        - meteo_days.pkl: List[datetime.date]
        - meteo_params.pkl: List[(hour, param_name, level_list)]

        Args:
            date_range: Optional (start_date, end_date) to limit scan
        """
        print("\n=== Step 2: Scanning GFS data ===", flush=True)

        if not os.path.exists(self.gfs_dir):
            print(f"  ERROR: GFS directory not found: {self.gfs_dir}", flush=True)
            sys.exit(1)

        # Scan for available days with complete GFS data (6, 12, 18 UTC)
        meteo_days = []

        # Scan all month directories
        if os.path.exists(self.gfs_dir):
            for month_dir in sorted(os.listdir(self.gfs_dir)):
                month_path = os.path.join(self.gfs_dir, month_dir)
                if not os.path.isdir(month_path):
                    continue

                # Parse month (YYYY-MM format)
                try:
                    year, month = map(int, month_dir.split('-'))
                except:
                    continue

                # Group files by date
                files_by_date = {}
                for filename in os.listdir(month_path):
                    if filename.startswith('gfsanl_3_') and filename.endswith('.grb2'):
                        # Extract date from filename: gfsanl_3_YYYYMMDD_HH00_000.grb2
                        parts = filename.split('_')
                        if len(parts) >= 4:
                            date_str = parts[2]  # YYYYMMDD
                            hour_str = parts[3][:2]  # HH

                            if date_str not in files_by_date:
                                files_by_date[date_str] = set()
                            files_by_date[date_str].add(hour_str)

                # Check which dates have all required hours (6, 12, 18)
                for date_str, hours in sorted(files_by_date.items()):
                    if '06' in hours and '12' in hours and '18' in hours:
                        # Parse date
                        year_val = int(date_str[:4])
                        month_val = int(date_str[4:6])
                        day_val = int(date_str[6:8])
                        meteo_days.append(date(year_val, month_val, day_val))

        print(f"  Found {len(meteo_days)} days with complete GFS data", flush=True)

        self.meteo_days = meteo_days
        self.nb_days = len(meteo_days)
        self.save_pkl("meteo_days", meteo_days)

        # Build meteo_params (fixed list from GfsData)
        print("  Building meteo_params...", flush=True)
        meteo_params = self.build_meteo_params()
        self.save_pkl("meteo_params", meteo_params)

    def build_meteo_params(self) -> List:
        """
        Build list of weather parameters.

        Returns:
            List of (hour, param_name, level_list) tuples
        """
        if not HAS_GRIB:
            print("  WARNING: GfsData not available, using hardcoded params", flush=True)

        # Use GfsData to get parameter structure (or hardcode if not available)

        meteo_params = []

        # Build parameters for 3 time steps (06, 12, 18 UTC)
        for hour in [6, 12, 18]:
            # For each parameter type
            for param_name, levels in [
                ('Precipitable water', [[('entireAtmosphere', 0), ('unknown', 0)]]),
                ('Cloud water', [[('entireAtmosphere', 0), ('unknown', 0)]]),
                ('Vertical velocity', [[('isobaricInhPa', p)] for p in [1000, 900, 800, 700, 600, 500, 400, 300, 200]]),
                ('Geopotential Height', [[('isobaricInhPa', p)] for p in [1000, 900, 800, 700, 600, 500, 400, 300, 200]]),
                ('Absolute vorticity', [[('isobaricInhPa', p)] for p in [1000, 900, 800, 700, 600, 500, 400, 300, 200]]),
                ('Temperature', [[('isobaricInhPa', p)] for p in [1000, 900, 800, 700, 600, 500, 400, 300, 200]]),
                ('Relative humidity', [[('isobaricInhPa', p)] for p in [1000, 900, 800, 700, 600, 500, 400, 300, 200]]),
                ('U component of wind', [[('isobaricInhPa', p)] for p in [1000, 900, 800, 700, 600, 500, 400, 300, 200]]),
                ('V component of wind', [[('isobaricInhPa', p)] for p in [1000, 900, 800, 700, 600, 500, 400, 300, 200]]),
            ]:
                for level_list in levels:
                    meteo_params.append((hour, param_name, level_list))

        print(f"    Total parameters: {len(meteo_params)}", flush=True)
        return meteo_params

    def build_meteo_content(self) -> None:
        """
        Step 3: Extract weather data from GRIB files.

        Creates:
        - meteo_content_by_cell_day.pkl: np.array[nb_days*nb_cells, 195]
        """
        print("\n=== Step 3: Extracting weather data ===", flush=True)

        if not self.meteo_days:
            print("  WARNING: No meteo days found. Creating empty matrix.", flush=True)
            matrix = np.zeros((0, 195), dtype=np.float32)
            self.save_pkl("meteo_content_by_cell_day", matrix)
            return

        # TODO: Implement actual GRIB extraction
        # For now, create placeholder
        matrix = np.zeros((self.nb_days * self.nb_cells, 195), dtype=np.float32)

        print(f"  Matrix shape: {matrix.shape}", flush=True)
        self.save_pkl("meteo_content_by_cell_day", matrix)

    def build_flights_by_cell_day(self, source: str = "skygr") -> None:
        """
        Step 4: Build flights_by_cell_day.pkl from database.

        Creates:
        - flights_by_cell_day.pkl: List[List[Tuple]]

        Args:
            source: flight source to query
        """
        print("\n=== Step 4: Building flights_by_cell_day ===", flush=True)

        if not self.db:
            print("  ERROR: Database connection required", flush=True)
            sys.exit(1)

        if not self.meteo_days:
            print("  WARNING: No meteo days. Creating empty structure.", flush=True)
            flights_by_cell_day = []
            self.save_pkl("flights_by_cell_day", flights_by_cell_day)
            return

        # Initialize structure
        flights_by_cell_day = [[] for _ in range(self.nb_days * self.nb_cells)]

        # Build day index
        day_to_idx = {day: idx for idx, day in enumerate(self.meteo_days)}

        # Cell mapping function
        def get_cell_idx(lat: float, lon: float) -> Optional[int]:
            cell_lat = int(math.floor(lat))
            cell_lon = int(math.floor(lon))
            try:
                return self.cells_latlon.index((float(cell_lat), float(cell_lon)))
            except ValueError:
                return None

        # Query flights from database
        query = """
            SELECT
                flight_date,
                takeoff_datetime,
                takeoff_lat,
                takeoff_lon,
                takeoff_alt,
                max_alt,
                plaf,
                xc_score,
                distance_km
            FROM flights
            WHERE
                source=?
                AND status IN ('parsed', 'downloaded')
                AND takeoff_lat IS NOT NULL
                AND takeoff_lon IS NOT NULL
                AND flight_date IS NOT NULL
                AND (xc_score IS NOT NULL OR distance_km IS NOT NULL)
            ORDER BY flight_date
        """

        print("  Querying flights from database...", flush=True)
        cursor = self.db.execute(query, (source,))
        rows = cursor.fetchall()

        print(f"  Processing {len(rows)} flights...", flush=True)

        processed = 0
        skipped_date = 0
        skipped_bbox = 0

        for row in tqdm.tqdm(rows, desc="  Processing"):
            flight_date_str = row[0]
            takeoff_datetime = row[1]
            lat = row[2]
            lon = row[3]
            takeoff_alt = row[4] or 0.0
            max_alt = row[5] or 0.0
            plaf = row[6] or 0.0
            xc_score = row[7]
            distance_km = row[8]

            # Parse date
            try:
                flight_date = datetime.strptime(flight_date_str, '%Y-%m-%d').date()
            except:
                continue

            # Check if date in meteo_days
            if flight_date not in day_to_idx:
                skipped_date += 1
                continue

            day_idx = day_to_idx[flight_date]

            # Determine cell
            cell_idx = get_cell_idx(lat, lon)
            if cell_idx is None:
                skipped_bbox += 1
                continue

            # Get mountainess
            mountainess = self.get_mountainess(lat, lon)

            # Determine score (priority: xc_score from parse_igc_with_libs.py > distance_km fallback)
            score = xc_score if xc_score else distance_km if distance_km else 0.0

            # Build flight record
            # Format: (datetime_str, (score, alt, plaf, lat, lon, takeoff_alt, mountainess))
            flight_record = (
                takeoff_datetime or f"{flight_date_str} 12:00:00",
                (
                    float(score),
                    float(max_alt),  # alt field (not used in current code)
                    float(plaf),
                    float(lat),
                    float(lon),
                    float(takeoff_alt),
                    float(mountainess)
                )
            )

            # Add to structure
            idx = day_idx * self.nb_cells + cell_idx
            flights_by_cell_day[idx].append(flight_record)
            processed += 1

        print(f"  Successfully processed: {processed} flights", flush=True)
        if skipped_date > 0:
            print(f"  Skipped (not in meteo_days): {skipped_date} flights", flush=True)
        if skipped_bbox > 0:
            print(f"  Skipped (outside bbox): {skipped_bbox} flights", flush=True)

        # Save
        self.save_pkl("flights_by_cell_day", flights_by_cell_day)

    def get_mountainess(self, lat: float, lon: float) -> float:
        """
        Get mountainess value for given coordinates.

        Args:
            lat, lon: coordinates

        Returns:
            Mountainess value [0, 1] or 0.5 if not found
        """
        zoom = 7
        try:
            coords = TilesMaths.LatLonToTileCoords(zoom, lat, lon)
            filepath = os.path.join(
                self.elevation_dir,
                str(zoom),
                str(coords['tx']),
                f"{coords['ty']}.mountainess"
            )

            if not os.path.exists(filepath):
                return 0.5  # Default value

            with open(filepath, 'rb') as f:
                content = f.read(256 * 256)

            byte_idx = coords['x'] * 256 + coords['y']
            if byte_idx >= len(content):
                return 0.5

            value = struct.unpack('B', content[byte_idx:byte_idx+1])[0]
            return float(value) / 255.0

        except Exception:
            return 0.5  # Default on error

    def build_mountainess_by_cell_alt(self) -> None:
        """
        Step 5: Build mountainess_by_cell_alt.pkl.

        Creates:
        - mountainess_by_cell_alt.pkl: List[List[float]] [nb_cells][5]
        """
        print("\n=== Step 5: Building mountainess_by_cell_alt ===", flush=True)

        mountainess_by_cell_alt = []

        for lat, lon in tqdm.tqdm(self.cells_latlon, desc="  Processing cells"):
            # Get mountainess for cell center
            mountainess = self.get_mountainess(lat, lon)

            # Duplicate for 5 altitude levels (1000, 900, 800, 700, 600 hPa)
            mountainess_by_cell_alt.append([mountainess] * 5)

        self.save_pkl("mountainess_by_cell_alt", mountainess_by_cell_alt)

    def validate(self) -> bool:
        """
        Validate that all PKL files were created correctly.

        Returns:
            True if validation passed
        """
        print("\n=== Validation ===", flush=True)

        required_files = [
            "sorted_cells_latlon",
            "sorted_cells",
            "meteo_days",
            "meteo_params",
            "meteo_content_by_cell_day",
            "flights_by_cell_day",
            "mountainess_by_cell_alt"
        ]

        all_ok = True
        for name in required_files:
            if BinObj.exists(name, path=self.out_dir):
                print(f"  ✓ {name}.pkl", flush=True)
            else:
                print(f"  ✗ {name}.pkl MISSING", flush=True)
                all_ok = False

        if all_ok:
            # Load and check dimensions
            cells = BinObj.load("sorted_cells_latlon", path=self.out_dir)
            days = BinObj.load("meteo_days", path=self.out_dir)
            params = BinObj.load("meteo_params", path=self.out_dir)
            meteo = BinObj.load("meteo_content_by_cell_day", path=self.out_dir)
            flights = BinObj.load("flights_by_cell_day", path=self.out_dir)
            mountainess = BinObj.load("mountainess_by_cell_alt", path=self.out_dir)

            print(f"\n  Dimensions:", flush=True)
            print(f"    Cells: {len(cells)}", flush=True)
            print(f"    Days: {len(days)}", flush=True)
            print(f"    Parameters: {len(params)}", flush=True)
            print(f"    Meteo shape: {meteo.shape}", flush=True)
            print(f"    Flights: {len(flights)} (days*cells)", flush=True)
            print(f"    Mountainess: {len(mountainess)}x5", flush=True)

            # Count total flights
            total_flights = sum(len(day_cell) for day_cell in flights)
            print(f"    Total flights: {total_flights}", flush=True)

        return all_ok


def parse_bbox(bbox_str: str) -> Tuple[float, float, float, float]:
    """Parse bbox string 'lat_min,lat_max,lon_min,lon_max'."""
    parts = bbox_str.split(',')
    if len(parts) != 4:
        raise ValueError("bbox must be: lat_min,lat_max,lon_min,lon_max")
    return tuple(float(p) for p in parts)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build PKL dataset for neural network training"
    )
    parser.add_argument("--db-url",
                       default=os.environ.get("IGC_DB_URL", "postgresql://paraglidable:paraglidable@localhost:5432/paraglidable"),
                       help="PostgreSQL connection URL (default: from IGC_DB_URL env or localhost)")
    parser.add_argument("--bbox", required=True,
                       help="Bounding box: lat_min,lat_max,lon_min,lon_max (e.g., '43,49,4,37')")
    parser.add_argument("--gfs-dir", default="data/gfs/anl",
                       help="Path to GFS GRIB files directory")
    parser.add_argument("--elevation-dir", default="../tiler/_cache/elevation",
                       help="Path to elevation tiles directory")
    parser.add_argument("--out-dir", default="../neural_network/bin/data",
                       help="Output directory for PKL files")
    parser.add_argument("--source", default="skygr",
                       help="Flight source to use")
    parser.add_argument("--skip-meteo", action="store_true",
                       help="Skip meteorological data processing")
    parser.add_argument("--skip-flights", action="store_true",
                       help="Skip flights processing")

    args = parser.parse_args()

    # Parse bbox
    try:
        bbox = parse_bbox(args.bbox)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    # Convert paths to absolute
    out_dir = os.path.abspath(args.out_dir)
    elevation_dir = os.path.abspath(args.elevation_dir)
    gfs_dir = os.path.abspath(args.gfs_dir)

    print("=== PKL Dataset Builder ===", flush=True)
    print(f"Bbox: {bbox}", flush=True)
    print(f"Output directory: {out_dir}", flush=True)

    # Connect to database (if needed for flights)
    db = None
    if not args.skip_flights:
        print(f"\nConnecting to database...", flush=True)
        try:
            db = connect_db(args.db_url, None)
        except Exception as e:
            print(f"ERROR: Failed to connect to database: {e}", file=sys.stderr)
            return 1

    try:
        # Create builder
        builder = PKLDatasetBuilder(
            bbox=bbox,
            gfs_dir=gfs_dir,
            elevation_dir=elevation_dir,
            out_dir=out_dir,
            db=db
        )

        # Step 1: Build cells
        builder.build_cells()

        # Step 2: Scan meteo data
        if not args.skip_meteo:
            builder.build_meteo_days()

            # Step 3: Extract meteo content
            builder.build_meteo_content()
        else:
            print("\n=== Skipping meteorological data ===", flush=True)
            # Create empty placeholders
            builder.save_pkl("meteo_days", [])
            builder.save_pkl("meteo_params", [])
            builder.save_pkl("meteo_content_by_cell_day", np.zeros((0, 195), dtype=np.float32))

        # Step 4: Build flights
        if not args.skip_flights:
            builder.build_flights_by_cell_day(source=args.source)
        else:
            print("\n=== Skipping flights data ===", flush=True)
            builder.save_pkl("flights_by_cell_day", [])

        # Step 5: Build mountainess
        builder.build_mountainess_by_cell_alt()

        # Validate
        if builder.validate():
            print("\n✓ Dataset built successfully!", flush=True)
            return 0
        else:
            print("\n✗ Validation failed", flush=True)
            return 1

    except KeyboardInterrupt:
        print("\nInterrupted by user", flush=True)
        return 130
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1
    finally:
        if db:
            db.close()


if __name__ == "__main__":
    raise SystemExit(main())
