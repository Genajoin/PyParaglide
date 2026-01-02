"""
SPOTS dataset builder for PyParaglide.

Creates SPOTS-related PKL files from xContest JSON flight data.
Extracted from scripts/build_dataset/flight_processor.py
"""

import json
import math
import pickle
from collections import defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any


def load_flights_from_json(flights_dir: Path) -> Optional[List[Dict]]:
    """
    Load all flight data from JSON files in directory.

    Parses xContest API JSON format and extracts:
    - spot_id, spot_name from takeoff
    - lat, lon from takeoff.link
    - datetime from timeClaim
    - score from league.route.points

    Args:
        flights_dir: Directory containing xContest JSON files

    Returns:
        List of flight dictionaries, or None if error
    """
    flights_dir = Path(flights_dir)
    if not flights_dir.exists():
        print(f"  Flights directory not found: {flights_dir}")
        return None

    all_flights = []
    json_files = list(flights_dir.glob("*.json"))

    if not json_files:
        print(f"  No JSON files found in {flights_dir}")
        return None

    print(f"  Loading {len(json_files)} JSON files...")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, list):
                    continue

                for raw_flight in data:
                    # Extract takeoff info
                    takeoff = raw_flight.get('takeoff', {})
                    if not takeoff:
                        continue

                    # Extract coordinates from takeoff.link
                    # Format: https://www.xcontest.org/.../filter[point]=LON LAT&...
                    takeoff_link = takeoff.get('link', '')
                    lat = lon = None
                    if 'filter[point]=' in takeoff_link:
                        try:
                            coords_str = takeoff_link.split('filter[point]=')[1].split('&')[0]
                            lon_str, lat_str = coords_str.split()
                            lon = float(lon_str)
                            lat = float(lat_str)
                        except (ValueError, IndexError):
                            continue

                    if lat is None or lon is None:
                        continue

                    # Extract datetime from timeClaim
                    time_claim = raw_flight.get('timeClaim', '')
                    if not time_claim:
                        continue
                    # Convert ISO format to expected format
                    try:
                        dt = datetime.fromisoformat(time_claim.replace('Z', '+00:00'))
                        datetime_str = dt.strftime('%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        continue

                    # Extract score
                    league = raw_flight.get('league', {})
                    route = league.get('route', {})
                    score = route.get('points', 0)
                    if score is None:
                        score = 0

                    # Build flight record
                    flight = {
                        'spot_id': takeoff.get('id'),
                        'spot_name': takeoff.get('name'),
                        'lat': lat,
                        'lon': lon,
                        'datetime': datetime_str,
                        'score': score,
                        'takeoff_alt': 0,  # Not available in xContest API
                    }
                    all_flights.append(flight)

        except Exception as e:
            print(f"    Warning: Error loading {json_file}: {e}")

    print(f"  Loaded {len(all_flights)} flights")
    return all_flights


class FlightIndexer:
    """Computes cell_index and day_index from PKL data."""

    def __init__(self, cells_latlon: List[Tuple[float, float]], meteo_days: List[date]):
        """
        Args:
            cells_latlon: List of (lat, lon) tuples for training cells
            meteo_days: List of dates with meteo data
        """
        self.cells_latlon = cells_latlon
        self.meteo_days = meteo_days
        self.day_to_idx = {day: idx for idx, day in enumerate(meteo_days)}
        self.nb_cells = len(cells_latlon)
        self.nb_days = len(meteo_days)

    def get_cell_index(self, lat: float, lon: float) -> Optional[int]:
        """Get cell index from coordinates."""
        cell_lat = int(math.floor(lat))
        cell_lon = int(math.floor(lon))
        try:
            return self.cells_latlon.index((float(cell_lat), float(cell_lon)))
        except ValueError:
            return None

    def get_day_index(self, dt_str: str) -> Optional[int]:
        """Get day index from datetime string."""
        try:
            flight_date = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S').date()
            return self.day_to_idx.get(flight_date)
        except (ValueError, TypeError):
            return None

    def index_flights(self, flights: List[Dict]) -> List[Dict]:
        """
        Add cell_index and day_index to flight records.

        Args:
            flights: List of flight dictionaries

        Returns:
            List of flights with added indices
        """
        indexed = []
        skipped = 0

        for f in flights:
            lat = f.get('lat')
            lon = f.get('lon')
            dt_str = f.get('datetime')

            if lat is None or lon is None or dt_str is None:
                skipped += 1
                continue

            cell_idx = self.get_cell_index(float(lat), float(lon))
            day_idx = self.get_day_index(dt_str)

            f_copy = f.copy()
            f_copy['cell_index'] = cell_idx
            f_copy['day_index'] = day_idx
            indexed.append(f_copy)

        print(f"  Indexed {len(indexed)} flights (skipped {skipped})")
        return indexed


def filter_spots_by_flights(flights: List[Dict],
                             min_flights: int = 200) -> set:
    """
    Filter spots by minimum flight count.

    Args:
        flights: List of flight dictionaries with spot_id
        min_flights: Minimum number of flights per spot

    Returns:
        Set of valid spot IDs
    """
    spot_counts = defaultdict(int)
    for f in flights:
        spot_id = f.get('spot_id')
        if spot_id:
            spot_counts[spot_id] += 1

    valid_spots = {spot_id for spot_id, count in spot_counts.items()
                   if count >= min_flights}

    print(f"  Found {len(valid_spots)} spots with >= {min_flights} flights")
    return valid_spots


def create_spots_pkls(flights: List[Dict],
                      valid_spots: set,
                      out_dir: Path,
                      nb_cells: int,
                      nb_days: int) -> Tuple[List, Dict]:
    """
    Create all SPOTS-related PKL files.

    Creates:
    - spots.pkl: [('name', lat, lon), ...]
    - spots_by_cell.pkl: [[spot_id, ...], ...]
    - flights_by_spot.pkl: [(datetime, (score, None, takeoff_alt, lat, lon)), ...]

    Args:
        flights: List of indexed flight dictionaries
        valid_spots: Set of valid spot IDs
        out_dir: Output directory for PKL files
        nb_cells: Number of cells
        nb_days: Number of days

    Returns:
        (spots_list, xcontest_to_new_id) mapping
    """
    # Build spots list and mapping
    spots_dict = {}
    for f in flights:
        spot_id = f.get('spot_id')
        if spot_id in valid_spots and spot_id not in spots_dict:
            spots_dict[spot_id] = (
                f.get('spot_name') or f"spot_{spot_id}",
                f['lat'],
                f['lon']
            )

    spots_list = list(spots_dict.values())
    xcontest_to_new_id = {old_id: new_id for new_id, old_id in enumerate(spots_dict.keys())}

    # Save spots.pkl
    with open(out_dir / "spots.pkl", 'wb') as f:
        pickle.dump(spots_list, f)
    print(f"  Saved: spots.pkl ({len(spots_list)} spots)")

    # Save spots_by_cell.pkl
    spots_by_cell = [[] for _ in range(nb_cells)]
    spot_cells = set()
    for f in flights:
        spot_id = f.get('spot_id')
        cell_idx = f.get('cell_index')
        if spot_id in valid_spots and cell_idx is not None:
            new_spot_id = xcontest_to_new_id[spot_id]
            spot_cells.add((new_spot_id, cell_idx))

    for new_spot_id, cell_idx in spot_cells:
        if 0 <= cell_idx < nb_cells:
            spots_by_cell[cell_idx].append(new_spot_id)

    with open(out_dir / "spots_by_cell.pkl", 'wb') as f:
        pickle.dump(spots_by_cell, f)

    cells_with_spots = sum(1 for cell_spots in spots_by_cell if cell_spots)
    print(f"  Saved: spots_by_cell.pkl ({cells_with_spots}/{nb_cells} cells with spots)")

    # Save flights_by_spot.pkl
    nb_spots = len(xcontest_to_new_id)
    flights_by_spot = [[] for _ in range(nb_spots)]

    for f in flights:
        spot_id = f.get('spot_id')
        if spot_id in valid_spots:
            new_spot_id = xcontest_to_new_id[spot_id]
            record = (
                f['datetime'],
                (
                    float(f.get('score', 0.0) or 0.0),
                    None,  # alt - always None for SPOTS
                    float(f.get('takeoff_alt', 0.0) or 0.0),
                    float(f['lat']),
                    float(f['lon'])
                )
            )
            flights_by_spot[new_spot_id].append(record)

    with open(out_dir / "flights_by_spot.pkl", 'wb') as f:
        pickle.dump(flights_by_spot, f)

    total_flights = sum(len(flights) for flights in flights_by_spot)
    print(f"  Saved: flights_by_spot.pkl ({total_flights} flights)")

    return spots_list, xcontest_to_new_id


def build_spots_dataset(
    flights_dir: Path,
    pkl_dir: Path,
    min_flights_per_spot: int = 200,
) -> bool:
    """
    Build SPOTS PKL files from flight data.

    Args:
        flights_dir: Directory containing xContest JSON files
        pkl_dir: Directory containing PKL files (will also be output dir)
        min_flights_per_spot: Minimum flights required per spot

    Returns:
        True if successful, False otherwise
    """
    from pyparaglide.data.dataset import load_pkl

    print("Building SPOTS dataset...")

    # Load existing PKL metadata
    try:
        meteo_days = load_pkl("meteo_days.pkl", pkl_dir)
        sorted_cells_latlon = load_pkl("sorted_cells_latlon.pkl", pkl_dir)
    except Exception as e:
        print(f"  Error loading PKL metadata: {e}")
        return False

    # Load flights
    flights = load_flights_from_json(flights_dir)
    if not flights:
        return False

    # Index flights
    indexer = FlightIndexer(sorted_cells_latlon, meteo_days)
    indexed_flights = indexer.index_flights(flights)

    # Filter spots
    valid_spots = filter_spots_by_flights(
        indexed_flights,
        min_flights=min_flights_per_spot
    )

    if not valid_spots:
        print("  No valid spots found")
        return False

    # Create PKL files
    create_spots_pkls(
        indexed_flights,
        valid_spots,
        Path(pkl_dir),
        indexer.nb_cells,
        indexer.nb_days
    )

    print("SPOTS dataset build complete!")
    return True
