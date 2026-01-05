"""
Flight data processing for xContest API data.

Merges functionality from:
- scripts/extract_training_data.py (JSON loading, indexing, clustering)
- scripts/build_pkl_from_xcontest.py (spot filtering, PKL generation)
"""

import json
import math
import pickle
import re
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any


class FlightIndexer:
    """Computes cell_index and day_index from PKL data."""

    def __init__(self, cells_latlon: List[Tuple[float, float]],
                 meteo_days: List[date],
                 date_ranges: Optional[List[Tuple[date, date]]] = None):
        """
        Args:
            cells_latlon: List of (lat, lon) tuples for training cells
            meteo_days: List of dates with meteo data
            date_ranges: Optional date ranges for filtering
        """
        self.cells_latlon = cells_latlon

        # Apply date filtering if specified
        if date_ranges:
            self.meteo_days = _filter_meteo_days(meteo_days, date_ranges)
            print(f"  Filtered days: {len(meteo_days)} -> {len(self.meteo_days)}")
        else:
            self.meteo_days = meteo_days

        self.day_to_idx = {day: idx for idx, day in enumerate(self.meteo_days)}
        self.nb_cells = len(self.cells_latlon)
        self.nb_days = len(self.meteo_days)

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
        except (ValueError, AttributeError):
            return None


def load_flights_from_json(flights_dir: Path) -> Optional[List[Dict]]:
    """
    Load and merge all JSON files from data/flights/ directory.

    Returns:
        List of flight dictionaries or None if no files found
    """
    json_files = list(flights_dir.glob("*.json"))

    if not json_files:
        print(f"  No JSON files found in {flights_dir}")
        return None

    print(f"  Found {len(json_files)} JSON files")

    all_flights = []
    duplicates = 0

    for filepath in sorted(json_files):
        data = _load_json_file(filepath)
        if data is None:
            continue

        if not isinstance(data, list):
            print(f"  WARNING: Invalid format in {filepath.name}")
            continue

        print(f"    {filepath.name}: {len(data)} records")
        all_flights.extend(data)

    # Remove duplicates by id
    unique_flights = {}
    for flight in all_flights:
        flight_id = flight.get('id')
        if flight_id and flight_id not in unique_flights:
            unique_flights[flight_id] = flight
        elif flight_id in unique_flights:
            duplicates += 1

    merged = list(unique_flights.values())
    print(f"  Total: {len(merged)} unique flights ({duplicates} duplicates removed)")

    return merged


def _load_json_file(filepath: Path) -> Optional[List]:
    """Load a JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"  ERROR reading {filepath.name}: {e}")
        return None


def extract_coords_from_link(link: str) -> Tuple[Optional[float], Optional[float]]:
    """Extract coordinates from xContest takeoff.link URL."""
    try:
        match = re.search(r'filter\[point\]=([0-9.-]+)\s+([0-9.-]+)', link)
        if match:
            return float(match.group(2)), float(match.group(1))  # lat, lon
    except (ValueError, AttributeError):
        pass
    return None, None


def parse_datetime(dt_str: str) -> Optional[str]:
    """Parse datetime string to format 'yyyy-mm-dd hh:mm:ss'."""
    if not dt_str:
        return None

    formats = [
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S.%fZ',
        '%Y-%m-%d %H:%M:%S',
        '%Y-%m-%dT%H:%M:%S%z',
        '%Y-%m-%dT%H:%M:%S',
    ]

    for fmt in formats:
        try:
            dt = datetime.strptime(dt_str, fmt)
            return dt.strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            continue
    return None


def parse_duration(duration_str: str) -> Optional[int]:
    """Parse ISO 8601 duration string (PT04H01M35S) to seconds."""
    try:
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str)
        if match:
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            seconds = int(match.group(3) or 0)
            return hours * 3600 + minutes * 60 + seconds
    except (ValueError, AttributeError):
        pass
    return None


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two points in kilometers using Haversine formula."""
    R = 6371.0
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def cluster_nearby_spots(spots: Dict[str, Dict],
                         distance_km: float = 2.0) -> Tuple[Dict, Dict]:
    """
    Cluster nearby spots and return merged dictionary + mapping.

    Args:
        spots: Dictionary {spot_id: {id, name, lat, lon, flight_count, ...}}
        distance_km: Maximum distance for clustering

    Returns:
        (clustered_spots, spot_id_mapping) - merged spots and old_id -> new_id mapping
    """
    if not spots or len(spots) <= 1:
        return spots, {}

    spot_list = list(spots.values())
    clusters = []
    assigned = set()

    # Hierarchical clustering
    for i, spot in enumerate(spot_list):
        if spot['id'] in assigned:
            continue

        cluster = [spot]
        assigned.add(spot['id'])

        for j, other in enumerate(spot_list):
            if i == j or other['id'] in assigned:
                continue

            dist = haversine_distance(
                spot['lat'], spot['lon'],
                other['lat'], other['lon']
            )

            if dist <= distance_km:
                cluster.append(other)
                assigned.add(other['id'])

        clusters.append(cluster)

    # Build clustered spots
    clustered_spots = {}
    spot_id_mapping = {}

    new_id = 0
    for cluster in clusters:
        if not cluster:
            continue

        # Sort by flight_count descending
        cluster_sorted = sorted(cluster, key=lambda s: s.get('flight_count', 0), reverse=True)
        representative = cluster_sorted[0]

        # Weighted centroid by flight_count
        total_flights = sum(s.get('flight_count', 0) for s in cluster_sorted)
        if total_flights > 0:
            weighted_lat = sum(s['lat'] * s.get('flight_count', 0) for s in cluster_sorted) / total_flights
            weighted_lon = sum(s['lon'] * s.get('flight_count', 0) for s in cluster_sorted) / total_flights
        else:
            weighted_lat = representative['lat']
            weighted_lon = representative['lon']

        new_spot = {
            'id': str(new_id),
            'name': representative['name'],
            'lat': weighted_lat,
            'lon': weighted_lon,
            'countryIso': representative.get('countryIso', ''),
            'cell_lat': int(math.floor(weighted_lat)),
            'cell_lon': int(math.floor(weighted_lon)),
            'flight_count': total_flights,
            'cluster_size': len(cluster_sorted),
            'original_spots': [s['id'] for s in cluster_sorted]
        }

        clustered_spots[str(new_id)] = new_spot

        for old_spot in cluster_sorted:
            spot_id_mapping[old_spot['id']] = str(new_id)

        new_id += 1

    print(f"  Clustering: {len(spots)} spots -> {len(clustered_spots)} clusters (radius {distance_km}km)")

    return clustered_spots, spot_id_mapping


def is_within_bbox(lat: float, lon: float, bbox: Tuple[float, float, float, float]) -> bool:
    """Check if coordinates are within bounding box."""
    lat_min, lat_max, lon_min, lon_max = bbox
    return lat_min <= lat < lat_max and lon_min <= lon < lon_max


def _is_date_in_ranges(target_date: date, ranges: List[Tuple[date, date]]) -> bool:
    """Check if date falls within any of the date ranges."""
    if not ranges:
        return True
    for start, end in ranges:
        if start <= target_date <= end:
            return True
    return False


def _filter_meteo_days(meteo_days: List[date], ranges: List[Tuple[date, date]]) -> List[date]:
    """Filter meteo_days list, keeping only dates in specified ranges."""
    if not ranges:
        return meteo_days
    return [d for d in meteo_days if _is_date_in_ranges(d, ranges)]


def process_flights(flights: List[Dict],
                    indexer: 'FlightIndexer',
                    elev_reader,
                    bbox: Optional[Tuple[float, float, float, float]] = None,
                    date_ranges: Optional[List[Tuple[date, date]]] = None,
                    cluster_distance_km: Optional[float] = None) -> Dict:
    """
    Process xContest flights and generate training data structures.

    Args:
        flights: List of flight dictionaries from xContest API
        indexer: FlightIndexer for computing indices
        elev_reader: ElevationReader for elevation/mountainess data
        bbox: Optional bounding box filter
        date_ranges: Optional date range filters
        cluster_distance_km: Optional spot clustering radius

    Returns:
        Dictionary with processed data and statistics
    """
    from pyparaglide.preprocessing.flights.elevation_reader import ElevationReader

    training_flights = []
    spots = {}

    stats = {
        'total': len(flights),
        'with_coords': 0,
        'with_valid_time': 0,
        'by_country': defaultdict(int),
        'by_cell': defaultdict(int),
        'spots_created': 0,
        'errors': defaultdict(int),
    }

    for flight in flights:
        takeoff = flight.get('takeoff', {})
        point_start = flight.get('pointStart', {})
        flight_stats = flight.get('stats', {})
        league = flight.get('league', {})

        # Extract coordinates
        link = takeoff.get('link', '')
        lat, lon = extract_coords_from_link(link)

        if lat is None or lon is None:
            stats['errors']['no_coords'] += 1
            continue

        stats['with_coords'] += 1

        # Parse datetime
        time_str = point_start.get('time', '')
        dt_formatted = parse_datetime(time_str)
        if not dt_formatted:
            stats['errors']['invalid_time'] += 1
            continue

        stats['with_valid_time'] += 1

        # Filter by bbox
        if bbox and not is_within_bbox(lat, lon, bbox):
            stats['errors']['outside_bbox'] += 1
            continue

        # Filter by date ranges
        if date_ranges:
            flight_date = datetime.strptime(dt_formatted, '%Y-%m-%d %H:%M:%S').date()
            if not _is_date_in_ranges(flight_date, date_ranges):
                stats['errors']['outside_date_range'] += 1
                continue

        # Compute cell
        cell_lat, cell_lon = int(math.floor(lat)), int(math.floor(lon))
        cell_key = f"{cell_lat}_{cell_lon}"
        stats['by_cell'][cell_key] += 1

        # Country
        country = takeoff.get('countryIso', '')
        if country:
            stats['by_country'][country] += 1

        # Duration
        duration_str = flight_stats.get('duration', '')
        duration_sec = parse_duration(duration_str)

        # Spot data
        spot_id = takeoff.get('id')
        if spot_id and spot_id not in spots:
            spots[spot_id] = {
                'id': spot_id,
                'name': takeoff.get('name', ''),
                'lat': lat,
                'lon': lon,
                'countryIso': country,
                'cell_lat': cell_lat,
                'cell_lon': cell_lon,
                'flight_count': 0
            }
            stats['spots_created'] += 1

        if spot_id in spots:
            spots[spot_id]['flight_count'] += 1

        # Score data
        route = league.get('route', {})
        score = route.get('points') if route else None
        avg_speed = route.get('avgSpeed') if route else None

        # Elevation data
        takeoff_alt = None
        mountainess = 0.5
        if isinstance(elev_reader, ElevationReader):
            takeoff_alt = elev_reader.get_elevation(lat, lon)
            mountainess = elev_reader.get_mountainess(lat, lon)

        # Indices
        cell_index = indexer.get_cell_index(lat, lon)
        day_index = indexer.get_day_index(dt_formatted)

        if cell_index is None:
            stats['errors']['outside_training_cells'] += 1
            continue
        if day_index is None:
            stats['errors']['outside_training_dates'] += 1
            continue

        # Build training entry
        training_entry = {
            'datetime': dt_formatted,
            'score': score,
            'alt': None,
            'plaf': avg_speed,
            'lat': lat,
            'lon': lon,
            'takeoff_alt': takeoff_alt,
            'mountainess': mountainess,
            'cell_lat': cell_lat,
            'cell_lon': cell_lon,
            'cell_index': cell_index,
            'day_index': day_index,
            'spot_id': spot_id,
            'spot_name': takeoff.get('name', ''),
        }

        training_flights.append(training_entry)

    # Cluster spots if requested
    if cluster_distance_km and spots:
        spots, spot_id_mapping = cluster_nearby_spots(spots, cluster_distance_km)
        # Update spot_id in flights
        for flight_entry in training_flights:
            old_spot_id = flight_entry.get('spot_id')
            if old_spot_id in spot_id_mapping:
                flight_entry['spot_id'] = spot_id_mapping[old_spot_id]

    return {
        'flights': training_flights,
        'spots': spots,
        'stats': dict(stats),
    }


def filter_spots_by_flights(flights: List[Dict],
                            min_flights: int = 200,
                            bbox: Optional[Tuple[float, float, float, float]] = None,
                            date_ranges: Optional[List[Tuple[date, date]]] = None) -> set:
    """
    Filter spots with minimum flight count.

    Args:
        flights: List of processed flight entries
        min_flights: Minimum flights per spot
        bbox: Optional bbox filter
        date_ranges: Optional date range filter

    Returns:
        Set of valid spot IDs
    """
    filtered_flights = flights

    if bbox:
        filtered_flights = [f for f in filtered_flights
                            if is_within_bbox(f['lat'], f['lon'], bbox)]

    if date_ranges:
        filtered_flights = [f for f in filtered_flights
                            if _is_date_in_ranges(
                                datetime.strptime(f['datetime'], '%Y-%m-%d %H:%M:%S').date(),
                                date_ranges
                            )]

    # Count flights per spot
    flight_count = defaultdict(int)
    for f in filtered_flights:
        spot_id = f.get('spot_id')
        if spot_id is not None:
            flight_count[spot_id] += 1

    valid_spots = {spot_id for spot_id, count in flight_count.items()
                   if count >= min_flights}

    print(f"  Spots with >= {min_flights} flights: {len(valid_spots)}")

    return valid_spots


def create_spots_pkls(flights: List[Dict],
                      valid_spots: set,
                      out_dir: Path,
                      nb_cells: int,
                      nb_days: int) -> Tuple[List, Dict]:
    """
    Create all takeoff spot-related PKL files.

    Creates:
    - spots.pkl: [('name', lat, lon), ...]
    - spots_by_cell.pkl: [[spot_id, ...], ...]
    - flights_by_spot.pkl: [(datetime, (score, None, takeoff_alt, lat, lon)), ...]
    - flights_by_cell_day_spot.pkl: [[[{spot_id: [...]}, ...], ...], ...]
    - flights_by_cell_day.pkl: [[(datetime, (...)), ...], ...]

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
                    None,  # alt - always None for takeoff spot records
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

    # Save flights_by_cell_day_spot.pkl
    structure = [[{} for _ in range(nb_days)] for _ in range(nb_cells)]

    for f in flights:
        spot_id = f.get('spot_id')
        cell_idx = f.get('cell_index')
        day_idx = f.get('day_index')

        if spot_id in valid_spots and cell_idx is not None and day_idx is not None:
            if 0 <= cell_idx < nb_cells and 0 <= day_idx < nb_days:
                new_spot_id = xcontest_to_new_id[spot_id]
                record = (
                    f['datetime'],
                    (
                        float(f.get('score', 0.0) or 0.0),
                        None,
                        float(f.get('takeoff_alt', 0.0) or 0.0),
                        float(f['lat']),
                        float(f['lon'])
                    )
                )

                if new_spot_id not in structure[cell_idx][day_idx]:
                    structure[cell_idx][day_idx][new_spot_id] = []
                structure[cell_idx][day_idx][new_spot_id].append(record)

    with open(out_dir / "flights_by_cell_day_spot.pkl", 'wb') as f:
        pickle.dump(structure, f)

    total_entries = sum(len(structure[c][d]) for c in range(nb_cells) for d in range(nb_days))
    print(f"  Saved: flights_by_cell_day_spot.pkl ({total_entries} cell-day-spot entries)")

    # Save flights_by_cell_day.pkl
    flights_by_cell_day = [[] for _ in range(nb_days * nb_cells)]

    for f in flights:
        cell_idx = f.get('cell_index')
        day_idx = f.get('day_index')

        if cell_idx is not None and day_idx is not None:
            if 0 <= cell_idx < nb_cells and 0 <= day_idx < nb_days:
                record = (
                    f['datetime'],
                    (
                        float(f.get('score', 0.0) or 0.0),  # [1][0] - score for crossability
                        float(f['lat']),                    # [1][1] - latitude
                        float(f['lon']),                    # [1][2] - longitude
                    )
                )

                idx = day_idx * nb_cells + cell_idx
                flights_by_cell_day[idx].append(record)

    with open(out_dir / "flights_by_cell_day.pkl", 'wb') as f:
        pickle.dump(flights_by_cell_day, f)

    total = sum(len(day_cell) for day_cell in flights_by_cell_day)
    print(f"  Saved: flights_by_cell_day.pkl ({total} flights in {nb_days*nb_cells} slots)")

    return spots_list, xcontest_to_new_id
