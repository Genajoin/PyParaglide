#!/usr/bin/env python3
"""
Анализ распределения полётов по ячейкам bbox.

Показывает:
- Общее количество полётов
- Распределение по ячейкам 1°×1°
- Автоматическое определение кластеров ячеек
- Распределение по месяцам
- Топ спотов

Usage:
    python scripts/analyze_flights.py                        # Анализ всех полётов
    python scripts/analyze_flights.py --bbox 45,47,13,16  # Фильтр по bbox
    python scripts/analyze_flights.py --min-flights 50     # Фильтр по min flights
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pyparaglide.config import settings


def extract_coords(link: str):
    """Extract coordinates from xContest takeoff.link URL."""
    match = re.search(r'filter\[point\]=([0-9.-]+)\s+([0-9.-]+)', link)
    if match:
        return float(match.group(2)), float(match.group(1))  # lat, lon
    return None, None


def parse_datetime(dt_str: str):
    """Parse datetime string."""
    formats = [
        '%Y-%m-%dT%H:%M:%SZ',
        '%Y-%m-%dT%H:%M:%S.%fZ',
        '%Y-%m-%d %H:%M:%S',
    ]
    for fmt in formats:
        try:
            return datetime.strptime(dt_str, fmt)
        except ValueError:
            continue
    return None


def detect_cell_clusters(by_cell):
    """Auto-detect clusters of contiguous cells using BFS."""
    cell_list = []
    for cell_key, count in by_cell.items():
        lat, lon = map(int, cell_key.split(','))
        cell_list.append((lat, lon, count))

    # Sort by flight count descending
    cell_list.sort(key=lambda x: -x[2])

    clusters = []
    assigned = set()

    for lat, lon, count in cell_list:
        cell_key = (lat, lon)
        if cell_key in assigned:
            continue

        # Start new cluster with this cell
        cluster = {'cells': [(lat, lon)], 'flights': count}
        assigned.add(cell_key)

        # Find nearby cells (BFS)
        queue = [(lat, lon)]
        while queue:
            curr_lat, curr_lon = queue.pop(0)

            # Check neighbors (within 2 degrees)
            for other_lat, other_lon, other_count in cell_list:
                other_key = (other_lat, other_lon)
                if other_key in assigned:
                    continue

                # Check if within 2 degrees
                if abs(curr_lat - other_lat) <= 2 and abs(curr_lon - other_lon) <= 2:
                    cluster['cells'].append((other_lat, other_lon))
                    cluster['flights'] += other_count
                    assigned.add(other_key)
                    queue.append((other_lat, other_lon))

        # Calculate bbox for this cluster
        lats = [c[0] for c in cluster['cells']]
        lons = [c[1] for c in cluster['cells']]
        cluster['lat_min'] = min(lats)
        cluster['lat_max'] = max(lats) + 1
        cluster['lon_min'] = min(lons)
        cluster['lon_max'] = max(lons) + 1
        cluster['count'] = len(cluster['cells'])

        clusters.append(cluster)

    # Sort by flights descending
    clusters.sort(key=lambda x: -x['flights'])
    return clusters


def find_top_spot_in_cluster(cells, by_spot):
    """Find top spot in a cluster."""
    top_spot = ""
    top_count = 0
    for lat, lon in cells:
        for spot_id, data in by_spot.items():
            if abs(data['lat'] - lat) < 0.5 and abs(data['lon'] - lon) < 0.5:
                if data['count'] > top_count:
                    top_count = data['count']
                    top_spot = data['name']
    return top_spot or "-"


def combine_clusters(clusters):
    """Combine multiple clusters into one bbox."""
    all_lats = []
    all_lons = []
    total_flights = 0
    total_cells = 0

    for c in clusters:
        for lat, lon in c['cells']:
            all_lats.append(lat)
            all_lons.append(lon)
        total_flights += c['flights']
        total_cells += c['count']

    return {
        'lat_min': min(all_lats),
        'lat_max': max(all_lats) + 1,
        'lon_min': min(all_lons),
        'lon_max': max(all_lons) + 1,
        'flights': total_flights,
        'count': total_cells,
    }


def analyze_flights(flights_dir: Path, bbox: tuple = None, min_flights: int = 0):
    """Analyze flight distribution by cells."""
    # Load all JSON files
    json_files = list(flights_dir.glob("*.json"))
    if not json_files:
        print(f"❌ No JSON files found in {flights_dir}")
        return

    print(f"📂 Loading flights from {len(json_files)} files...")

    all_flights = []
    for filepath in json_files:
        with open(filepath) as f:
            all_flights.extend(json.load(f))

    print(f"✓ Loaded {len(all_flights):,} unique flights\n")

    # Parse data
    by_cell = defaultdict(int)
    by_month = defaultdict(int)
    by_spot = defaultdict(lambda: {'count': 0, 'lat': 0, 'lon': 0, 'name': ''})
    by_country = defaultdict(int)

    inside_bbox = 0
    outside_bbox = 0
    no_coords = 0

    for flight in all_flights:
        # Extract coordinates
        link = flight.get('takeoff', {}).get('link', '')
        lat, lon = extract_coords(link)

        if lat is None or lon is None:
            no_coords += 1
            continue

        # Cell
        cell_lat = int(lat)
        cell_lon = int(lon)
        cell_key = f"{cell_lat},{cell_lon}"

        # Check bbox (only if specified)
        if bbox:
            lat_min, lat_max, lon_min, lon_max = bbox
            if lat_min <= lat < lat_max and lon_min <= lon < lon_max:
                inside_bbox += 1
            else:
                outside_bbox += 1
                continue

        if not bbox:
            inside_bbox += 1

        by_cell[cell_key] += 1

        # Month
        time_str = flight.get('pointStart', {}).get('time', '')
        if time_str:
            dt = parse_datetime(time_str)
            if dt:
                by_month[dt.month] += 1

        # Country
        country = flight.get('takeoff', {}).get('countryIso', '')
        if country:
            by_country[country] += 1

        # Spot
        spot_id = flight.get('takeoff', {}).get('id', '')
        spot_name = flight.get('takeoff', {}).get('name', '')
        if spot_id:
            by_spot[spot_id]['count'] += 1
            by_spot[spot_id]['lat'] = lat
            by_spot[spot_id]['lon'] = lon
            by_spot[spot_id]['name'] = spot_name

    # Print results
    print("=" * 60)
    print(f"📊 FLIGHT ANALYSIS")
    print("=" * 60)

    if bbox:
        print(f"\n📍 Filtered by BBox: {bbox}")
        print(f"📈 Flights in bbox: {inside_bbox:,} ({inside_bbox/(inside_bbox+outside_bbox)*100:.1f}%)")
        print(f"📈 Flights outside bbox: {outside_bbox:,}")
    else:
        print(f"\n📈 Total flights: {inside_bbox:,}")

    if no_coords > 0:
        print(f"⚠️  No coordinates: {no_coords:,}")

    # By cell
    if by_cell:
        print(f"\n🗺️  BY CELL (1°×1°) - Top 20")
        print("-" * 40)
        sorted_cells = sorted(by_cell.items(), key=lambda x: -x[1])
        for cell, count in sorted_cells[:20]:
            lat, lon = map(int, cell.split(','))
            print(f"  Cell ({lat:3d}, {lon:3d}): {count:5d} flights")
        if len(by_cell) > 20:
            print(f"  ... and {len(by_cell)-20} more cells")

    # By month
    if by_month:
        print(f"\n📅 BY MONTH")
        print("-" * 40)
        month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
                      7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}
        for month in sorted(by_month.keys()):
            print(f"  {month_names[month]}: {by_month[month]:,} flights")

    # By country
    if by_country:
        print(f"\n🌍 BY COUNTRY (top 10)")
        print("-" * 40)
        sorted_countries = sorted(by_country.items(), key=lambda x: -x[1])
        for country, count in sorted_countries[:10]:
            print(f"  {country}: {count:,} flights")

    # By spot
    if by_spot:
        print(f"\n🪁 BY SPOT (top 20)")
        print("-" * 40)

        # Filter by min_flights
        filtered_spots = {k: v for k, v in by_spot.items() if v['count'] >= min_flights}

        sorted_spots = sorted(filtered_spots.items(), key=lambda x: -x[1]['count'])
        for spot_id, data in sorted_spots[:20]:
            print(f"  {data['name'][:30]:30} ({data['count']:4d} flights) [{data['lat']:.2f}, {data['lon']:.2f}]")

        print(f"\n  Total spots with >= {min_flights} flights: {len(filtered_spots)}")

    # Recommendations
    print(f"\n💡 RECOMMENDATIONS")
    print("-" * 40)

    if by_cell:
        # Auto-detect cell clusters
        clusters = detect_cell_clusters(by_cell)

        print(f"\n  📍 DETECTED CELL CLUSTERS (auto-discovered):")
        print(f"     {'BBox':<25} {'Cells':>6} {'Flights':>10} {'Top Spot':<20}")
        print(f"     {'-'*25} {'-'*6} {'-'*10} {'-'*20}")

        for i, cluster in enumerate(clusters[:5], 1):  # Top 5 clusters
            bbox_str = f"{cluster['lat_min']},{cluster['lat_max']},{cluster['lon_min']},{cluster['lon_max']}"
            top_spot = find_top_spot_in_cluster(cluster['cells'], by_spot)
            print(f"     {i}) {bbox_str:<23} {cluster['count']:>6} {cluster['flights']:>10,} {top_spot[:20]}")

        # Show overall bbox
        lats = [int(c.split(',')[0]) for c in by_cell.keys()]
        lons = [int(c.split(',')[1]) for c in by_cell.keys()]
        all_count = sum(by_cell.values())

        print(f"\n  🎯 SUGGESTED TRAINING BBOXES:")
        if clusters:
            best = clusters[0]
            print(f"     Best cluster (most flights):")
            print(f"       BBOX={best['lat_min']},{best['lat_max']},{best['lon_min']},{best['lon_max']}")
            print(f"       → {best['flights']:,} flights in {best['count']} cells")

            if len(clusters) > 1:
                print(f"\n     Top 2 clusters combined:")
                combined = combine_clusters(clusters[:2])
                print(f"       BBOX={combined['lat_min']},{combined['lat_max']},{combined['lon_min']},{combined['lon_max']}")
                print(f"       → {combined['flights']:,} flights in {combined['count']} cells")

        print(f"\n     All regions:")
        print(f"       BBOX={min(lats)},{max(lats)+1},{min(lons)},{max(lons)+1}")
        print(f"       → {all_count:,} flights in {len(by_cell)} cells")

    if min_flights > 0:
        min_flights_suggestions = [10, 25, 50, 100, 150, 200]
        print(f"\n  🪁 SPOTS BY MIN_FLIGHTS THRESHOLD:")
        for threshold in min_flights_suggestions:
            count = sum(1 for s in by_spot.values() if s['count'] >= threshold)
            print(f"     >= {threshold:3d} flights: {count:3d} spots")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze flight distribution by cells to determine optimal training bbox",
        epilog="Examples:\n"
                "  python scripts/analyze_flights.py                # Analyze all flights\n"
                "  python scripts/analyze_flights.py --bbox 45,47,13,16  # Filter by bbox\n"
                "  python scripts/analyze_flights.py --min-flights 50     # Filter spots",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--flights-dir', type=str, default=None,
                        help='Flights directory (default: from .env)')
    parser.add_argument('--bbox', type=str, default=None,
                        help='Optional bounding box filter: lat_min,lat_max,lon_min,lon_max')
    parser.add_argument('--min-flights', type=int, default=0,
                        help='Minimum flights per spot to include in spot list')

    args = parser.parse_args()

    # Parse bbox (optional)
    bbox = None
    if args.bbox:
        parts = [float(x.strip()) for x in args.bbox.split(',')]
        if len(parts) == 4:
            bbox = tuple(parts)
        else:
            print(f"❌ Invalid bbox format: {args.bbox}")
            print("   Expected: lat_min,lat_max,lon_min,lon_max")
            return

    # Flights dir
    flights_dir = Path(args.flights_dir) if args.flights_dir else Path(settings.flights_dir)

    analyze_flights(flights_dir, bbox, args.min_flights)


if __name__ == '__main__':
    main()
