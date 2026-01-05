"""
Flight data analyzer.

Analyzes xContest flight JSON files to determine optimal training bbox
by detecting clusters of flights in 1°×1° cells.
"""

import json
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Any


@dataclass
class SpotData:
    """Data about a take-off spot."""
    count: int
    lat: float
    lon: float
    name: str


@dataclass
class CellCluster:
    """A cluster of adjacent 1°×1° cells with flights."""
    cells: list[tuple[int, int]]  # [(lat, lon), ...]
    flights: int
    lat_min: int
    lat_max: int
    lon_min: int
    lon_max: int
    count: int  # number of cells
    top_spot: str = ""


@dataclass
class BBoxCoverage:
    """Coverage of flights within a bbox."""
    inside: int
    outside: int
    no_coords: int


@dataclass
class FlightAnalysisResult:
    """Result of flight analysis."""
    total_flights: int
    by_cell: dict[str, int]
    by_month: dict[int, int]
    by_country: dict[str, int]
    by_spot: dict[str, SpotData]
    clusters: list[CellCluster]
    bbox_coverage: BBoxCoverage | None = None


class FlightAnalyzer:
    """Analyze flight distribution by cells and spots."""

    def __init__(
        self,
        flights_dir: Path | str,
        bbox: tuple[float, float, float, float] | None = None,
    ):
        """
        Args:
            flights_dir: Directory with xContest JSON files
            bbox: Optional filter (lat_min, lat_max, lon_min, lon_max)
        """
        self.flights_dir = Path(flights_dir)
        self.bbox = bbox

    def load_flights(self) -> list[dict[str, Any]]:
        """Load all JSON files from flights directory."""
        json_files = list(self.flights_dir.glob("*.json"))
        if not json_files:
            raise ValueError(f"No JSON files found in {self.flights_dir}")

        all_flights = []
        for filepath in json_files:
            with open(filepath) as f:
                all_flights.extend(json.load(f))

        return all_flights

    def extract_coords(self, link: str) -> tuple[float | None, float | None]:
        """Extract coordinates from xContest takeoff.link URL."""
        match = re.search(r"filter\[point\]=([0-9.-]+)\s+([0-9.-]+)", link)
        if match:
            # xContest format: filter[point]=lon lat
            # group(1) is lon, group(2) is lat
            # We swap to return (lat, lon) for consistency
            return float(match.group(2)), float(match.group(1))
        return None, None

    def parse_datetime(self, dt_str: str) -> datetime | None:
        """Parse datetime string."""
        formats = [
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%d %H:%M:%S",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(dt_str, fmt)
            except ValueError:
                continue
        return None

    def detect_cell_clusters(
        self,
        by_cell: dict[str, int],
        max_distance_degrees: float = 2.0,
    ) -> list[CellCluster]:
        """
        Auto-detect clusters of contiguous cells using BFS.

        Args:
            by_cell: Dictionary {cell_key: flight_count}
            max_distance_degrees: Max distance for cluster adjacency (default 2°)

        Returns:
            List of CellCluster sorted by flight_count DESC
        """
        cell_list = []
        for cell_key, count in by_cell.items():
            lat, lon = map(int, cell_key.split(","))
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
            cluster = CellCluster(
                cells=[(lat, lon)],
                flights=count,
                lat_min=lat,
                lat_max=lat,
                lon_min=lon,
                lon_max=lon,
                count=1,
            )
            assigned.add(cell_key)

            # Find nearby cells (BFS)
            queue = deque([(lat, lon)])
            while queue:
                curr_lat, curr_lon = queue.popleft()

                # Check neighbors (within max_distance_degrees)
                for other_lat, other_lon, other_count in cell_list:
                    other_key = (other_lat, other_lon)
                    if other_key in assigned:
                        continue

                    # Check if within max_distance_degrees
                    if (
                        abs(curr_lat - other_lat) <= max_distance_degrees
                        and abs(curr_lon - other_lon) <= max_distance_degrees
                    ):
                        cluster.cells.append((other_lat, other_lon))
                        cluster.flights += other_count
                        cluster.lat_min = min(cluster.lat_min, other_lat)
                        cluster.lat_max = max(cluster.lat_max, other_lat)
                        cluster.lon_min = min(cluster.lon_min, other_lon)
                        cluster.lon_max = max(cluster.lon_max, other_lon)
                        cluster.count += 1
                        assigned.add(other_key)
                        queue.append((other_lat, other_lon))

            # Add 1 to max for bbox format (max is exclusive)
            cluster.lat_max += 1
            cluster.lon_max += 1

            clusters.append(cluster)

        # Sort by flights descending
        clusters.sort(key=lambda x: -x.flights)
        return clusters

    def find_top_spot_in_cluster(
        self,
        cluster: CellCluster,
        by_spot: dict[str, SpotData],
    ) -> str:
        """Find top spot in a cluster."""
        top_spot = ""
        top_count = 0
        for lat, lon in cluster.cells:
            for spot_id, data in by_spot.items():
                if abs(data.lat - lat) < 0.5 and abs(data.lon - lon) < 0.5:
                    if data.count > top_count:
                        top_count = data.count
                        top_spot = data.name
        return top_spot or "-"

    def combine_clusters(self, clusters: list[CellCluster]) -> dict[str, Any]:
        """Combine multiple clusters into one bbox."""
        all_lats = []
        all_lons = []
        total_flights = 0
        total_cells = 0

        for c in clusters:
            for lat, lon in c.cells:
                all_lats.append(lat)
                all_lons.append(lon)
            total_flights += c.flights
            total_cells += c.count

        return {
            "lat_min": min(all_lats),
            "lat_max": max(all_lats) + 1,
            "lon_min": min(all_lons),
            "lon_max": max(all_lons) + 1,
            "flights": total_flights,
            "count": total_cells,
        }

    def analyze(
        self,
        min_flights_threshold: int = 0,
    ) -> FlightAnalysisResult:
        """
        Analyze flight distribution.

        Args:
            min_flights_threshold: Minimum flights per spot to include

        Returns:
            FlightAnalysisResult with distribution data and clusters
        """
        flights = self.load_flights()

        # Parse data
        by_cell = defaultdict(int)
        by_month = defaultdict(int)
        by_spot: dict[str, SpotData] = defaultdict(
            lambda: {"count": 0, "lat": 0.0, "lon": 0.0, "name": ""}
        )
        by_country = defaultdict(int)

        inside_bbox = 0
        outside_bbox = 0
        no_coords = 0

        for flight in flights:
            # Extract coordinates
            link = flight.get("takeoff", {}).get("link", "")
            lat, lon = self.extract_coords(link)

            if lat is None or lon is None:
                no_coords += 1
                continue

            # Cell
            cell_lat = int(lat)
            cell_lon = int(lon)
            cell_key = f"{cell_lat},{cell_lon}"

            # Check bbox (only if specified)
            if self.bbox:
                lat_min, lat_max, lon_min, lon_max = self.bbox
                if lat_min <= lat < lat_max and lon_min <= lon < lon_max:
                    inside_bbox += 1
                else:
                    outside_bbox += 1
                    continue

            if not self.bbox:
                inside_bbox += 1

            by_cell[cell_key] += 1

            # Month
            time_str = flight.get("pointStart", {}).get("time", "")
            if time_str:
                dt = self.parse_datetime(time_str)
                if dt:
                    by_month[dt.month] += 1

            # Country
            country = flight.get("takeoff", {}).get("countryIso", "")
            if country:
                by_country[country] += 1

            # Spot
            spot_id = flight.get("takeoff", {}).get("id", "")
            spot_name = flight.get("takeoff", {}).get("name", "")
            if spot_id:
                by_spot[spot_id]["count"] += 1
                by_spot[spot_id]["lat"] = lat
                by_spot[spot_id]["lon"] = lon
                by_spot[spot_id]["name"] = spot_name

        # Convert by_spot to SpotData objects
        by_spot_data: dict[str, SpotData] = {}
        for spot_id, data in by_spot.items():
            if data["count"] >= min_flights_threshold:
                by_spot_data[spot_id] = SpotData(
                    count=data["count"],
                    lat=data["lat"],
                    lon=data["lon"],
                    name=data["name"],
                )

        # Detect clusters
        clusters = self.detect_cell_clusters(by_cell)

        # Find top spots for clusters
        for cluster in clusters:
            cluster.top_spot = self.find_top_spot_in_cluster(cluster, by_spot_data)

        # BBox coverage
        bbox_coverage = None
        if self.bbox:
            bbox_coverage = BBoxCoverage(
                inside=inside_bbox,
                outside=outside_bbox,
                no_coords=no_coords,
            )

        return FlightAnalysisResult(
            total_flights=inside_bbox,
            by_cell=dict(by_cell),
            by_month=dict(by_month),
            by_country=dict(by_country),
            by_spot=by_spot_data,
            clusters=clusters,
            bbox_coverage=bbox_coverage,
        )
