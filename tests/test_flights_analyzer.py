"""
Tests for FlightAnalyzer.
"""

import json
from pathlib import Path
from datetime import datetime
import pytest

from pyparaglide.analysis import FlightAnalyzer


@pytest.fixture
def sample_flights(tmp_path: Path) -> Path:
    """Create sample flight JSON files."""
    flights_dir = tmp_path / "flights"
    flights_dir.mkdir()

    flights = [
        {
            "takeoff": {
                "link": "https://www.xcontest.org/portal/map/?filter[point]=46.5 13.5",
                "countryIso": "SI",
                "id": "spot1",
                "name": "Kobala",
            },
            "pointStart": {"time": "2024-06-15T12:00:00Z"},
        },
        {
            "takeoff": {
                "link": "https://www.xcontest.org/portal/map/?filter[point]=45.8 13.9",
                "countryIso": "SI",
                "id": "spot2",
                "name": "Lijak",
            },
            "pointStart": {"time": "2024-07-20T14:30:00Z"},
        },
        {
            "takeoff": {
                "link": "https://www.xcontest.org/portal/map/?filter[point]=46.3 13.4",
                "countryIso": "SI",
                "id": "spot3",
                "name": "Stol",
            },
            "pointStart": {"time": "2024-08-05T10:00:00Z"},
        },
        # Flight with no coordinates
        {
            "takeoff": {
                "link": "",
                "countryIso": "SI",
                "id": "spot4",
                "name": "No Coords",
            },
            "pointStart": {"time": "2024-06-15T12:00:00Z"},
        },
    ]

    with open(flights_dir / "flights.json", "w") as f:
        json.dump(flights, f)

    return flights_dir


def test_flight_analyzer_load(sample_flights: Path):
    """Test loading flights."""
    analyzer = FlightAnalyzer(flights_dir=sample_flights)
    flights = analyzer.load_flights()
    assert len(flights) == 4


def test_flight_analyzer_load_empty(tmp_path: Path):
    """Test loading from empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    analyzer = FlightAnalyzer(flights_dir=empty_dir)
    with pytest.raises(ValueError, match="No JSON files found"):
        analyzer.load_flights()


def test_extract_coords():
    """Test coordinate extraction."""
    analyzer = FlightAnalyzer(flights_dir=Path("/tmp"))

    # Valid xContest URL
    lat, lon = analyzer.extract_coords("filter[point]=46.5 13.5")
    assert lat == 46.5
    assert lon == 13.5

    # Invalid URL
    lat, lon = analyzer.extract_coords("no coordinates here")
    assert lat is None
    assert lon is None


def test_parse_datetime():
    """Test datetime parsing."""
    analyzer = FlightAnalyzer(flights_dir=Path("/tmp"))

    # ISO format with Z
    dt = analyzer.parse_datetime("2024-06-15T12:00:00Z")
    assert dt == datetime(2024, 6, 15, 12, 0, 0)

    # ISO format with microseconds
    dt = analyzer.parse_datetime("2024-06-15T12:00:00.123456Z")
    assert dt == datetime(2024, 6, 15, 12, 0, 0, 123456)

    # Space format
    dt = analyzer.parse_datetime("2024-06-15 12:00:00")
    assert dt == datetime(2024, 6, 15, 12, 0, 0)

    # Invalid format
    dt = analyzer.parse_datetime("invalid")
    assert dt is None


def test_detect_cell_clusters():
    """Test cluster detection with BFS algorithm."""
    analyzer = FlightAnalyzer(flights_dir=Path("/tmp"))

    by_cell = {
        "46,13": 100,
        "46,14": 80,  # adjacent to 46,13
        "45,13": 60,  # adjacent to 46,13
        "50,20": 50,  # far from cluster (4+ degrees away)
    }

    clusters = analyzer.detect_cell_clusters(by_cell)

    # Should detect 2 clusters
    assert len(clusters) == 2

    # First cluster has most flights (100 + 80 + 60 = 240)
    assert clusters[0].flights == 240
    assert clusters[0].count == 3
    assert len(clusters[0].cells) == 3

    # Second cluster
    assert clusters[1].flights == 50
    assert clusters[1].count == 1


def test_detect_cell_clusters_with_custom_distance():
    """Test cluster detection with custom max distance."""
    analyzer = FlightAnalyzer(flights_dir=Path("/tmp"))

    by_cell = {
        "46,13": 100,
        "48,13": 80,  # 2 degrees away - should be same cluster with max_distance=2
    }

    # Default distance (2.0) - should be same cluster
    clusters = analyzer.detect_cell_clusters(by_cell, max_distance_degrees=2.0)
    assert len(clusters) == 1
    assert clusters[0].flights == 180

    # Smaller distance (1.0) - should be separate clusters
    clusters = analyzer.detect_cell_clusters(by_cell, max_distance_degrees=1.0)
    assert len(clusters) == 2


def test_analyze(sample_flights: Path):
    """Test full analysis."""
    analyzer = FlightAnalyzer(flights_dir=sample_flights)
    result = analyzer.analyze(min_flights_threshold=0)

    # Check totals (excluding no_coords flight)
    assert result.total_flights == 3
    assert len(result.by_cell) == 2  # Kobala and Stol are both in cell (46,13)
    assert len(result.by_month) == 3
    assert len(result.by_country) == 1  # All SI
    assert len(result.by_spot) == 3

    # Check cells
    assert "46,13" in result.by_cell  # Kobala and Stol
    assert "45,13" in result.by_cell  # Lijak

    # Check months
    assert 6 in result.by_month
    assert 7 in result.by_month
    assert 8 in result.by_month

    # Check spots
    assert "spot1" in result.by_spot
    assert result.by_spot["spot1"].name == "Kobala"
    assert result.by_spot["spot1"].count == 1

    # Check clusters
    assert len(result.clusters) > 0


def test_analyze_with_bbox_filter(sample_flights: Path):
    """Test analysis with bbox filter."""
    analyzer = FlightAnalyzer(
        flights_dir=sample_flights,
        bbox=(45.0, 46.0, 13.5, 14.5),  # Only Lijak (45.8, 13.9) inside
    )
    result = analyzer.analyze(min_flights_threshold=0)

    # Should only include flights inside bbox (only Lijak at 45.8, 13.9)
    assert result.total_flights == 1
    assert "45,13" in result.by_cell
    assert result.bbox_coverage is not None
    assert result.bbox_coverage.inside == 1
    assert result.bbox_coverage.outside == 2  # Kobala (46.5, 13.5) and Stol (46.3, 13.4) are outside (lat > 46.0)


def test_analyze_with_min_flights_threshold(sample_flights: Path):
    """Test analysis with min flights threshold."""
    analyzer = FlightAnalyzer(flights_dir=sample_flights)
    result = analyzer.analyze(min_flights_threshold=2)

    # All spots have only 1 flight, so none should pass the threshold
    assert len(result.by_spot) == 0


def test_find_top_spot_in_cluster(sample_flights: Path):
    """Test finding top spot in a cluster."""
    analyzer = FlightAnalyzer(flights_dir=sample_flights)
    result = analyzer.analyze(min_flights_threshold=0)

    if result.clusters:
        cluster = result.clusters[0]
        top_spot = analyzer.find_top_spot_in_cluster(cluster, result.by_spot)

        # Should return a spot name or "-"
        assert isinstance(top_spot, str)
        assert len(top_spot) > 0


def test_combine_clusters():
    """Test combining multiple clusters."""
    analyzer = FlightAnalyzer(flights_dir=Path("/tmp"))

    cluster1 = analyzer.detect_cell_clusters({"46,13": 100, "46,14": 80})[0]
    cluster2 = analyzer.detect_cell_clusters({"45,13": 60})[0]

    combined = analyzer.combine_clusters([cluster1, cluster2])

    assert combined["flights"] == 100 + 80 + 60
    assert combined["count"] == 3  # 2 + 1 cells
    assert combined["lat_min"] == 45
    assert combined["lat_max"] == 47  # 46 + 1
    assert combined["lon_min"] == 13
    assert combined["lon_max"] == 15  # 14 + 1
