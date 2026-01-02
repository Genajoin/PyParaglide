"""
Tests for MeteoAnalyzer.
"""

from datetime import date, timedelta
from pathlib import Path
import pytest

from pyparaglide.analysis import MeteoAnalyzer


@pytest.fixture
def sample_gfs(tmp_path: Path) -> Path:
    """Create sample GFS directory structure."""
    gfs_dir = tmp_path / "gfs" / "anl"
    gfs_dir.mkdir(parents=True)

    # Create month directory
    month_dir = gfs_dir / "2024-06"
    month_dir.mkdir()

    # Create complete days (06h, 12h, 18h)
    for day in [1, 2, 3]:
        for hour in ["06", "12", "18"]:
            filename = f"gfsanl_3_202406{day:02d}_{hour}00_000.grb2"
            (month_dir / filename).touch()

    # Create incomplete day (missing 18h)
    for hour in ["06", "12"]:
        filename = f"gfsanl_3_20240604_{hour}00_000.grb2"
        (month_dir / filename).touch()

    return gfs_dir


def test_scan_available_days(sample_gfs: Path):
    """Test scanning GFS directory."""
    analyzer = MeteoAnalyzer(gfs_dir=sample_gfs)
    days = analyzer.scan_available_days()

    # Should find 3 complete days
    assert len(days) == 3
    assert date(2024, 6, 1) in days
    assert date(2024, 6, 2) in days
    assert date(2024, 6, 3) in days
    # Day 4 is incomplete, should not be included
    assert date(2024, 6, 4) not in days


def test_scan_available_days_empty(tmp_path: Path):
    """Test scanning empty directory."""
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    analyzer = MeteoAnalyzer(gfs_dir=empty_dir)
    days = analyzer.scan_available_days()

    # No days, but no error
    assert len(days) == 0


def test_scan_available_days_nonexistent():
    """Test scanning non-existent directory."""
    analyzer = MeteoAnalyzer(gfs_dir=Path("/nonexistent/path"))
    with pytest.raises(ValueError, match="GFS directory not found"):
        analyzer.scan_available_days()


def test_scan_available_days_with_date_ranges(sample_gfs: Path):
    """Test scanning with date ranges filter."""
    analyzer = MeteoAnalyzer(
        gfs_dir=sample_gfs,
        date_ranges=[
            (date(2024, 6, 1), date(2024, 6, 2)),  # Only first 2 days
        ],
    )
    days = analyzer.scan_available_days()

    # Should only return days in the date range
    assert len(days) == 2
    assert date(2024, 6, 1) in days
    assert date(2024, 6, 2) in days
    assert date(2024, 6, 3) not in days


def test_is_date_in_ranges():
    """Test date range filtering."""
    analyzer = MeteoAnalyzer(gfs_dir=Path("/tmp"))

    # No ranges - all dates pass
    assert analyzer._is_date_in_ranges(date(2024, 6, 1), [])
    assert analyzer._is_date_in_ranges(date(2024, 12, 31), [])

    # With ranges
    ranges = [
        (date(2024, 6, 1), date(2024, 6, 30)),
        (date(2024, 8, 1), date(2024, 8, 31)),
    ]

    assert analyzer._is_date_in_ranges(date(2024, 6, 15), ranges)
    assert analyzer._is_date_in_ranges(date(2024, 8, 15), ranges)
    assert not analyzer._is_date_in_ranges(date(2024, 7, 15), ranges)


def test_check_completeness(sample_gfs: Path):
    """Test completeness check."""
    analyzer = MeteoAnalyzer(gfs_dir=sample_gfs)

    # Expected: 4 days (3 complete + 1 incomplete)
    expected_days = [
        date(2024, 6, 1),
        date(2024, 6, 2),
        date(2024, 6, 3),
        date(2024, 6, 4),
    ]

    result = analyzer.check_completeness(expected_days=expected_days)

    # Check available and missing
    assert len(result.available_days) == 3
    assert len(result.missing_days) == 1
    assert date(2024, 6, 4) in result.missing_days

    # Check percentage
    assert result.complete_percentage == 75.0  # 3/4 = 75%

    # Check by_month
    assert "2024-06" in result.by_month
    stats = result.by_month["2024-06"]
    assert stats.expected_days == 4
    assert stats.available_days == 3
    assert len(stats.missing_days) == 1


def test_check_completeness_with_date_ranges(sample_gfs: Path):
    """Test completeness check with date ranges from init."""
    analyzer = MeteoAnalyzer(
        gfs_dir=sample_gfs,
        date_ranges=[
            (date(2024, 6, 1), date(2024, 6, 4)),
        ],
    )

    result = analyzer.check_completeness()

    # Should use date_ranges to determine expected days
    assert len(result.available_days) == 3
    assert len(result.missing_days) == 1
    assert result.complete_percentage == 75.0


def test_check_completeness_no_expected_days(sample_gfs: Path):
    """Test completeness check without expected days."""
    analyzer = MeteoAnalyzer(gfs_dir=sample_gfs)

    result = analyzer.check_completeness(expected_days=None)

    # Should use available days as expected (no missing)
    assert len(result.available_days) == 3
    assert len(result.missing_days) == 0
    assert result.complete_percentage == 100.0


def test_month_stats_completeness_percentage():
    """Test MonthStats completeness percentage calculation."""
    from pyparaglide.analysis.meteo_analyzer import MonthStats

    # 100% complete
    stats = MonthStats(month="2024-06", expected_days=30, available_days=30, missing_days=[])
    assert stats.completeness_percentage == 100.0

    # 50% complete
    stats = MonthStats(month="2024-06", expected_days=30, available_days=15, missing_days=[])
    assert stats.completeness_percentage == 50.0

    # 0% complete
    stats = MonthStats(month="2024-06", expected_days=30, available_days=0, missing_days=[])
    assert stats.completeness_percentage == 0.0

    # No expected days
    stats = MonthStats(month="2024-06", expected_days=0, available_days=0, missing_days=[])
    assert stats.completeness_percentage == 0.0


def test_check_completeness_multiple_months(tmp_path: Path):
    """Test completeness check across multiple months."""
    gfs_dir = tmp_path / "gfs" / "anl"
    gfs_dir.mkdir(parents=True)

    # Create two months
    for month in ["2024-06", "2024-07"]:
        month_dir = gfs_dir / month
        month_dir.mkdir()

        # June: all days complete
        # July: only half days complete
        if month == "2024-06":
            days = 30
        else:
            days = 15

        for day in range(1, days + 1):
            for hour in ["06", "12", "18"]:
                filename = f"gfsanl_3_{month.replace('-', '')}{day:02d}_{hour}00_000.grb2"
                (month_dir / filename).touch()

    analyzer = MeteoAnalyzer(gfs_dir=gfs_dir)

    # Expected: full June, full July
    expected_days = []
    for day in range(1, 31):
        expected_days.append(date(2024, 6, day))
    for day in range(1, 32):
        expected_days.append(date(2024, 7, day))

    result = analyzer.check_completeness(expected_days=expected_days)

    # Check by_month
    assert len(result.by_month) == 2

    # June: 100% complete
    june_stats = result.by_month["2024-06"]
    assert june_stats.expected_days == 30
    assert june_stats.available_days == 30
    assert june_stats.completeness_percentage == 100.0

    # July: ~48% complete (15/31)
    july_stats = result.by_month["2024-07"]
    assert july_stats.expected_days == 31
    assert july_stats.available_days == 15
    assert july_stats.completeness_percentage == pytest.approx(48.39, rel=0.1)
