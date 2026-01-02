"""
Meteo data analyzer.

Analyzes downloaded GFS GRIB files to check data completeness
and help users determine optimal training dates.
"""

import os
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any


@dataclass
class MonthStats:
    """Statistics for a month."""
    month: str  # "2021-06"
    expected_days: int
    available_days: int
    missing_days: list[date]

    @property
    def completeness_percentage(self) -> float:
        """Percentage completeness."""
        if self.expected_days == 0:
            return 0.0
        return (self.available_days / self.expected_days) * 100


@dataclass
class MeteoCompletenessResult:
    """Result of GFS data completeness check."""
    available_days: list[date]
    missing_days: list[date]
    complete_percentage: float
    by_month: dict[str, MonthStats]


class MeteoAnalyzer:
    """Analyze downloaded GFS GRIB files."""

    def __init__(
        self,
        gfs_dir: Path | str,
        date_ranges: list[tuple[date, date]] | None = None,
    ):
        """
        Args:
            gfs_dir: Path to GFS GRIB files (e.g., data/gfs/anl/)
            date_ranges: Optional list of (start_date, end_date) to filter
        """
        self.gfs_dir = Path(gfs_dir)
        self.date_ranges = date_ranges

    def scan_available_days(self) -> list[date]:
        """
        Scan GFS directory for available days with complete data.

        A day is considered complete if it has GRIB files for
        06h, 12h, and 18h forecasts.

        Returns:
            Sorted list of dates with complete data
        """
        if not self.gfs_dir.exists():
            raise ValueError(f"GFS directory not found: {self.gfs_dir}")

        all_meteo_days = {}

        for month_dir in sorted(os.listdir(self.gfs_dir)):
            month_path = self.gfs_dir / month_dir
            if not month_path.is_dir():
                continue

            files_by_date = {}
            for filename in os.listdir(month_path):
                if filename.startswith("gfsanl_3_") and filename.endswith(".grb2"):
                    parts = filename.split("_")
                    if len(parts) >= 4:
                        date_str = parts[2]
                        hour_str = parts[3][:2]

                        if date_str not in files_by_date:
                            files_by_date[date_str] = set()
                        files_by_date[date_str].add(hour_str)

            for date_str, hours in sorted(files_by_date.items()):
                if "06" in hours and "12" in hours and "18" in hours:
                    year = int(date_str[:4])
                    month = int(date_str[4:6])
                    day = int(date_str[6:8])
                    day_date = date(year, month, day)
                    all_meteo_days[day_date] = True

        available_days = sorted(all_meteo_days.keys())

        # Filter by date ranges if specified
        if self.date_ranges:
            available_days = [
                d for d in available_days if self._is_date_in_ranges(d, self.date_ranges)
            ]

        return available_days

    def _is_date_in_ranges(
        self,
        target_date: date,
        ranges: list[tuple[date, date]],
    ) -> bool:
        """Check if date falls within any range."""
        if not ranges:
            return True
        for start, end in ranges:
            if start <= target_date <= end:
                return True
        return False

    def check_completeness(
        self,
        expected_days: list[date] | None = None,
    ) -> MeteoCompletenessResult:
        """
        Check GFS data completeness.

        Args:
            expected_days: Optional list of expected dates (e.g., from TRAINING_DATES)
                          If None, uses date_ranges from __init__

        Returns:
            MeteoCompletenessResult with completeness statistics
        """
        # Scan available days
        available_days = self.scan_available_days()
        available_set = set(available_days)

        # Determine expected days
        if expected_days is None:
            if self.date_ranges:
                # Generate expected days from date_ranges
                expected_days = []
                for start, end in self.date_ranges:
                    current = start
                    while current <= end:
                        expected_days.append(current)
                        current += timedelta(days=1)
                expected_days = sorted(set(expected_days))
            else:
                # Use available days as expected (no missing days)
                expected_days = available_days

        # Find missing days
        missing_days = [d for d in expected_days if d not in available_set]

        # Calculate completeness percentage
        if expected_days:
            complete_percentage = (len(available_days) / len(expected_days)) * 100
        else:
            complete_percentage = 0.0

        # Group by month
        by_month: dict[str, MonthStats] = {}
        for d in expected_days:
            month_key = d.strftime("%Y-%m")
            if month_key not in by_month:
                by_month[month_key] = MonthStats(
                    month=month_key,
                    expected_days=0,
                    available_days=0,
                    missing_days=[],
                )
            by_month[month_key].expected_days += 1

        for d in available_days:
            month_key = d.strftime("%Y-%m")
            if month_key in by_month:
                by_month[month_key].available_days += 1

        for d in missing_days:
            month_key = d.strftime("%Y-%m")
            if month_key in by_month:
                by_month[month_key].missing_days.append(d)

        return MeteoCompletenessResult(
            available_days=available_days,
            missing_days=missing_days,
            complete_percentage=complete_percentage,
            by_month=by_month,
        )
