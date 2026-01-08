"""
PyParaglide Configuration.

Pydantic-based settings management with environment variable support.
All configuration is loaded from environment variables with PYPARAGLIDE_ prefix.
"""

import datetime as dt
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


def parse_date_ranges(
    dates_str: str | None,
) -> list[tuple[dt.date, dt.date]]:
    """
    Parse date ranges string (same format as TRAINING_DATES in .env).

    Format: "YYYY-MM-DD:YYYY-MM-DD,YYYY-MM-DD:YYYY-MM-DD,..."

    Args:
        dates_str: Date ranges string or None

    Returns:
        List of (start_date, end_date) tuples

    Raises:
        ValueError: If format is invalid

    Examples:
        >>> parse_date_ranges("2024-06-01:2024-08-31")
        [(date(2024, 6, 1), date(2024, 8, 31))]

        >>> parse_date_ranges("2024-06-01:2024-08-31,2025-06-01:2025-08-31")
        [(date(2024, 6, 1), date(2024, 8, 31)), (date(2025, 6, 1), date(2025, 8, 31))]
    """
    if not dates_str:
        return []

    ranges = []
    for part in dates_str.split(","):
        part = part.strip()
        if not part:
            continue
        dates = part.split(":")
        if len(dates) != 2:
            raise ValueError(f"Invalid date range: {part} (expected format: YYYY-MM-DD:YYYY-MM-DD)")
        try:
            start = dt.datetime.strptime(dates[0].strip(), "%Y-%m-%d").date()
            end = dt.datetime.strptime(dates[1].strip(), "%Y-%m-%d").date()
            if start > end:
                raise ValueError(f"Start date ({start}) must be before or equal to end date ({end})")
            ranges.append((start, end))
        except ValueError as e:
            raise ValueError(f"Invalid date format in range: {part} ({e})")

    return ranges


class Settings(BaseSettings):
    """
    PyParaglide application settings.

    All settings can be configured via environment variables with PYPARAGLIDE_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="PYPARAGLIDE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ==============================================================================
    # Geographic Bounds
    # ==============================================================================
    # Bounding box in GeoJSON RFC 7946 format (lon_min,lat_min,lon_max,lat_max)
    bbox: str = "13,45,15,47"

    # ==============================================================================
    # Data Directories
    # ==============================================================================
    # Directory for GFS GRIB weather files
    gfs_dir: str = "data/gfs/anl"

    # Directory for PKL dataset files (training data)
    pkl_dir: str = "data/pkl"

    # Directory containing trained model weights
    models_dir: str = "data/models"

    # Output directory for forecast results
    output_dir: str = "output/forecasts"

    # Directory for flight data (xContest JSON files)
    flights_dir: str = "data/flights"

    # Directory for elevation tiles
    elevation_dir: str = "data/elevation"

    # Elevation data source: SRTM1 (30m) or SRTM3 (90m)
    elevation_source: str = "SRTM3"

    # Auto-download elevation if missing during build-dataset
    elevation_auto_download: bool = True

    # ==============================================================================
    # Forecast Parameters
    # ==============================================================================
    # Number of days to forecast
    forecast_days: int = 10

    # ==============================================================================
    # Training Parameters
    # ==============================================================================
    # Date ranges for training data (comma-separated: YYYY-MM-DD:YYYY-MM-DD,...)
    training_dates: str = (
        "2021-06-01:2021-08-31,2022-06-01:2022-08-31,2023-06-01:2023-08-31"
    )

    # Minimum number of flights required per spot for training
    min_flights_per_spot: int = 200

    # Minimum number of flights required per cell for training (0 = no filtering)
    min_flights_per_cell: int = 20

    # Spot clustering radius in kilometers (0 = disabled)
    spot_cluster_distance_km: float = 2.0

    # ==============================================================================
    # Evaluation Parameters
    # ==============================================================================
    # Default year for evaluate command and experiment test metrics
    evaluate_year: int = 2025

    # Default threshold for evaluate command
    evaluate_threshold: float = 0.5

    # ==============================================================================
    # Processing
    # ==============================================================================
    # Enable debug mode (keep temporary files, verbose output)
    debug: bool = False

    # Number of parallel workers for data processing
    workers: int = 4

    def parse_bbox(self) -> tuple[float, float, float, float]:
        """Parse bbox string into (lon_min, lat_min, lon_max, lat_max) following GeoJSON RFC 7946."""
        parts = [float(x.strip()) for x in self.bbox.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Invalid bbox format: {self.bbox}")

        lon_min, lat_min, lon_max, lat_max = parts

        # Validate coordinate ranges
        if not (-180 <= lon_min <= 180 and -180 <= lon_max <= 180):
            raise ValueError(f"Invalid longitude in bbox (must be -180..180): {self.bbox}")
        if not (-90 <= lat_min <= 90 and -90 <= lat_max <= 90):
            raise ValueError(f"Invalid latitude in bbox (must be -90..90): {self.bbox}")

        # Validate order (detect old format usage)
        if lon_min >= lon_max:
            raise ValueError(
                f"Invalid bbox: lon_min ({lon_min}) >= lon_max ({lon_max}). "
                f"Bbox must be in GeoJSON format: lon_min,lat_min,lon_max,lat_max. "
                f"If you're migrating from old format (lat_min,lat_max,lon_min,lon_max), "
                f"update your .env file."
            )
        if lat_min >= lat_max:
            raise ValueError(
                f"Invalid bbox: lat_min ({lat_min}) >= lat_max ({lat_max}). "
                f"Bbox must be in GeoJSON format: lon_min,lat_min,lon_max,lat_max."
            )

        return lon_min, lat_min, lon_max, lat_max

    def parse_training_dates(self) -> list[tuple[str, str]]:
        """
        Parse training_dates string into list of (start_date, end_date) tuples.

        Uses parse_date_ranges() utility function but returns strings for backward compatibility.
        """
        date_tuples = parse_date_ranges(self.training_dates)
        # Convert date objects to ISO format strings for backward compatibility
        return [(start.isoformat(), end.isoformat()) for start, end in date_tuples]


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are loaded only once per process.
    """
    return Settings()


# Global settings instance
settings = get_settings()
