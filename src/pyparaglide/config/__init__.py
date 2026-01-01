"""
PyParaglide Configuration.

Pydantic-based settings management with environment variable support.
All configuration is loaded from environment variables with PYPARAGLIDE_ prefix.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


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
    # Bounding box for forecast area: lat_min,lat_max,lon_min,lon_max
    bbox: str = "45,47,13,15"

    # ==============================================================================
    # Data Directories
    # ==============================================================================
    # Directory for GFS GRIB weather files
    gfs_dir: str = "data/gfs/anl"

    # Directory containing trained model weights
    models_dir: str = "data/models/CLASSIFICATION_1.0.0"

    # Output directory for forecast results
    output_dir: str = "output/forecasts"

    # Directory for flight data (xContest JSON files)
    flights_dir: str = "data/flights"

    # Directory for elevation tiles
    elevation_dir: str = "data/elevation"

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

    # Spot clustering radius in kilometers (0 = disabled)
    spot_cluster_distance_km: float = 2.0

    # ==============================================================================
    # Processing
    # ==============================================================================
    # Enable debug mode (keep temporary files, verbose output)
    debug: bool = False

    # Number of parallel workers for data processing
    workers: int = 4

    def parse_bbox(self) -> tuple[float, float, float, float]:
        """Parse bbox string into (lat_min, lat_max, lon_min, lon_max)."""
        parts = [float(x.strip()) for x in self.bbox.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Invalid bbox format: {self.bbox}")
        return parts[0], parts[1], parts[2], parts[3]

    def parse_training_dates(self) -> list[tuple[str, str]]:
        """Parse training_dates string into list of (start_date, end_date) tuples."""
        ranges = []
        for part in self.training_dates.split(","):
            part = part.strip()
            if not part:
                continue
            dates = part.split(":")
            if len(dates) != 2:
                raise ValueError(f"Invalid training date range: {part}")
            ranges.append((dates[0].strip(), dates[1].strip()))
        return ranges


@lru_cache
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Uses lru_cache to ensure settings are loaded only once per process.
    """
    return Settings()
