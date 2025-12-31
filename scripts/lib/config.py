"""Configuration management for Paraglidable scripts."""

import os
from pathlib import Path
from typing import Optional

DEFAULT_DB_URL = "postgresql://paraglidable:paraglidable@paraglidable-postgres:5432/paraglidable"


class Config:
    """Centralized configuration with .env file support."""

    def __init__(self, env_file: Optional[Path] = None):
        self.env_file = env_file or Path.cwd() / ".env"
        self._load_env()

    def _load_env(self) -> None:
        """Load .env file if exists."""
        if self.env_file.exists():
            try:
                from dotenv import load_dotenv
                load_dotenv(self.env_file)
            except ImportError:
                pass  # python-dotenv is optional

    @property
    def db_url(self) -> str:
        """Get database URL from environment or default."""
        return os.environ.get("IGC_DB_URL", DEFAULT_DB_URL)

    @property
    def ucar_email(self) -> str:
        """Get UCAR RDA email from environment."""
        email = os.environ.get("UCAR_EMAIL")
        if not email:
            raise ValueError("UCAR_EMAIL not set in environment or .env")
        return email

    @property
    def ucar_password(self) -> str:
        """Get UCAR RDA password from environment."""
        pwd = os.environ.get("UCAR_PASS")
        if not pwd:
            raise ValueError("UCAR_PASS not set in environment or .env")
        return pwd

    @property
    def training_dates(self) -> str | None:
        """Get training date ranges (comma-separated start:end pairs) or None."""
        return os.environ.get("TRAINING_DATES")

    @property
    def training_bbox(self) -> str | None:
        """Get training bbox as 'lat_min,lat_max,lon_min,lon_max' or None."""
        return os.environ.get("TRAINING_BBOX")

    @property
    def min_flights_per_spot(self) -> int:
        """Get minimum flights per spot for PKL filtering."""
        return int(os.environ.get("MIN_FLIGHTS_PER_SPOT", "200"))

    @property
    def flights_dir(self) -> str:
        """Get flights data directory."""
        return os.environ.get("FLIGHTS_DIR", "data/flights")

    @property
    def pkl_dir(self) -> str:
        """Get PKL data directory."""
        return os.environ.get("PKL_DIR", "neural_network/bin/data")

    @property
    def spot_cluster_distance_km(self) -> float:
        """Get spot clustering distance in km (0 = disabled)."""
        return float(os.environ.get("SPOT_CLUSTER_DISTANCE_KM", "0"))


def get_default_db_url() -> str:
    """Get default database URL from environment or fallback."""
    return os.environ.get("IGC_DB_URL", DEFAULT_DB_URL)
