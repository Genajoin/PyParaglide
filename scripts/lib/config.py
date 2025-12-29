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


def get_default_db_url() -> str:
    """Get default database URL from environment or fallback."""
    return os.environ.get("IGC_DB_URL", DEFAULT_DB_URL)
