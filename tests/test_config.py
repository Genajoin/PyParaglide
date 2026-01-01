"""
Tests for configuration module.
"""

import os
import pytest
from pydantic import ValidationError


class TestSettings:
    """Test Settings configuration."""

    def test_default_settings(self, monkeypatch):
        """Test default settings values."""
        # Set required env vars
        monkeypatch.setenv("PYPARAGLIDE_TRAINING_DATES", "2024-08-01:2024-08-31")
        monkeypatch.setenv("PYPARAGLIDE_BBOX", "45,46,13,14")
        monkeypatch.setenv("PYPARAGLIDE_GFS_DIR", "data/gfs")

        from pyparaglide.config import Settings

        settings = Settings()

        assert settings.training_dates == "2024-08-01:2024-08-31"
        assert settings.bbox == "45,46,13,14"
        assert settings.gfs_dir == "data/gfs"

    def test_missing_required_env_var(self, monkeypatch):
        """Test that missing required env vars raise error."""
        # Clear all PYPARAGLIDE env vars
        for key in list(os.environ):
            if key.startswith("PYPARAGLIDE_"):
                monkeypatch.delenv(key, raising=False)

        from pyparaglide.config import Settings

        # Settings without required env vars should fail
        try:
            Settings()
            assert False, "Should have raised an error"
        except (ValidationError, Exception):
            # Expected
            pass

    def test_bbox_parsing(self, monkeypatch):
        """Test BBOX parsing."""
        monkeypatch.setenv("PYPARAGLIDE_TRAINING_DATES", "2024-08-01:2024-08-31")
        monkeypatch.setenv("PYPARAGLIDE_BBOX", "45.0,46.5,13.2,14.8")
        monkeypatch.setenv("PYPARAGLIDE_GFS_DIR", "data/gfs")

        from pyparaglide.config import Settings

        settings = Settings()
        assert settings.bbox == "45.0,46.5,13.2,14.8"

    def test_training_dates_single_range(self, monkeypatch):
        """Test single date range."""
        monkeypatch.setenv("PYPARAGLIDE_TRAINING_DATES", "2024-06-01:2024-08-31")
        monkeypatch.setenv("PYPARAGLIDE_BBOX", "45,46,13,14")
        monkeypatch.setenv("PYPARAGLIDE_GFS_DIR", "data/gfs")

        from pyparaglide.config import Settings

        settings = Settings()
        assert settings.training_dates == "2024-06-01:2024-08-31"

    def test_training_dates_multiple_ranges(self, monkeypatch):
        """Test multiple date ranges (comma-separated)."""
        monkeypatch.setenv("PYPARAGLIDE_TRAINING_DATES", "2021-06-01:2021-08-31,2022-06-01:2022-08-31")
        monkeypatch.setenv("PYPARAGLIDE_BBOX", "45,46,13,14")
        monkeypatch.setenv("PYPARAGLIDE_GFS_DIR", "data/gfs")

        from pyparaglide.config import Settings

        settings = Settings()
        assert settings.training_dates == "2021-06-01:2021-08-31,2022-06-01:2022-08-31"

    def test_get_settings_singleton(self, monkeypatch):
        """Test get_settings returns singleton instance."""
        monkeypatch.setenv("PYPARAGLIDE_TRAINING_DATES", "2024-08-01:2024-08-31")
        monkeypatch.setenv("PYPARAGLIDE_BBOX", "45,46,13,14")
        monkeypatch.setenv("PYPARAGLIDE_GFS_DIR", "data/gfs")

        from pyparaglide.config import get_settings

        settings1 = get_settings()
        settings2 = get_settings()

        # Should return same instance
        assert settings1 is settings2
