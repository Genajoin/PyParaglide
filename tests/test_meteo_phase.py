"""
Unit tests for meteo_phase auto-detection of new complete days.
"""

import pickle
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pyparaglide.preprocessing.phases.meteo_phase import BuildMeteoPhase


@pytest.fixture
def temp_gfs_dir(tmp_path):
    """Create a temporary GFS directory structure."""
    gfs_dir = tmp_path / "gfs" / "anl"
    gfs_dir.mkdir(parents=True)

    # Create 2021-05 directory with complete day (2021-05-02)
    may_dir = gfs_dir / "2021-05"
    may_dir.mkdir()

    # Create complete day: 2021-05-02 (all 3 hours)
    for hour in [6, 12, 18]:
        grb_file = may_dir / f"gfsanl_3_20210502_{hour:02d}00_000.grb2"
        grb_file.write_bytes(b"fake grib content")

    # Create incomplete day: 2021-05-01 (only 2 hours)
    for hour in [12, 18]:  # Missing 06:00
        grb_file = may_dir / f"gfsanl_3_20210501_{hour:02d}00_000.grb2"
        grb_file.write_bytes(b"fake grib content")

    return gfs_dir


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create a temporary output directory."""
    output_dir = tmp_path / "pkl"
    output_dir.mkdir(parents=True)
    return output_dir


@pytest.fixture
def sample_cells_latlon():
    """Create sample cell coordinates."""
    return [(45.0, 13.0), (45.0, 14.0), (46.0, 13.0), (46.0, 14.0)]


class TestScanMeteoDaysQuick:
    """Test suite for _scan_meteo_days_quick method."""

    def test_scan_finds_complete_days(self, temp_gfs_dir, temp_output_dir, sample_cells_latlon):
        """Test that quick scan finds only complete days."""
        phase = BuildMeteoPhase(
            bbox=(45.0, 47.0, 13.0, 15.0),
            gfs_dir=temp_gfs_dir,
            cells_latlon=sample_cells_latlon,
            out_dir=temp_output_dir,
        )

        complete_days = phase._scan_meteo_days_quick()

        # Should only find 2021-05-02 (complete day)
        assert len(complete_days) == 1
        assert date(2021, 5, 2) in complete_days
        assert date(2021, 5, 1) not in complete_days  # Incomplete

    def test_scan_returns_empty_when_no_files(self, tmp_path, temp_output_dir, sample_cells_latlon):
        """Test that quick scan returns empty list when no GRIB files exist."""
        empty_gfs_dir = tmp_path / "empty_gfs"
        empty_gfs_dir.mkdir()

        phase = BuildMeteoPhase(
            bbox=(45.0, 47.0, 13.0, 15.0),
            gfs_dir=empty_gfs_dir,
            cells_latlon=sample_cells_latlon,
            out_dir=temp_output_dir,
        )

        complete_days = phase._scan_meteo_days_quick()
        assert complete_days == []


class TestAutoDetectionNewDays:
    """Test suite for auto-detection of new complete days in execute method."""

    def test_detects_new_complete_day(self, temp_gfs_dir, temp_output_dir, sample_cells_latlon):
        """
        Test scenario: existing PKL has 1 day, GFS dir has 2 complete days.
        Should detect the new day and process it.
        """
        # Create existing meteo_days.pkl with only 2021-05-02
        existing_days = [date(2021, 5, 2)]
        with open(temp_output_dir / "meteo_days.pkl", 'wb') as f:
            pickle.dump(existing_days, f)

        # Create other required PKL files to satisfy _check_existing_pkl
        meteo_params = [(6, 'Precipitable water', [('entireAtmosphere', 0)])] * 195
        with open(temp_output_dir / "meteo_params.pkl", 'wb') as f:
            pickle.dump(meteo_params, f)

        # Create meteo_content_by_cell_day.pkl
        meteo_content = np.zeros((4, 195), dtype=np.float32)  # 4 cells * 195 params
        with open(temp_output_dir / "meteo_content_by_cell_day.pkl", 'wb') as f:
            pickle.dump(meteo_content, f)

        # Add a new complete day to GFS directory: 2021-05-03
        may_dir = temp_gfs_dir / "2021-05"
        for hour in [6, 12, 18]:
            grb_file = may_dir / f"gfsanl_3_20210503_{hour:02d}00_000.grb2"
            grb_file.write_bytes(b"fake grib content")

        # Mock the expensive operations
        with patch.object(BuildMeteoPhase, '_build_meteo_content') as mock_build:
            with patch.object(BuildMeteoPhase, '_build_meteo_params') as mock_params:
                mock_params.return_value = meteo_params
                phase = BuildMeteoPhase(
                    bbox=(45.0, 47.0, 13.0, 15.0),
                    gfs_dir=temp_gfs_dir,
                    cells_latlon=sample_cells_latlon,
                    out_dir=temp_output_dir,
                    date_ranges=[(date(2021, 5, 1), date(2021, 5, 31))],
                )

                # Execute should detect new day
                result = phase.execute()

                # Should have called _build_meteo_content (not skipped)
                assert mock_build.called, "Should process new day, not skip"

    def test_skips_when_no_new_days(self, temp_gfs_dir, temp_output_dir, sample_cells_latlon):
        """
        Test scenario: existing PKL has 1 day, GFS dir has same 1 complete day.
        Should skip processing.
        """
        # Create existing meteo_days.pkl with 2021-05-02
        existing_days = [date(2021, 5, 2)]
        with open(temp_output_dir / "meteo_days.pkl", 'wb') as f:
            pickle.dump(existing_days, f)

        # Create other required PKL files
        meteo_params = [(6, 'Precipitable water', [('entireAtmosphere', 0)])] * 195
        with open(temp_output_dir / "meteo_params.pkl", 'wb') as f:
            pickle.dump(meteo_params, f)

        meteo_content = np.zeros((4, 195), dtype=np.float32)
        with open(temp_output_dir / "meteo_content_by_cell_day.pkl", 'wb') as f:
            pickle.dump(meteo_content, f)

        # Create metadata file with matching training_dates
        import json
        metadata = {
            "bbox": [45.0, 47.0, 13.0, 15.0],
            "training_dates": "2021-05-01:2021-05-31"
        }
        with open(temp_output_dir / "dataset_config.json", 'w') as f:
            json.dump(metadata, f)

        # Mock the expensive operations
        with patch.object(BuildMeteoPhase, '_build_meteo_content') as mock_build:
            with patch.object(BuildMeteoPhase, '_build_meteo_params') as mock_params:
                mock_params.return_value = meteo_params
                phase = BuildMeteoPhase(
                    bbox=(45.0, 47.0, 13.0, 15.0),
                    gfs_dir=temp_gfs_dir,
                    cells_latlon=sample_cells_latlon,
                    out_dir=temp_output_dir,
                    date_ranges=[(date(2021, 5, 1), date(2021, 5, 31))],
                )

                # Execute should skip
                result = phase.execute()

                # Should NOT have called _build_meteo_content (skipped)
                assert not mock_build.called, "Should skip when no new days"
                assert result == existing_days

    def test_processes_when_force_flag(self, temp_gfs_dir, temp_output_dir, sample_cells_latlon):
        """
        Test scenario: even when no new days, --force flag triggers rebuild.
        """
        # Create existing meteo_days.pkl
        existing_days = [date(2021, 5, 2)]
        with open(temp_output_dir / "meteo_days.pkl", 'wb') as f:
            pickle.dump(existing_days, f)

        # Create other required PKL files
        meteo_params = [(6, 'Precipitable water', [('entireAtmosphere', 0)])] * 195
        with open(temp_output_dir / "meteo_params.pkl", 'wb') as f:
            pickle.dump(meteo_params, f)

        meteo_content = np.zeros((4, 195), dtype=np.float32)
        with open(temp_output_dir / "meteo_content_by_cell_day.pkl", 'wb') as f:
            pickle.dump(meteo_content, f)

        # Mock the expensive operations
        with patch.object(BuildMeteoPhase, '_build_meteo_content') as mock_build:
            with patch.object(BuildMeteoPhase, '_build_meteo_params') as mock_params:
                mock_params.return_value = meteo_params
                phase = BuildMeteoPhase(
                    bbox=(45.0, 47.0, 13.0, 15.0),
                    gfs_dir=temp_gfs_dir,
                    cells_latlon=sample_cells_latlon,
                    out_dir=temp_output_dir,
                    date_ranges=[(date(2021, 5, 1), date(2021, 5, 31))],
                    force=True,  # Force flag
                )

                # Execute should process despite no new days
                result = phase.execute()

                # Should have called _build_meteo_content (forced rebuild)
                assert mock_build.called, "Should process with --force flag"


class TestScanMeteoDaysDetailed:
    """Test suite for detailed _scan_meteo_days method with output."""

    def test_shows_newly_complete_days(self, temp_gfs_dir, temp_output_dir, sample_cells_latlon, capsys):
        """
        Test that _scan_meteo_days shows message about newly complete days.
        """
        # Create existing meteo_days.pkl with 2021-05-02
        existing_days = [date(2021, 5, 2)]
        with open(temp_output_dir / "meteo_days.pkl", 'wb') as f:
            pickle.dump(existing_days, f)

        # Add a new complete day: 2021-05-03
        may_dir = temp_gfs_dir / "2021-05"
        for hour in [6, 12, 18]:
            grb_file = may_dir / f"gfsanl_3_20210503_{hour:02d}00_000.grb2"
            grb_file.write_bytes(b"fake grib content")

        phase = BuildMeteoPhase(
            bbox=(45.0, 47.0, 13.0, 15.0),
            gfs_dir=temp_gfs_dir,
            cells_latlon=sample_cells_latlon,
            out_dir=temp_output_dir,
            date_ranges=[(date(2021, 5, 1), date(2021, 5, 31))],
        )

        result = phase._scan_meteo_days()

        captured = capsys.readouterr()
        output = captured.out

        # Should show auto-detected message
        assert "Auto-detected" in output or "newly complete" in output.lower()
        assert "2021-05-03" in output

    def test_filters_by_date_ranges(self, temp_gfs_dir, temp_output_dir, sample_cells_latlon):
        """
        Test that _scan_meteo_days filters by date_ranges correctly.
        """
        # Add complete days outside the training range
        may_dir = temp_gfs_dir / "2021-05"
        for day in [1, 2, 3, 15]:  # Day 1 is incomplete, others complete
            for hour in [6, 12, 18]:
                if day != 1:  # Skip day 1 hours (keep it incomplete)
                    grb_file = may_dir / f"gfsanl_3_202105{day:02d}_{hour:02d}00_000.grb2"
                    grb_file.write_bytes(b"fake grib content")

        phase = BuildMeteoPhase(
            bbox=(45.0, 47.0, 13.0, 15.0),
            gfs_dir=temp_gfs_dir,
            cells_latlon=sample_cells_latlon,
            out_dir=temp_output_dir,
            date_ranges=[(date(2021, 5, 1), date(2021, 5, 10))],  # Only first 10 days
        )

        result = phase._scan_meteo_days()

        # Should only include days 2-3 (day 1 incomplete, day 15 out of range)
        assert date(2021, 5, 2) in result
        assert date(2021, 5, 3) in result
        assert date(2021, 5, 15) not in result
        assert date(2021, 5, 1) not in result  # Incomplete


@pytest.mark.integration
class TestAutoDetectionIntegration:
    """Integration tests for auto-detection with realistic scenarios."""

    def test_scenario_incomplete_becomes_complete(self, temp_gfs_dir, temp_output_dir, sample_cells_latlon):
        """
        Test real scenario: day is incomplete, user downloads missing file,
        re-runs command, and it auto-detects the newly complete day.
        """
        # Step 1: Initial state - 2021-05-01 is incomplete (missing 06:00)
        # Already set up in temp_gfs_dir fixture

        # Step 2: Create PKL with only 2021-05-02
        existing_days = [date(2021, 5, 2)]
        with open(temp_output_dir / "meteo_days.pkl", 'wb') as f:
            pickle.dump(existing_days, f)

        meteo_params = [(6, 'Precipitable water', [('entireAtmosphere', 0)])] * 195
        with open(temp_output_dir / "meteo_params.pkl", 'wb') as f:
            pickle.dump(meteo_params, f)

        meteo_content = np.zeros((4, 195), dtype=np.float32)
        with open(temp_output_dir / "meteo_content_by_cell_day.pkl", 'wb') as f:
            pickle.dump(meteo_content, f)

        # Step 3: Simulate user downloading missing file (add 06:00 for 2021-05-01)
        may_dir = temp_gfs_dir / "2021-05"
        missing_file = may_dir / "gfsanl_3_20210501_0600_000.grb2"
        missing_file.write_bytes(b"downloaded grib content")

        # Step 4: Re-run - should auto-detect newly complete day
        with patch.object(BuildMeteoPhase, '_build_meteo_content') as mock_build:
            with patch.object(BuildMeteoPhase, '_build_meteo_params') as mock_params:
                mock_params.return_value = meteo_params
                phase = BuildMeteoPhase(
                    bbox=(45.0, 47.0, 13.0, 15.0),
                    gfs_dir=temp_gfs_dir,
                    cells_latlon=sample_cells_latlon,
                    out_dir=temp_output_dir,
                    date_ranges=[(date(2021, 5, 1), date(2021, 5, 31))],
                )

                result = phase.execute()

                # Should have processed the new day
                assert mock_build.called
                # Result should include both days
                assert date(2021, 5, 1) in result
                assert date(2021, 5, 2) in result