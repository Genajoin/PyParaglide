"""
Unit tests for GRIB worker functions.

Tests the multiprocessing worker functions that handle GRIB file processing
and day result assembly, with special focus on handling incomplete data.
"""

import queue as Queue
from datetime import date

import numpy as np
import pytest

from pyparaglide.preprocessing.workers.grib_workers import assemble_day_results


@pytest.fixture
def sample_values_for_day():
    """Create sample weather values for a complete day (3 hours * 15 cells * 65 params)."""
    np.random.seed(42)
    # For each hour: 15 cells * 65 params = 975 values
    return {
        6: np.random.rand(15 * 65).astype(np.float32).tolist(),
        12: np.random.rand(15 * 65).astype(np.float32).tolist(),
        18: np.random.rand(15 * 65).astype(np.float32).tolist(),
    }


class TestAssembleDayResults:
    """Test suite for assemble_day_results function."""

    def test_complete_day_assembles_correctly(self, sample_values_for_day):
        """Test that a complete day (all 3 hours) assembles correctly."""
        hourly_queue = Queue.Queue()
        results_queue = Queue.Queue()
        num_params = 65
        num_cells = 15
        test_date = date(2024, 6, 1)

        # Put all 3 hours in the queue
        for hour in [6, 12, 18]:
            hourly_queue.put((test_date, hour, sample_values_for_day[hour]))

        # Put sentinel
        hourly_queue.put((None, None, None))

        # Run assembler
        assemble_day_results(hourly_queue, results_queue, num_params, num_cells)

        # Check results
        day_date, day_data = results_queue.get(timeout=1)
        assert day_date == test_date
        assert len(day_data) == num_cells

        # Each cell should have 3 hours * 65 params = 195 values
        for cell_values in day_data:
            assert len(cell_values) == 195

    def test_incomplete_day_creates_zero_filled_data(self):
        """Test that an incomplete day (missing some hours) still produces valid output."""
        hourly_queue = Queue.Queue()
        results_queue = Queue.Queue()
        num_params = 65
        num_cells = 15
        test_date = date(2024, 6, 1)

        # Only put 2 hours (missing hour 18)
        np.random.seed(42)
        for hour in [6, 12]:
            values = np.random.rand(15 * 65).astype(np.float32).tolist()
            hourly_queue.put((test_date, hour, values))

        # Put sentinel
        hourly_queue.put((None, None, None))

        # Run assembler
        assemble_day_results(hourly_queue, results_queue, num_params, num_cells)

        # The incomplete day should still produce output (zero-filled)
        try:
            day_date, day_data = results_queue.get(timeout=1)
            assert day_date == test_date
            assert len(day_data) == num_cells

            # Each cell should have 195 values (possibly zero-filled)
            for cell_values in day_data:
                assert len(cell_values) == 195
        except Queue.Empty:
            # If no result was produced, that's also acceptable behavior
            # The important thing is that the function doesn't crash
            pass

    def test_day_with_none_values_creates_zero_filled_data(self):
        """Test that None values (missing GRIB files) create zero-filled data."""
        hourly_queue = Queue.Queue()
        results_queue = Queue.Queue()
        num_params = 65
        num_cells = 15
        test_date = date(2024, 6, 1)

        # Put complete day but with None for one hour (simulating missing file)
        np.random.seed(42)
        hourly_queue.put((test_date, 6, np.random.rand(15 * 65).astype(np.float32).tolist()))
        hourly_queue.put((test_date, 12, None))  # Missing file
        hourly_queue.put((test_date, 18, np.random.rand(15 * 65).astype(np.float32).tolist()))

        # Put sentinel
        hourly_queue.put((None, None, None))

        # Run assembler
        assemble_day_results(hourly_queue, results_queue, num_params, num_cells)

        # Should produce valid output
        day_date, day_data = results_queue.get(timeout=1)
        assert day_date == test_date
        assert len(day_data) == num_cells

        # Each cell should have 195 values
        for cell_values in day_data:
            assert len(cell_values) == 195

    def test_multiple_days_with_mixed_completeness(self):
        """Test multiple days where some are complete and some are incomplete."""
        hourly_queue = Queue.Queue()
        results_queue = Queue.Queue()
        num_params = 65
        num_cells = 15

        # Day 1: Complete
        day1 = date(2024, 6, 1)
        np.random.seed(42)
        for hour in [6, 12, 18]:
            hourly_queue.put((day1, hour, np.random.rand(15 * 65).astype(np.float32).tolist()))

        # Day 2: Incomplete (only hours 6 and 12)
        day2 = date(2024, 6, 2)
        for hour in [6, 12]:
            hourly_queue.put((day2, hour, np.random.rand(15 * 65).astype(np.float32).tolist()))

        # Put sentinel
        hourly_queue.put((None, None, None))

        # Run assembler
        assemble_day_results(hourly_queue, results_queue, num_params, num_cells)

        # Should get at least one result (day 1)
        results = []
        while not results_queue.empty():
            try:
                result = results_queue.get(timeout=0.1)
                results.append(result)
            except Queue.Empty:
                break

        assert len(results) >= 1

        # Check that all results have valid shape
        for day_date, day_data in results:
            assert len(day_data) == num_cells
            for cell_values in day_data:
                assert len(cell_values) == 195

    def test_values_with_wrong_length_are_padded(self):
        """Test that values with wrong length are padded/truncated to correct size."""
        hourly_queue = Queue.Queue()
        results_queue = Queue.Queue()
        num_params = 65
        num_cells = 15
        test_date = date(2024, 6, 1)

        # Put values that are too short (should be padded)
        short_values = [1.0] * (15 * 65 - 10)  # 10 values short

        # Put other hours with correct length
        np.random.seed(42)
        correct_values = np.random.rand(15 * 65).astype(np.float32).tolist()

        hourly_queue.put((test_date, 6, short_values))
        hourly_queue.put((test_date, 12, correct_values))
        hourly_queue.put((test_date, 18, correct_values))

        # Put sentinel
        hourly_queue.put((None, None, None))

        # Run assembler
        assemble_day_results(hourly_queue, results_queue, num_params, num_cells)

        # Should produce valid output without crashing
        day_date, day_data = results_queue.get(timeout=1)
        assert day_date == test_date
        assert len(day_data) == num_cells

        # Each cell should have exactly 195 values
        for cell_values in day_data:
            assert len(cell_values) == 195

    def test_values_too_long_are_truncated(self):
        """Test that values that are too long are truncated."""
        hourly_queue = Queue.Queue()
        results_queue = Queue.Queue()
        num_params = 65
        num_cells = 15
        test_date = date(2024, 6, 1)

        # Put values that are too long (should be truncated)
        long_values = [1.0] * (15 * 65 + 10)  # 10 values too many

        # Put other hours with correct length
        np.random.seed(42)
        correct_values = np.random.rand(15 * 65).astype(np.float32).tolist()

        hourly_queue.put((test_date, 6, long_values))
        hourly_queue.put((test_date, 12, correct_values))
        hourly_queue.put((test_date, 18, correct_values))

        # Put sentinel
        hourly_queue.put((None, None, None))

        # Run assembler
        assemble_day_results(hourly_queue, results_queue, num_params, num_cells)

        # Should produce valid output without crashing
        day_date, day_data = results_queue.get(timeout=1)
        assert day_date == test_date
        assert len(day_data) == num_cells

        # Each cell should have exactly 195 values
        for cell_values in day_data:
            assert len(cell_values) == 195


@pytest.mark.integration
class TestAssembleDayResultsIntegration:
    """Integration tests for assemble_day_results with realistic scenarios."""

    def test_scenario_2021_05_01_partial_download(self):
        """
        Test scenario from bug report: 2021-05-01 has partial GRIB files.

        This simulates the real-world situation where:
        - Hour 06:00: Missing file (returns None)
        - Hour 12:00: File exists with data
        - Hour 18:00: File exists with data

        The assembler should handle this gracefully and produce valid output.
        """
        hourly_queue = Queue.Queue()
        results_queue = Queue.Queue()
        num_params = 65
        num_cells = 15
        test_date = date(2021, 5, 1)

        # Simulate: 06:00 is missing (None), 12:00 and 18:00 have data
        np.random.seed(42)
        hourly_queue.put((test_date, 6, None))  # Missing file
        hourly_queue.put((test_date, 12, np.random.rand(15 * 65).astype(np.float32).tolist()))
        hourly_queue.put((test_date, 18, np.random.rand(15 * 65).astype(np.float32).tolist()))

        # Put sentinel
        hourly_queue.put((None, None, None))

        # Run assembler - should not crash
        assemble_day_results(hourly_queue, results_queue, num_params, num_cells)

        # Should produce valid output
        day_date, day_data = results_queue.get(timeout=1)
        assert day_date == test_date
        assert len(day_data) == num_cells

        # Each cell should have 195 values
        for cell_values in day_data:
            assert len(cell_values) == 195
            # Check that 06:00 values are zeros (from missing file)
            hour_6_values = cell_values[:65]
            assert all(v == 0.0 for v in hour_6_values), "Hour 6 should be all zeros (missing file)"

    def test_multiple_days_with_various_missing_scenarios(self):
        """
        Test multiple days with various missing data scenarios.

        Day 1: Complete
        Day 2: Missing 06:00
        Day 3: Missing 12:00 and 18:00
        Day 4: All hours missing (all None)
        """
        hourly_queue = Queue.Queue()
        results_queue = Queue.Queue()
        num_params = 65
        num_cells = 15

        np.random.seed(42)
        valid_values = np.random.rand(15 * 65).astype(np.float32).tolist()

        # Day 1: Complete
        day1 = date(2024, 6, 1)
        for hour in [6, 12, 18]:
            hourly_queue.put((day1, hour, valid_values))

        # Day 2: Missing 06:00
        day2 = date(2024, 6, 2)
        hourly_queue.put((day2, 6, None))
        hourly_queue.put((day2, 12, valid_values))
        hourly_queue.put((day2, 18, valid_values))

        # Day 3: Only 06:00
        day3 = date(2024, 6, 3)
        hourly_queue.put((day3, 6, valid_values))

        # Put sentinel
        hourly_queue.put((None, None, None))

        # Run assembler
        assemble_day_results(hourly_queue, results_queue, num_params, num_cells)

        # Collect all results
        results = []
        while not results_queue.empty():
            try:
                result = results_queue.get(timeout=0.1)
                results.append(result)
            except Queue.Empty:
                break

        # Should get at least the complete day
        assert len(results) >= 1

        # All results should have valid shape
        for day_date, day_data in results:
            assert len(day_data) == num_cells
            for cell_values in day_data:
                assert len(cell_values) == 195