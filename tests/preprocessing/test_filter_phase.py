"""Unit tests for FilterCellsPhase."""

import pickle
from pathlib import Path

import numpy as np
import pytest

from pyparaglide.preprocessing.phases.filter_phase import FilterCellsPhase


@pytest.fixture
def mock_pkl_data(tmp_path):
    """Create mock PKL files for testing."""
    nb_cells = 10
    nb_days = 5

    # Cell flight counts: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    cell_flight_counts = [i * 5 for i in range(nb_cells)]

    # Create cell_statistics.pkl
    stats_dict = {
        'total_flights_per_cell': cell_flight_counts,
        'nb_cells_original': nb_cells,
        'nb_days': nb_days,
    }
    with open(tmp_path / "cell_statistics.pkl", 'wb') as f:
        pickle.dump(stats_dict, f)

    # Create sorted_cells_latlon.pkl
    cells_latlon = [(45.0 + i, 13.0 + i) for i in range(nb_cells)]
    with open(tmp_path / "sorted_cells_latlon.pkl", 'wb') as f:
        pickle.dump(cells_latlon, f)

    # Create sorted_cells.pkl
    cells_grib = [(i, i * 2) for i in range(nb_cells)]
    with open(tmp_path / "sorted_cells.pkl", 'wb') as f:
        pickle.dump(cells_grib, f)

    # Create meteo_content_by_cell_day.pkl
    # Shape: (nb_days * nb_cells, 207)
    meteo_content = np.random.rand(nb_days * nb_cells, 207).astype(np.float32)
    with open(tmp_path / "meteo_content_by_cell_day.pkl", 'wb') as f:
        pickle.dump(meteo_content, f)

    # Create flights_by_cell_day.pkl
    # Shape: (nb_days * nb_cells,) dtype=object
    flights_by_cell_day = np.zeros((nb_days * nb_cells,), dtype=object)
    for i in range(len(flights_by_cell_day)):
        cell_id = i % nb_cells
        # Add flights based on cell_flight_counts
        num_flights = cell_flight_counts[cell_id] // nb_days
        flights_by_cell_day[i] = [
            (f"2024-06-{j:02d}T12:00:00", (100.0, 45.0 + j, 13.0 + j))
            for j in range(num_flights)
        ]
    with open(tmp_path / "flights_by_cell_day.pkl", 'wb') as f:
        pickle.dump(flights_by_cell_day, f)

    # Create mountainess_by_cell_alt.pkl
    # Shape: (nb_cells, 5)
    mountainess = np.random.rand(nb_cells, 5).astype(np.float32)
    with open(tmp_path / "mountainess_by_cell_alt.pkl", 'wb') as f:
        pickle.dump(mountainess, f)

    return {
        'nb_cells': nb_cells,
        'nb_days': nb_days,
        'cell_flight_counts': cell_flight_counts,
    }


def test_filter_phase_basic(tmp_path, mock_pkl_data):
    """Test basic cell filtering."""
    nb_cells = mock_pkl_data['nb_cells']
    nb_days = mock_pkl_data['nb_days']

    # Filter with threshold 20
    # Cell flight counts: [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    # Expected: 6 cells kept (indices 4,5,6,7,8,9)
    phase = FilterCellsPhase(out_dir=tmp_path, min_flights_per_cell=20)
    result = phase.execute()

    assert result['nb_cells_before'] == nb_cells
    assert result['nb_cells_after'] == 6
    assert result['cells_filtered_count'] == 4
    assert result['cells_kept'] == [4, 5, 6, 7, 8, 9]

    # Verify PKL files were updated
    with open(tmp_path / "sorted_cells_latlon.pkl", 'rb') as f:
        cells_latlon_new = pickle.load(f)
    assert len(cells_latlon_new) == 6
    assert cells_latlon_new[0] == (49.0, 17.0)  # Cell 4

    # Verify meteo shape
    with open(tmp_path / "meteo_content_by_cell_day.pkl", 'rb') as f:
        meteo_new = pickle.load(f)
    assert meteo_new.shape == (nb_days * 6, 207)

    # Verify flights shape
    with open(tmp_path / "flights_by_cell_day.pkl", 'rb') as f:
        flights_new = pickle.load(f)
    assert flights_new.shape == (nb_days * 6,)

    # Verify mountainess shape
    with open(tmp_path / "mountainess_by_cell_alt.pkl", 'rb') as f:
        mountainess_new = pickle.load(f)
    assert mountainess_new.shape == (6, 5)


def test_filter_phase_all_filtered(tmp_path, mock_pkl_data):
    """Test edge case: all cells filtered."""
    nb_cells = mock_pkl_data['nb_cells']

    # Filter with very high threshold (all cells will be filtered)
    phase = FilterCellsPhase(out_dir=tmp_path, min_flights_per_cell=1000)
    result = phase.execute()

    # Should keep all cells with warning
    assert result['nb_cells_before'] == nb_cells
    assert result['nb_cells_after'] == nb_cells
    assert result['cells_filtered_count'] == 0  # None filtered due to edge case handling
    assert result['cells_kept'] == list(range(nb_cells))


def test_filter_phase_zero_threshold(tmp_path, mock_pkl_data):
    """Test filtering disabled (threshold=0)."""
    nb_cells = mock_pkl_data['nb_cells']

    # Filter with threshold 0 should keep all cells
    phase = FilterCellsPhase(out_dir=tmp_path, min_flights_per_cell=0)
    result = phase.execute()

    assert result['nb_cells_before'] == nb_cells
    assert result['nb_cells_after'] == nb_cells
    assert result['cells_filtered_count'] == 0
    assert result['cells_kept'] == list(range(nb_cells))


def test_filter_phase_missing_statistics(tmp_path):
    """Test error when cell_statistics.pkl is missing."""
    phase = FilterCellsPhase(out_dir=tmp_path, min_flights_per_cell=20)

    with pytest.raises(FileNotFoundError):
        phase.execute()


def test_filter_phase_negative_threshold(tmp_path, mock_pkl_data):
    """Test error with negative threshold."""
    phase = FilterCellsPhase(out_dir=tmp_path, min_flights_per_cell=-10)

    with pytest.raises(ValueError, match="min_flights_per_cell must be >= 0"):
        phase.execute()


def test_index_mapping_correctness(tmp_path, mock_pkl_data):
    """Test that reindexing preserves data integrity."""
    nb_days = mock_pkl_data['nb_days']

    # Read original meteo data before filtering
    with open(tmp_path / "meteo_content_by_cell_day.pkl", 'rb') as f:
        meteo_old = pickle.load(f)

    # Filter
    phase = FilterCellsPhase(out_dir=tmp_path, min_flights_per_cell=20)
    result = phase.execute()

    # Read filtered meteo data
    with open(tmp_path / "meteo_content_by_cell_day.pkl", 'rb') as f:
        meteo_new = pickle.load(f)

    # Verify that data for kept cells matches
    # Cell 4 in old data should be cell 0 in new data
    cells_kept = result['cells_kept']

    # Check first day, first kept cell (cell 4 → cell 0)
    old_idx = 0 * 10 + cells_kept[0]  # day 0, cell 4
    new_idx = 0 * 6 + 0  # day 0, cell 0 in filtered data

    np.testing.assert_array_almost_equal(
        meteo_old[old_idx],
        meteo_new[new_idx],
        decimal=5
    )
