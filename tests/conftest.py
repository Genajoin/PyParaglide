"""
Pytest fixtures for PyParaglide tests.
"""

from datetime import date
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf


@pytest.fixture
def sample_data_dir(tmp_path: Path) -> Path:
    """Create a minimal sample dataset directory for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    nb_days = 10
    nb_cells = 2

    # Create minimal PKL files
    import pickle

    # meteo_days.pkl
    meteo_days = [date(2024, 8, 1 + i) for i in range(nb_days)]
    with open(data_dir / "meteo_days.pkl", "wb") as f:
        pickle.dump(meteo_days, f)

    # sorted_cells.pkl
    sorted_cells = [(45.5, 13.5), (45.6, 13.6)]
    with open(data_dir / "sorted_cells.pkl", "wb") as f:
        pickle.dump(sorted_cells, f)

    # meteo_params.pkl - minimal structure
    meteo_params = [
        (6, "Temperature", [[("isobaricInhPa", 850)]]),
        (12, "Temperature", [[("isobaricInhPa", 850)]]),
        (18, "Temperature", [[("isobaricInhPa", 850)]]),
    ]
    with open(data_dir / "meteo_params.pkl", "wb") as f:
        pickle.dump(meteo_params, f)

    # meteo_content_by_cell_day.pkl
    meteo_content = np.random.randn(nb_cells * nb_days, len(meteo_params)).astype(np.float32)
    with open(data_dir / "meteo_content_by_cell_day.pkl", "wb") as f:
        pickle.dump(meteo_content, f)

    # flights_by_cell_day.pkl
    flights = np.empty((nb_cells * nb_days,), dtype=object)
    for i in range(len(flights)):
        flights[i] = []
    with open(data_dir / "flights_by_cell_day.pkl", "wb") as f:
        pickle.dump(flights, f)

    # mountainess_by_cell_alt.pkl
    mountainess = np.zeros((nb_cells, 5), dtype=np.float32)
    with open(data_dir / "mountainess_by_cell_alt.pkl", "wb") as f:
        pickle.dump(mountainess, f)

    # spots.pkl
    spots = []
    with open(data_dir / "spots.pkl", "wb") as f:
        pickle.dump(spots, f)

    # sorted_cells_latlon.pkl
    with open(data_dir / "sorted_cells_latlon.pkl", "wb") as f:
        pickle.dump(sorted_cells, f)

    return data_dir


@pytest.fixture
def sample_grib_file(tmp_path: Path) -> Path:
    """Create a minimal sample GRIB file for testing."""
    # Note: Creating actual GRIB files requires pygrib/eccodes
    # For now, we just return a path that tests can skip if file doesn't exist
    grib_path = tmp_path / "sample.grib2"
    return grib_path


@pytest.fixture
def sample_model_config():
    """Sample model configuration for testing."""
    return {
        "problem_formulation": "CLASSIFICATION",
        "nb_cells": 2,
        "wind_dim": 8,
        "other_dim": 45,
        "humidity_dim": 2,
        "nb_altitudes": 5,
        "super_resolution": 1,
    }


@pytest.fixture
def reset_tf_session():
    """Reset TensorFlow session between tests."""
    yield
    tf.keras.backend.clear_session()
