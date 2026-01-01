"""
Tests for model enums.
"""

import pytest

from pyparaglide.models.enums import (
    ModelSettings,
    ModelType,
    ProblemFormulation,
)


class TestModelType:
    """Test ModelType enum."""

    def test_cells_value(self):
        """Test CELLS enum value."""
        assert ModelType.CELLS.value == 0
        assert ModelType.CELLS.name == "CELLS"

    def test_spots_value(self):
        """Test SPOTS enum value."""
        assert ModelType.SPOTS.value == 1
        assert ModelType.SPOTS.name == "SPOTS"

    def test_from_string(self):
        """Test creating ModelType from string."""
        assert ModelType["CELLS"] == ModelType.CELLS
        assert ModelType["SPOTS"] == ModelType.SPOTS


class TestProblemFormulation:
    """Test ProblemFormulation enum."""

    def test_classification_value(self):
        """Test CLASSIFICATION enum value."""
        assert ProblemFormulation.CLASSIFICATION.value == 0
        assert ProblemFormulation.CLASSIFICATION.name == "CLASSIFICATION"

    def test_regression_value(self):
        """Test REGRESSION enum value."""
        assert ProblemFormulation.REGRESSION.value == 1
        assert ProblemFormulation.REGRESSION.name == "REGRESSION"


class TestModelSettings:
    """Test ModelSettings class."""

    def test_optimize_dow_default(self):
        """Test default optimize_dow value."""
        assert ModelSettings.optimize_dow is False

    def test_dow_init_shape(self):
        """Test dow_init shape."""
        import numpy as np

        assert ModelSettings.dow_init.shape == (1, 7)
        # Check that values sum to approximately 7 (normalized)
        assert abs(np.sum(ModelSettings.dow_init) - 7.0) < 0.1

    def test_dow_init_values(self):
        """Test dow_init has reasonable values."""
        import numpy as np

        # All values should be positive
        assert np.all(ModelSettings.dow_init > 0)
        # Weekend values should be higher (based on init data)
        # Indices: Mon=0, Tue=1, Wed=2, Thu=3, Fri=4, Sat=5, Sun=6
        assert ModelSettings.dow_init[0, 5] > ModelSettings.dow_init[0, 0]  # Sat > Mon
        assert ModelSettings.dow_init[0, 6] > ModelSettings.dow_init[0, 1]  # Sun > Tue
