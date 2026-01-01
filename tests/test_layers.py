"""
Tests for custom Keras layers.
"""

import numpy as np
import pytest
import tensorflow as tf

from pyparaglide.models.enums import ProblemFormulation
from pyparaglide.models.layers import (
    CrossabilityBlock,
    FlyabilityBlock,
    HumidityFlyabilityBlock,
    PopulationBlock,
    WindBlockCells,
    WindBlockSpots,
    WindFlyabilityBlock,
)


@pytest.mark.usefixtures("reset_tf_session")
class TestWindBlockCells:
    """Test WindBlockCells layer."""

    def test_build_and_call(self):
        """Test building and calling WindBlockCells."""
        layer = WindBlockCells()
        nb_cells = 2
        nb_altitudes = 5
        batch_size = 4

        # Create inputs
        mountainess = tf.constant(np.random.randn(batch_size, nb_cells, nb_altitudes), dtype=tf.float32)
        wind = tf.constant(np.random.randn(batch_size, nb_cells, nb_altitudes, 3, 8), dtype=tf.float32)

        # Build layer
        layer.build([mountainess.shape, wind.shape])

        # Check weights
        assert hasattr(layer, "mountainess_factor")
        assert layer.mountainess_factor.shape == (1,)

        # Call layer
        output = layer([mountainess, wind])

        # Check output shape
        assert output.shape == (batch_size, nb_cells, nb_altitudes, 3)

    def test_mountainess_factor_effect(self):
        """Test that mountainess_factor affects output."""
        layer = WindBlockCells()
        nb_cells = 1
        nb_altitudes = 1
        batch_size = 1

        # Create inputs with correct shapes
        # mountainess: (batch, nb_cells, nb_altitudes)
        mountainess = tf.constant([[[1.0]]], dtype=tf.float32)  # (1, 1, 1)
        # wind: (batch, nb_cells, nb_altitudes, nb_hours, nb_wind_dims)
        wind = tf.constant([[[[[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]]]], dtype=tf.float32)  # (1, 1, 1, 1, 8)

        layer.build([mountainess.shape, wind.shape])

        # Set mountainess_factor to 1.0
        layer.mountainess_factor.assign([1.0])

        output1 = layer([mountainess, wind])

        # Set mountainess_factor to 0.0
        layer.mountainess_factor.assign([0.0])

        output2 = layer([mountainess, wind])

        # Output should be different
        assert not np.allclose(output1.numpy(), output2.numpy())


@pytest.mark.usefixtures("reset_tf_session")
class TestWindFlyabilityBlock:
    """Test WindFlyabilityBlock layer."""

    def test_build_and_call(self):
        """Test building and calling WindFlyabilityBlock."""
        nb_altitudes = 5
        nb_cells = 2
        humidity_dim = 2

        layer = WindFlyabilityBlock(nb_altitudes, nb_cells, humidity_dim)
        batch_size = 4

        # Create input: (batch, nb_cells, nb_altitudes, 3)
        x = tf.constant(np.random.randn(batch_size, nb_cells, nb_altitudes, 3), dtype=tf.float32)

        # Call layer
        output = layer(x)

        # Check output shape: (batch, nb_cells, nb_altitudes)
        assert output.shape == (batch_size, nb_cells, nb_altitudes)

    def test_output_range(self):
        """Test that output is in [0, 1] range (sigmoid activation)."""
        nb_altitudes = 5
        nb_cells = 1
        humidity_dim = 2

        layer = WindFlyabilityBlock(nb_altitudes, nb_cells, humidity_dim)
        batch_size = 10

        # Create input with various values
        x = tf.constant(np.random.randn(batch_size, nb_cells, nb_altitudes, 3) * 10, dtype=tf.float32)

        output = layer(x)

        # All values should be in [0, 1]
        assert np.all(output.numpy() >= 0.0)
        assert np.all(output.numpy() <= 1.0)


@pytest.mark.usefixtures("reset_tf_session")
class TestHumidityFlyabilityBlock:
    """Test HumidityFlyabilityBlock layer."""

    def test_build_and_call(self):
        """Test building and calling HumidityFlyabilityBlock."""
        nb_altitudes = 5
        nb_cells = 2
        humidity_dim = 2

        layer = HumidityFlyabilityBlock(nb_altitudes, nb_cells, humidity_dim)
        batch_size = 4

        # Create input: (batch, nb_cells, 3, humidity_dim)
        x = tf.constant(np.random.randn(batch_size, nb_cells, 3, humidity_dim), dtype=tf.float32)

        # Call layer
        output = layer(x)

        # Check output shape: (batch, nb_cells, nb_altitudes)
        assert output.shape == (batch_size, nb_cells, nb_altitudes)

    def test_output_range(self):
        """Test that output is in [0, 1] range (sigmoid activation)."""
        nb_altitudes = 5
        nb_cells = 1
        humidity_dim = 2

        layer = HumidityFlyabilityBlock(nb_altitudes, nb_cells, humidity_dim)
        batch_size = 10

        x = tf.constant(np.random.randn(batch_size, nb_cells, 3, humidity_dim) * 10, dtype=tf.float32)

        output = layer(x)

        # All values should be in [0, 1]
        assert np.all(output.numpy() >= 0.0)
        assert np.all(output.numpy() <= 1.0)


@pytest.mark.usefixtures("reset_tf_session")
class TestFlyabilityBlock:
    """Test FlyabilityBlock layer."""

    def test_build_and_call(self):
        """Test building and calling FlyabilityBlock."""
        other_dim = 45
        humidity_dim = 2

        layer = FlyabilityBlock(other_dim, humidity_dim)
        batch_size = 4

        # Create inputs
        wind = tf.constant(np.random.randn(batch_size, 3), dtype=tf.float32)
        other = tf.constant(np.random.randn(batch_size, 3 * other_dim), dtype=tf.float32)
        rain = tf.constant(np.random.randn(batch_size, 3 * humidity_dim), dtype=tf.float32)

        # Call layer
        output = layer([wind, other, rain])

        # Check output shape: (batch, 1)
        assert output.shape == (batch_size, 1)

    def test_output_range(self):
        """Test that output is in [0, 1] range (sigmoid activation)."""
        other_dim = 10
        humidity_dim = 2

        layer = FlyabilityBlock(other_dim, humidity_dim)
        batch_size = 10

        wind = tf.constant(np.random.randn(batch_size, 3) * 10, dtype=tf.float32)
        other = tf.constant(np.random.randn(batch_size, 3 * other_dim) * 10, dtype=tf.float32)
        rain = tf.constant(np.random.randn(batch_size, 3 * humidity_dim) * 10, dtype=tf.float32)

        output = layer([wind, other, rain])

        # All values should be in [0, 1]
        assert np.all(output.numpy() >= 0.0)
        assert np.all(output.numpy() <= 1.0)


@pytest.mark.usefixtures("reset_tf_session")
class TestPopulationBlock:
    """Test PopulationBlock layer."""

    def test_build_and_call_classification(self):
        """Test PopulationBlock with CLASSIFICATION formulation."""
        nb_cells = 2
        nb_altitudes = 5
        super_resolution = 1

        var_date_factor = tf.Variable(np.array([[1.275]], dtype=np.float32))
        var_dow_factor = tf.constant(np.array([[1.0] * 7], dtype=np.float32))

        layer = PopulationBlock(
            ProblemFormulation.CLASSIFICATION,
            var_date_factor,
            var_dow_factor,
            super_resolution,
        )

        batch_size = 4

        # Create inputs
        prediction = tf.constant(np.random.rand(batch_size, nb_cells, nb_altitudes) * 0.5, dtype=tf.float32)
        date = tf.constant(np.random.rand(batch_size, 1), dtype=tf.float32)
        dow = tf.constant(np.random.rand(batch_size, 7), dtype=tf.float32)

        # Build layer
        layer.build([prediction.shape, date.shape, dow.shape])

        # Call layer
        output = layer([prediction, date, dow])

        # Check output shape
        expected_cells = nb_cells * super_resolution * super_resolution
        assert output.shape == (batch_size, expected_cells, nb_altitudes)

    def test_build_and_call_regression(self):
        """Test PopulationBlock with REGRESSION formulation."""
        nb_cells = 1
        nb_altitudes = 5
        super_resolution = 1

        var_date_factor = tf.Variable(np.array([[1.275]], dtype=np.float32))
        var_dow_factor = tf.constant(np.array([[1.0] * 7], dtype=np.float32))

        layer = PopulationBlock(
            ProblemFormulation.REGRESSION,
            var_date_factor,
            var_dow_factor,
            super_resolution,
        )

        batch_size = 4

        prediction = tf.constant(np.random.rand(batch_size, nb_cells, nb_altitudes), dtype=tf.float32)
        date = tf.constant(np.random.rand(batch_size, 1), dtype=tf.float32)
        dow = tf.constant(np.random.rand(batch_size, 7), dtype=np.float32)

        layer.build([prediction.shape, date.shape, dow.shape])
        output = layer([prediction, date, dow])

        assert output.shape == (batch_size, nb_cells, nb_altitudes)

    def test_popu_weights_non_negative(self):
        """Test that popu weights are constrained to be non-negative."""
        nb_cells = 1
        nb_altitudes = 3
        super_resolution = 1

        var_date_factor = tf.Variable(np.array([[1.275]], dtype=np.float32))
        var_dow_factor = tf.constant(np.array([[1.0] * 7], dtype=np.float32))

        layer = PopulationBlock(
            ProblemFormulation.CLASSIFICATION,
            var_date_factor,
            var_dow_factor,
            super_resolution,
        )

        prediction = tf.constant(np.random.rand(2, nb_cells, nb_altitudes), dtype=tf.float32)
        date = tf.constant(np.random.rand(2, 1), dtype=np.float32)
        dow = tf.constant(np.random.rand(2, 7), dtype=np.float32)

        layer.build([prediction.shape, date.shape, dow.shape])

        # Check that popu has NonNeg constraint
        assert layer.popu.constraint is not None
        # Check that all weights are non-negative
        assert np.all(layer.popu.numpy() >= 0.0)


@pytest.mark.usefixtures("reset_tf_session")
class TestWindBlockSpots:
    """Test WindBlockSpots layer."""

    def test_build_and_call(self):
        """Test building and calling WindBlockSpots."""
        nb_spots = 3

        layer = WindBlockSpots(nb_spots)
        batch_size = 4
        nb_altitudes = 5

        # Create input: (batch, nb_altitudes, 3, wind_dim)
        x = tf.constant(np.random.randn(batch_size, nb_altitudes, 3, 8), dtype=tf.float32)

        # Build layer
        layer.build(x.shape)

        # Check weights
        assert hasattr(layer, "windWeights")
        assert layer.windWeights.shape == (nb_spots, 8)
        assert hasattr(layer, "alt")
        assert layer.alt.shape == (nb_spots, 1)

        # Call layer
        output = layer(x)

        # Check output shape: (batch, nb_spots, 3)
        assert output.shape == (batch_size, nb_spots, 3)


@pytest.mark.usefixtures("reset_tf_session")
class TestCrossabilityBlock:
    """Test CrossabilityBlock layer."""

    def test_build_and_call(self):
        """Test building and calling CrossabilityBlock."""
        nb_altitudes = 5
        nb_cells = 2
        other_dim = 45
        humidity_dim = 2

        layer = CrossabilityBlock(other_dim, humidity_dim, nb_altitudes, nb_cells)
        batch_size = 4

        # Create inputs
        flyability = tf.constant(np.random.rand(batch_size, nb_cells, nb_altitudes), dtype=tf.float32)
        fufu_wind = tf.constant(np.random.rand(batch_size, nb_cells, nb_altitudes, 3), dtype=tf.float32)
        fufu_other = tf.constant(np.random.rand(batch_size, nb_cells, 3, other_dim), dtype=tf.float32)
        fufu_rain = tf.constant(np.random.rand(batch_size, nb_cells, 3, humidity_dim), dtype=tf.float32)

        # Call layer
        output = layer([flyability, fufu_wind, fufu_other, fufu_rain])

        # Check output shape: (batch, nb_cells, nb_altitudes)
        assert output.shape == (batch_size, nb_cells, nb_altitudes)

    def test_output_range(self):
        """Test that output is in [0, 1] range (sigmoid activation)."""
        nb_altitudes = 5
        nb_cells = 1
        other_dim = 10
        humidity_dim = 2

        layer = CrossabilityBlock(other_dim, humidity_dim, nb_altitudes, nb_cells)
        batch_size = 10

        flyability = tf.constant(np.random.rand(batch_size, nb_cells, nb_altitudes), dtype=tf.float32)
        fufu_wind = tf.constant(np.random.rand(batch_size, nb_cells, nb_altitudes, 3), dtype=tf.float32)
        fufu_other = tf.constant(np.random.rand(batch_size, nb_cells, 3, other_dim), dtype=tf.float32)
        fufu_rain = tf.constant(np.random.rand(batch_size, nb_cells, 3, humidity_dim), dtype=tf.float32)

        output = layer([flyability, fufu_wind, fufu_other, fufu_rain])

        # All values should be in [0, 1]
        assert np.all(output.numpy() >= 0.0)
        assert np.all(output.numpy() <= 1.0)
