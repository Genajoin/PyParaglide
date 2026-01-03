"""
Tests for ModelCells architecture.
"""

import numpy as np
import pytest
import tensorflow as tf

from pyparaglide.models.enums import ModelType, ProblemFormulation
from pyparaglide.models.model_cells import ModelCells


@pytest.mark.usefixtures("reset_tf_session")
class TestModelCells:
    """Test ModelCells model creation and execution."""

    def test_output_names(self):
        """Test output names list (after altitude binning removal)."""
        names = ModelCells.output_names()

        # After altitude binning removal: 4 outputs instead of 20
        assert len(names) == 4
        assert "flown" in names
        assert "crossed" in names
        assert "wind_flown" in names
        assert "humidity_flown" in names

    def test_create_model_classification(self):
        """Test creating CELLS model for CLASSIFICATION."""
        model = ModelCells.create_model(
            problem_formulation=ProblemFormulation.CLASSIFICATION,
            nb_cells=2,
            wind_dim=8,
            other_dim=45,
            humidity_dim=2,
            nb_altitudes=1,
            super_resolution=1,
        )

        # Check that model is created
        assert model is not None
        assert isinstance(model, tf.keras.Model)

        # Check number of inputs
        assert len(model.inputs) == 6

        # Check input names
        input_names = [inp.name.split(":")[0] for inp in model.inputs]
        assert "in_date" in input_names
        assert "in_dow" in input_names
        assert "in_mountainess" in input_names
        assert "in_other" in input_names
        assert "in_rain" in input_names
        assert "in_wind" in input_names

        # Check number of outputs (4 outputs: flown, crossed, wind_flown, humidity_flown)
        assert len(model.outputs) == 4

    def test_create_model_regression(self):
        """Test creating CELLS model for REGRESSION."""
        model = ModelCells.create_model(
            problem_formulation=ProblemFormulation.REGRESSION,
            nb_cells=1,
            wind_dim=8,
            other_dim=45,
            humidity_dim=2,
            nb_altitudes=1,
            super_resolution=1,
        )

        assert model is not None
        assert isinstance(model, tf.keras.Model)
        assert len(model.outputs) == 4

    def test_model_forward_pass(self):
        """Test forward pass through the model."""
        model = ModelCells.create_model(
            problem_formulation=ProblemFormulation.CLASSIFICATION,
            nb_cells=1,
            wind_dim=8,
            other_dim=45,
            humidity_dim=2,
            nb_altitudes=1,
            super_resolution=1,
        )

        batch_size = 4

        # Create inputs
        inputs = {
            "in_date": tf.constant(np.random.rand(batch_size, 1), dtype=tf.float32),
            "in_dow": tf.constant(np.random.rand(batch_size, 7), dtype=tf.float32),
            "in_mountainess": tf.constant(np.random.randn(batch_size, 1, 1), dtype=tf.float32),
            "in_other": tf.constant(np.random.randn(batch_size, 1, 3, 45), dtype=tf.float32),
            "in_rain": tf.constant(np.random.randn(batch_size, 1, 3, 2), dtype=tf.float32),
            "in_wind": tf.constant(np.random.randn(batch_size, 1, 1, 3, 8), dtype=tf.float32),
        }

        # Run forward pass
        outputs = model(inputs)

        # Check outputs
        assert len(outputs) == 4

        # Each output should have shape (batch, nb_cells, 1) after altitude binning removal
        for output in outputs:
            assert output.shape == (batch_size, 1, 1)

    def test_output_range(self):
        """Test that all outputs are in [0, 1] range (sigmoid activation)."""
        model = ModelCells.create_model(
            problem_formulation=ProblemFormulation.CLASSIFICATION,
            nb_cells=1,
            wind_dim=8,
            other_dim=45,
            humidity_dim=2,
            nb_altitudes=1,
            super_resolution=1,
        )

        batch_size = 10

        inputs = {
            "in_date": tf.constant(np.random.rand(batch_size, 1), dtype=tf.float32),
            "in_dow": tf.constant(np.random.rand(batch_size, 7), dtype=tf.float32),
            "in_mountainess": tf.constant(np.random.randn(batch_size, 1, 1), dtype=tf.float32),
            "in_other": tf.constant(np.random.randn(batch_size, 1, 3, 45), dtype=tf.float32),
            "in_rain": tf.constant(np.random.randn(batch_size, 1, 3, 2), dtype=tf.float32),
            "in_wind": tf.constant(np.random.randn(batch_size, 1, 1, 3, 8), dtype=tf.float32),
        }

        outputs = model(inputs)

        # All outputs should be in [0, 1]
        for output in outputs:
            assert np.all(output.numpy() >= 0.0)
            assert np.all(output.numpy() <= 1.0)

    def test_super_resolution(self):
        """Test model with super_resolution > 1."""
        super_resolution = 2

        model = ModelCells.create_model(
            problem_formulation=ProblemFormulation.CLASSIFICATION,
            nb_cells=1,
            wind_dim=8,
            other_dim=45,
            humidity_dim=2,
            nb_altitudes=1,
            super_resolution=super_resolution,
        )

        batch_size = 4

        inputs = {
            "in_date": tf.constant(np.random.rand(batch_size, 1), dtype=tf.float32),
            "in_dow": tf.constant(np.random.rand(batch_size, 7), dtype=tf.float32),
            "in_mountainess": tf.constant(np.random.randn(batch_size, 1, 1), dtype=tf.float32),
            "in_other": tf.constant(np.random.randn(batch_size, 1, 3, 45), dtype=tf.float32),
            "in_rain": tf.constant(np.random.randn(batch_size, 1, 3, 2), dtype=tf.float32),
            "in_wind": tf.constant(np.random.randn(batch_size, 1, 1, 3, 8), dtype=tf.float32),
        }

        outputs = model(inputs)

        # Output should have shape (batch, nb_cells * super_resolution^2, 1) after altitude binning removal
        expected_cells = 1 * super_resolution * super_resolution
        for output in outputs:
            assert output.shape == (batch_size, expected_cells, 1)

    def test_trainable_variables(self):
        """Test that model has trainable variables."""
        model = ModelCells.create_model(
            problem_formulation=ProblemFormulation.CLASSIFICATION,
            nb_cells=1,
            wind_dim=8,
            other_dim=45,
            humidity_dim=2,
            nb_altitudes=1,
            super_resolution=1,
        )

        # Should have trainable variables
        assert len(model.trainable_variables) > 0

        # Check for specific variables (names may have suffixes)
        var_names = [v.name for v in model.trainable_variables]
        # WindBlock has mountainess_factor
        assert any("mountainess" in n.lower() for n in var_names)
        # FlyabilityBlock or PopulationBlock has dense/kernel layers
        assert any("flyability" in n.lower() or "population" in n.lower() or "kernel" in n.lower() for n in var_names)

    def test_compile_model(self):
        """Test compiling the model."""
        model = ModelCells.create_model(
            problem_formulation=ProblemFormulation.CLASSIFICATION,
            nb_cells=1,
            wind_dim=8,
            other_dim=45,
            humidity_dim=2,
            nb_altitudes=1,
            super_resolution=1,
        )

        # Compile with binary_crossentropy (CLASSIFICATION)
        model.compile(optimizer="adam", loss="binary_crossentropy")

        # Check optimizer
        assert model.optimizer is not None

    def test_model_summary(self, capsys):
        """Test that model summary can be printed."""
        model = ModelCells.create_model(
            problem_formulation=ProblemFormulation.CLASSIFICATION,
            nb_cells=1,
            wind_dim=8,
            other_dim=45,
            humidity_dim=2,
            nb_altitudes=1,
            super_resolution=1,
        )

        # Print summary (should not raise)
        model.summary()

        # Check that output was captured
        captured = capsys.readouterr()
        assert "Model" in captured.out or "Functional" in captured.out


@pytest.mark.usefixtures("reset_tf_session")
class TestModelCellsEncapsulateFlyability:
    """Test _encapsulate_flyability static method."""

    def test_encapsulate_flyability(self):
        """Test encapsulate_flyability reshaping (after altitude binning removal)."""
        nb_cells = 2
        nb_altitudes = 1  # Changed from 5
        other_dim = 45
        humidity_dim = 2

        # Create simple flyability block that matches FlyabilityBlock interface
        # It expects a list of 3 inputs: wind, other, rain
        input_wind = tf.keras.layers.Input(shape=(3,), name="test_wind")
        input_other = tf.keras.layers.Input(shape=(3 * other_dim,), name="test_other")
        input_rain = tf.keras.layers.Input(shape=(3 * humidity_dim,), name="test_rain")
        merged = tf.keras.layers.Concatenate()([input_wind, input_other, input_rain])
        dense1 = tf.keras.layers.Dense(16, activation="tanh")(merged)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(dense1)
        flyability_block = tf.keras.Model([input_wind, input_other, input_rain], output)

        batch_size = 4

        # Create inputs
        wind = tf.constant(np.random.randn(batch_size, nb_cells, nb_altitudes, 3), dtype=tf.float32)
        other = tf.constant(np.random.randn(batch_size, nb_cells, 3, other_dim), dtype=tf.float32)
        rain = tf.constant(np.random.randn(batch_size, nb_cells, 3, humidity_dim), dtype=tf.float32)

        # Call encapsulate_flyability
        output = ModelCells._encapsulate_flyability(
            flyability_block,
            nb_cells,
            nb_altitudes,
            other_dim,
            humidity_dim,
            [wind, other, rain],
        )

        # Check output shape: (batch, nb_cells, 1) after altitude binning removal
        assert output.shape == (batch_size, nb_cells, nb_altitudes)
