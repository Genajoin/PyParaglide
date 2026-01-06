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
        """Test output names list (after removing redundant indicators)."""
        names = ModelCells.output_names()

        # After removing wind_flown and humidity_flown: 2 outputs
        assert len(names) == 2
        assert "flown" in names
        assert "crossed" in names

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

        # Check number of inputs (7 inputs including in_thermo)
        assert len(model.inputs) == 7

        # Check input names
        input_names = [inp.name.split(":")[0] for inp in model.inputs]
        assert "in_date" in input_names
        assert "in_dow" in input_names
        assert "in_mountainess" in input_names
        assert "in_other" in input_names
        assert "in_rain" in input_names
        assert "in_wind" in input_names
        assert "in_thermo" in input_names

        # Check number of outputs (2 outputs: flown, crossed)
        assert len(model.outputs) == 2

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
        assert len(model.outputs) == 2

    def test_model_forward_pass(self):
        """Test forward pass through the model."""
        model = ModelCells.create_model(
            problem_formulation=ProblemFormulation.CLASSIFICATION,
            nb_cells=1,
            wind_dim=8,
            other_dim=45,
            humidity_dim=2,
            nb_altitudes=1,
            thermo_dim=0,  # baseline model
            super_resolution=1,
        )

        batch_size = 4

        # Create inputs (including in_thermo with empty last dimension)
        inputs = {
            "in_date": tf.constant(np.random.rand(batch_size, 1), dtype=tf.float32),
            "in_dow": tf.constant(np.random.rand(batch_size, 7), dtype=tf.float32),
            "in_mountainess": tf.constant(np.random.randn(batch_size, 1, 1), dtype=tf.float32),
            "in_other": tf.constant(np.random.randn(batch_size, 1, 3, 45), dtype=tf.float32),
            "in_rain": tf.constant(np.random.randn(batch_size, 1, 3, 2), dtype=tf.float32),
            "in_wind": tf.constant(np.random.randn(batch_size, 1, 1, 3, 8), dtype=tf.float32),
            "in_thermo": tf.constant(np.random.randn(batch_size, 1, 3, 0), dtype=tf.float32),  # empty thermo
        }

        # Run forward pass
        outputs = model(inputs)

        # Check outputs
        assert len(outputs) == 2

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
            thermo_dim=0,  # baseline model
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
            "in_thermo": tf.constant(np.random.randn(batch_size, 1, 3, 0), dtype=tf.float32),  # empty thermo
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
            thermo_dim=0,  # baseline model
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
            "in_thermo": tf.constant(np.random.randn(batch_size, 1, 3, 0), dtype=tf.float32),  # empty thermo
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
        thermo_dim = 0  # baseline model

        # Create simple flyability block that matches FlyabilityBlock interface
        # It expects a list of 4 inputs: wind, other, rain, thermo
        input_wind = tf.keras.layers.Input(shape=(3,), name="test_wind")
        input_other = tf.keras.layers.Input(shape=(3 * other_dim,), name="test_other")
        input_rain = tf.keras.layers.Input(shape=(3 * humidity_dim,), name="test_rain")
        input_thermo = tf.keras.layers.Input(shape=(3 * thermo_dim,), name="test_thermo")  # empty for baseline
        merged = tf.keras.layers.Concatenate()([input_wind, input_other, input_rain, input_thermo])
        dense1 = tf.keras.layers.Dense(16, activation="tanh")(merged)
        output = tf.keras.layers.Dense(1, activation="sigmoid")(dense1)
        flyability_block = tf.keras.Model([input_wind, input_other, input_rain, input_thermo], output)

        batch_size = 4

        # Create inputs
        wind = tf.constant(np.random.randn(batch_size, nb_cells, nb_altitudes, 3), dtype=tf.float32)
        other = tf.constant(np.random.randn(batch_size, nb_cells, 3, other_dim), dtype=tf.float32)
        rain = tf.constant(np.random.randn(batch_size, nb_cells, 3, humidity_dim), dtype=tf.float32)
        thermo = tf.constant(np.random.randn(batch_size, nb_cells, 3, thermo_dim), dtype=tf.float32)  # empty for baseline

        # Call encapsulate_flyability with 4 inputs (including thermo)
        output = ModelCells._encapsulate_flyability(
            flyability_block,
            nb_cells,
            nb_altitudes,
            other_dim,
            humidity_dim,
            [wind, other, rain, thermo],  # inputs list comes before input_dim_thermo
            thermo_dim,
        )

        # Check output shape: (batch, nb_cells, 1) after altitude binning removal
        assert output.shape == (batch_size, nb_cells, nb_altitudes)
