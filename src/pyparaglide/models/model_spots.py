"""
ModelSpots - Location-specific flyability prediction model.

Predicts flyability for individual take-off spots with learned
spot-specific wind behavior (wind direction weights, optimal altitude).

Each spot gets its own prediction based on:
- Learned wind direction preferences
- Learned relevant altitude
- Shared flyability block (frozen from CELLS training)
"""

from typing import Any

import numpy as np
import tensorflow as tf

from pyparaglide.models.enums import ModelSettings, ProblemFormulation
from pyparaglide.models.layers import FlyabilityBlock, PopulationBlock, WindBlockSpots


class ModelSpots:
    """
    Location-specific (SPOTS) model for paragliding flyability prediction.

    Each spot has learned parameters:
    - Wind direction weights (8 directions)
    - Relevant altitude (for wind interpolation)

    The flyability block is shared and frozen (trained from CELLS model).
    """

    @classmethod
    def output_names(cls) -> list[str]:
        """Return the names of all model outputs."""
        return ["flown"]

    @classmethod
    def create_model(
        cls,
        problem_formulation: ProblemFormulation,
        cells_data: dict[int, dict[str, Any]],
        wind_dim: int,
        other_dim: int,
        humidity_dim: int,
        nb_altitudes: int,
        initialization: dict[str, Any],
    ) -> tf.keras.Model:
        """
        Create a new SPOTS model.

        Args:
            problem_formulation: CLASSIFICATION or REGRESSION
            cells_data: Dict mapping cell_id -> {'spots': [spot_list]}
            wind_dim: Wind direction dimensions (typically 8)
            other_dim: Other weather data dimensions
            humidity_dim: Humidity/rain data dimensions
            nb_altitudes: Number of altitude levels (typically 5)
            initialization: Dict with 'date_factor' and 'dow_factor' (required for spots)

        Returns:
            Compiled Keras model with one output per spot

        Raises:
            AssertionError: If initialization is None or missing required keys
        """
        # Check initialization (required for spots model)
        assert initialization is not None, "initialization dict is required for SPOTS model"
        for k in initialization:
            assert k in ["date_factor", "dow_factor"], f"Unknown initialization key: {k}"

        # ==============================================================================
        # Shared variables (not trainable for spots)
        # ==============================================================================

        var_date_factor = tf.constant(
            value=initialization["date_factor"],
            name="var_date_factor",
        )

        if ModelSettings.optimize_dow:
            var_dow_factor = tf.constant(
                value=initialization["dow_factor"],
                name="var_dow_factor",
            )
        else:
            var_dow_factor = tf.constant(
                value=ModelSettings.dow_init,
                name="var_dow_factor",
            )

        # ==============================================================================
        # Inputs
        # ==============================================================================

        nb_cells = len(cells_data)

        input_date = tf.keras.layers.Input(shape=(1,), name="in_date")
        input_dow = tf.keras.layers.Input(shape=(7,), name="in_dow")
        input_mountainess = tf.keras.layers.Input(
            shape=(nb_cells, nb_altitudes), name="in_mountainess"
        )
        input_other = tf.keras.layers.Input(shape=(nb_cells, 3, other_dim), name="in_other")
        input_humidity = tf.keras.layers.Input(shape=(nb_cells, 3, humidity_dim), name="in_rain")
        input_wind = tf.keras.layers.Input(
            shape=(nb_cells, nb_altitudes, 3, wind_dim), name="in_wind"
        )

        all_inputs = [
            input_date,
            input_dow,
            input_mountainess,
            input_other,
            input_humidity,
            input_wind,
        ]

        # ==============================================================================
        # Shared flyability block (frozen)
        # ==============================================================================

        flyability_model = FlyabilityBlock(other_dim, humidity_dim, name="flyability_block")
        flyability_model.trainable = False  # Freeze the block (trained from CELLS)

        # ==============================================================================
        # Process each cell with its spots
        # ==============================================================================

        all_outputs = []  # One tensor per spot

        for cell_idx, (cell_id, cell_data) in enumerate(cells_data.items()):
            spots = cell_data.get("spots", [])
            nb_spots = len(spots)

            if nb_spots > 0:
                # Extract inputs for this cell
                input_wind_this_cell = tf.keras.layers.Lambda(lambda x: x[:, cell_idx, ...])(input_wind)
                input_other_this_cell = tf.keras.layers.Lambda(lambda x: x[:, cell_idx, ...])(input_other)
                input_humidity_this_cell = tf.keras.layers.Lambda(lambda x: x[:, cell_idx, ...])(input_humidity)

                # Create blocks for this cell
                population_block = PopulationBlock(
                    problem_formulation,
                    var_date_factor,
                    var_dow_factor,
                    super_resolution=1,
                    name=f"population__cell_{cell_id}",
                )
                wind_block = WindBlockSpots(nb_spots, name=f"wind_block_spots__cell_{cell_id}")

                # Wind prediction with spot-specific behavior
                wind_prediction = wind_block(input_wind_this_cell)
                wind_prediction = tf.keras.layers.Lambda(
                    lambda x: tf.reshape(x, (-1, 1, nb_spots, 3))
                )(wind_prediction)

                # Flyability using shared (frozen) block
                flyability_prediction = cls._encapsulate_flyability(
                    flyability_model,
                    nb_cells=1,
                    nb_altitudes_or_nb_spots=nb_spots,
                    input_dim_other=other_dim,
                    input_dim_rain=humidity_dim,
                    inputs=[wind_prediction, input_other_this_cell, input_humidity_this_cell],
                )

                # Apply population factor
                flown_prediction = population_block([flyability_prediction, input_date, input_dow])
                flown_prediction = tf.keras.layers.Lambda(
                    lambda x: tf.reshape(x, (-1, nb_spots))
                )(flown_prediction)

                all_outputs.append(flown_prediction)

        # ==============================================================================
        # Create model
        # ==============================================================================

        return tf.keras.Model(all_inputs, all_outputs)

    @staticmethod
    def _encapsulate_flyability(
        flyability_model: tf.keras.Model,
        nb_cells: int,
        nb_altitudes_or_nb_spots: int,
        input_dim_other: int,
        input_dim_rain: int,
        inputs: list[tf.Tensor],
    ) -> tf.Tensor:
        """
        Encapsulate flyability prediction with proper input reshaping for spots.

        Args:
            flyability_model: The flyability block model (frozen)
            nb_cells: Number of cells (1 for spots)
            nb_altitudes_or_nb_spots: Number of spots (replaces altitudes)
            input_dim_other: Other weather data dimensions
            input_dim_rain: Rain/humidity data dimensions
            inputs: [wind, other, rain] tensors

        Returns:
            Flyability prediction reshaped to (batch, 1, nb_spots)
        """
        wind, other, rain = inputs

        reshape_in = tf.keras.layers.Lambda(
            lambda x: [
                # wind: (batch, 1, nb_spots, 3) -> (batch, nb_spots, 3)
                tf.reshape(x[0], (-1, 3 * 1)),
                # other: tile over spots -> (batch, nb_spots, 3*input_dim_other)
                tf.reshape(
                    tf.tile(
                        tf.reshape(x[1], (-1, nb_cells, 1, 3, input_dim_other)),
                        (1, 1, nb_altitudes_or_nb_spots, 1, 1),
                    ),
                    (-1, 3 * input_dim_other),
                ),
                # rain: tile over spots -> (batch, nb_spots, 3*input_dim_rain)
                tf.reshape(
                    tf.tile(
                        tf.reshape(x[2], (-1, nb_cells, 1, 3, input_dim_rain)),
                        (1, 1, nb_altitudes_or_nb_spots, 1, 1),
                    ),
                    (-1, 3 * input_dim_rain),
                ),
            ]
        )

        reshape_out = tf.keras.layers.Lambda(
            lambda x: tf.reshape(x, (-1, nb_cells, nb_altitudes_or_nb_spots))
        )

        pre = reshape_in([wind, other, rain])
        flyability_prediction = flyability_model(pre)
        return reshape_out(flyability_prediction)
