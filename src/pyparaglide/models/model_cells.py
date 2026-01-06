"""
ModelCells - Grid-based flyability prediction model.

Predicts flyability for 1°×1° grid cells with 2 outputs:
- flown: Overall probability of flights
- crossed: Cross-country potential
"""

from typing import Any

import numpy as np
import tensorflow as tf

from pyparaglide.models.enums import ModelSettings, ModelType, ProblemFormulation
from pyparaglide.models.layers import (
    CrossabilityBlock,
    FlyabilityBlock,
    PopulationBlock,
    WindBlockCells,
)


class ModelCells:
    """
    Grid-based (CELLS) model for paragliding flyability prediction.

    After removing redundant indicators (2026-01-04):
    Output names (2 total):
        - flown: Overall flight probability
        - crossed: Cross-country potential
    """

    @classmethod
    def output_names(cls) -> list[str]:
        """Return the names of all model outputs."""
        return [
            "flown",
            "crossed",
        ]

    @classmethod
    def create_model(
        cls,
        problem_formulation: ProblemFormulation,
        nb_cells: int,
        wind_dim: int,
        other_dim: int,
        humidity_dim: int,
        nb_altitudes: int,
        thermo_dim: int = 0,  # NEW: thermodynamic parameters
        super_resolution: int = 1,
        initialization: dict[str, Any] | None = None,
    ) -> tf.keras.Model:
        """
        Create a new CELLS model.

        Args:
            problem_formulation: CLASSIFICATION or REGRESSION
            nb_cells: Number of grid cells
            wind_dim: Wind direction dimensions (typically 8)
            other_dim: Other weather data dimensions
            humidity_dim: Humidity/rain data dimensions
            nb_altitudes: Number of altitude levels (typically 5)
            super_resolution: Super-resolution factor for population block
            initialization: Optional dict with 'date_factor' and 'dow_factor'

        Returns:
            Compiled Keras model
        """
        # ==============================================================================
        # Shared variables
        # ==============================================================================

        var_date_factor = tf.Variable(
            np.array([[1.275]], dtype=np.float32), name="var_date_factor", trainable=True
        )

        if ModelSettings.optimize_dow:
            var_dow_factor = tf.Variable(
                np.array(ModelSettings.dow_init, dtype=np.float32),
                name="var_dow_factor",
                trainable=True,
            )
        else:
            var_dow_factor = tf.constant(
                np.array(ModelSettings.dow_init, dtype=np.float32),
                name="var_dow_factor",
            )

        # ==============================================================================
        # Inputs
        # ==============================================================================

        input_date = tf.keras.layers.Input(shape=(1,), name="in_date")
        input_dow = tf.keras.layers.Input(shape=(7,), name="in_dow")
        input_mountainess = tf.keras.layers.Input(
            shape=(nb_cells, 1), name="in_mountainess"  # nb_altitudes removed
        )
        input_other = tf.keras.layers.Input(shape=(nb_cells, 3, other_dim), name="in_other")
        input_humidity = tf.keras.layers.Input(shape=(nb_cells, 3, humidity_dim), name="in_rain")
        input_wind = tf.keras.layers.Input(
            shape=(nb_cells, 1, 3, wind_dim), name="in_wind"  # nb_altitudes -> 1
        )

        # NEW: Add thermo input (always created, even for baseline with thermo_dim=0)
        input_thermo = tf.keras.layers.Input(
            shape=(nb_cells, 3, thermo_dim), name="in_thermo"
        )
        all_inputs = [
            input_date,
            input_dow,
            input_mountainess,
            input_other,
            input_humidity,
            input_wind,
            input_thermo,
        ]

        # ==============================================================================
        # Blocks
        # ==============================================================================

        wind_block = WindBlockCells(name="wind_block_cells")
        flyability_block = FlyabilityBlock(other_dim, humidity_dim, thermo_dim, name="flyability_block")  # NEW: thermo_dim
        crossability_block = CrossabilityBlock(
            other_dim, humidity_dim, nb_altitudes, nb_cells, name="crossability_block"
        )

        # Create separate population blocks for each output
        # (Original uses single block, but requires all inputs to have same shape)
        population_block = PopulationBlock(
            problem_formulation,
            var_date_factor,
            var_dow_factor,
            super_resolution,
            name="population_block",
        )

        # ==============================================================================
        # Flyability/crossability computation
        # ==============================================================================

        wind_prediction = wind_block([input_mountainess, input_wind])
        flyability_prediction = cls._encapsulate_flyability(
            flyability_block,
            nb_cells,
            nb_altitudes,
            other_dim,
            humidity_dim,
            [wind_prediction, input_other, input_humidity, input_thermo],  # NEW: added input_thermo
            thermo_dim,  # NEW
        )
        crossability_prediction = crossability_block(
            [flyability_prediction, wind_prediction, input_other, input_humidity]
        )

        # ==============================================================================
        # Apply population (separate blocks for each output)
        # ==============================================================================

        flown_prediction = population_block([flyability_prediction, input_date, input_dow])
        crossed_prediction = population_block([crossability_prediction, input_date, input_dow])

        # ==============================================================================
        # Create model
        # ==============================================================================

        return tf.keras.Model(
            all_inputs,
            [flown_prediction, crossed_prediction],
        )

    @staticmethod
    def _encapsulate_flyability(
        flyability_model: tf.keras.Model,
        nb_cells: int,
        nb_altitudes: int,
        input_dim_other: int,
        input_dim_rain: int,
        inputs: list[tf.Tensor],
        input_dim_thermo: int = 0,  # NEW: thermo dimension
    ) -> tf.Tensor:
        """
        Encapsulate flyability prediction (simplified for nb_altitudes=1).

        Reshapes inputs for the flyability model without altitude tiling.

        Args:
            flyability_model: The flyability block model
            nb_cells: Number of grid cells
            nb_altitudes: Always 1 (altitude binning removed)
            input_dim_other: Other weather data dimensions
            input_dim_rain: Rain/humidity data dimensions
            input_dim_thermo: Thermo data dimensions
            inputs: [wind, other, rain, thermo] tensors

        Returns:
            Flyability prediction reshaped to (batch, nb_cells, 1)
        """
        wind, other, rain, thermo = inputs  # NEW: added thermo

        # Simplified reshaping for nb_altitudes=1 (no tiling needed)
        # Reshape to (batch * nb_cells, feature_dim) for flyability block processing
        reshape_in = tf.keras.layers.Lambda(
            lambda x, d_other=input_dim_other, d_rain=input_dim_rain, d_thermo=input_dim_thermo: [
                # wind: (batch, nb_cells, 1, 3) -> (batch * nb_cells, 3)
                tf.reshape(x[0], (-1, 3)),
                # other: (batch, nb_cells, 3, input_dim_other) -> (batch * nb_cells, 3*input_dim_other)
                tf.reshape(x[1], (-1, 3 * d_other)),
                # rain: (batch, nb_cells, 3, input_dim_rain) -> (batch * nb_cells, 3*input_dim_rain)
                tf.reshape(x[2], (-1, 3 * d_rain)),
                # thermo: (batch, nb_cells, 3, input_dim_thermo) -> (batch * nb_cells, 3*input_dim_thermo)
                # IMPORTANT: Use explicit batch dimension from wind to avoid issues when d_thermo=0
                tf.reshape(x[3], (tf.shape(x[0])[0] * tf.shape(x[0])[1], 3 * d_thermo)),
            ]
        )

        reshape_out = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, nb_cells, 1)))

        pre = reshape_in([wind, other, rain, thermo])
        flyability_prediction = flyability_model(pre)
        return reshape_out(flyability_prediction)
