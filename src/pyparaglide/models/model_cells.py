"""
ModelCells - Grid-based flyability prediction model.

Predicts flyability for 1°×1° grid cells with multiple outputs:
- flown: Overall probability of flights
- crossed: Cross-country potential
- wind_flown: Wind-based flyability
- humidity_flown: Humidity-based flyability
"""

from typing import Any

import numpy as np
import tensorflow as tf

from pyparaglide.models.enums import ModelSettings, ModelType, ProblemFormulation
from pyparaglide.models.layers import (
    CrossabilityBlock,
    FlyabilityBlock,
    HumidityFlyabilityBlock,
    PopulationBlock,
    WindBlockCells,
    WindFlyabilityBlock,
)


class ModelCells:
    """
    Grid-based (CELLS) model for paragliding flyability prediction.

    Output names (21 total - 5 altitudes × 4 predictions + 1):
        - flown 1000/900/800/700/600: Overall flight probability
        - flown fufu 1000/900/800/700/600: Cross-country potential
        - flown of wind 1000/900/800/700/600: Wind-based flyability
        - flown of rain 1000/900/800/700/600: Humidity/rain-based flyability
    """

    @classmethod
    def output_names(cls) -> list[str]:
        """Return the names of all model outputs."""
        return [
            "flown 1000",
            "flown  900",
            "flown  800",
            "flown  700",
            "flown  600",
            "flown  fufu 1000",
            "flown  fufu  900",
            "flown  fufu  800",
            "flown  fufu  700",
            "flown  fufu  600",
            "flown of wind 1000",
            "flown of wind  900",
            "flown of wind  800",
            "flown of wind  700",
            "flown of wind  600",
            "flown of rain 1000",
            "flown of rain  900",
            "flown of rain  800",
            "flown of rain  700",
            "flown of rain  600",
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
        # Blocks
        # ==============================================================================

        wind_block = WindBlockCells(name="wind_block_cells")
        flyability_block = FlyabilityBlock(other_dim, humidity_dim, name="flyability_block")
        crossability_block = CrossabilityBlock(
            other_dim, humidity_dim, nb_altitudes, nb_cells, name="crossability_block"
        )
        wind_flyability_block = WindFlyabilityBlock(nb_altitudes, nb_cells, humidity_dim)
        humidity_flyability_block = HumidityFlyabilityBlock(nb_altitudes, nb_cells, humidity_dim)

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
            [wind_prediction, input_other, input_humidity],
        )
        crossability_prediction = crossability_block(
            [flyability_prediction, wind_prediction, input_other, input_humidity]
        )
        wind_flyability_prediction = wind_flyability_block(wind_prediction)
        humidity_flyability_prediction = humidity_flyability_block(input_humidity)

        # ==============================================================================
        # Apply population (separate blocks for each output)
        # ==============================================================================

        flown_prediction = population_block([flyability_prediction, input_date, input_dow])
        crossed_prediction = population_block([crossability_prediction, input_date, input_dow])
        wind_flown_prediction = population_block([wind_flyability_prediction, input_date, input_dow])
        humidity_flown_prediction = population_block([humidity_flyability_prediction, input_date, input_dow])

        # ==============================================================================
        # Create model
        # ==============================================================================

        return tf.keras.Model(
            all_inputs,
            [flown_prediction, crossed_prediction, wind_flown_prediction, humidity_flown_prediction],
        )

    @staticmethod
    def _encapsulate_flyability(
        flyability_model: tf.keras.Model,
        nb_cells: int,
        nb_altitudes: int,
        input_dim_other: int,
        input_dim_rain: int,
        inputs: list[tf.Tensor],
    ) -> tf.Tensor:
        """
        Encapsulate flyability prediction with proper input reshaping.

        Extrudes other and rain data over all altitudes, then reshapes for the flyability model.

        Args:
            flyability_model: The flyability block model
            nb_cells: Number of grid cells
            nb_altitudes: Number of altitude levels
            input_dim_other: Other weather data dimensions
            input_dim_rain: Rain/humidity data dimensions
            inputs: [wind, other, rain] tensors

        Returns:
            Flyability prediction reshaped to (batch, nb_cells, nb_altitudes)
        """
        wind, other, rain = inputs

        reshape_in = tf.keras.layers.Lambda(
            lambda x: [
                # wind: (batch, nb_cells, nb_altitudes, 3) -> (batch, nb_cells*nb_altitudes, 3)
                tf.reshape(x[0], (-1, 3 * 1)),
                # other: tile over altitudes -> (batch, nb_cells*nb_altitudes, 3*input_dim_other)
                tf.reshape(
                    tf.tile(
                        tf.reshape(x[1], (-1, nb_cells, 1, 3, input_dim_other)),
                        (1, 1, nb_altitudes, 1, 1),
                    ),
                    (-1, 3 * input_dim_other),
                ),
                # rain: tile over altitudes -> (batch, nb_cells*nb_altitudes, 3*input_dim_rain)
                tf.reshape(
                    tf.tile(
                        tf.reshape(x[2], (-1, nb_cells, 1, 3, input_dim_rain)),
                        (1, 1, nb_altitudes, 1, 1),
                    ),
                    (-1, 3 * input_dim_rain),
                ),
            ]
        )

        reshape_out = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, nb_cells, nb_altitudes)))

        pre = reshape_in([wind, other, rain])
        flyability_prediction = flyability_model(pre)
        return reshape_out(flyability_prediction)
