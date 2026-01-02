"""
Custom Keras layers for PyParaglide models.

Migrated from TensorFlow 1.15 to TensorFlow 2.x:
- Removed compute_output_shape() methods (automatic in TF2)
- Replaced tf.keras.backend.* with tf.* equivalents
- Added type hints and modern Python patterns
"""

from typing import Any

import numpy as np
import tensorflow as tf

from pyparaglide.models.enums import ProblemFormulation


class WindFlyabilityBlock(tf.keras.Model):
    """
    Wind flyability prediction block.

    Input shape: (batch, nb_cells, nb_altitudes, 3)
    Output shape: (batch, nb_cells, nb_altitudes)
    """

    def __init__(self, nb_altitudes: int, nb_cells: int, humidity_dim: int, name: str = "wind_flyability_block"):
        super().__init__(name=name)
        self.nb_altitudes = nb_altitudes
        self.nb_cells = nb_cells
        self.humidity_dim = humidity_dim
        self.dropout_rate = 0.05

        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense1 = tf.keras.layers.Dense(8, activation="tanh", name="WindFlyability_1")
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense2 = tf.keras.layers.Dense(1, activation="sigmoid", name="WindFlyability_2")

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        # Input: (batch, nb_cells, nb_altitudes, 3)
        # Reshape to (batch * nb_cells * nb_altitudes, 3) - flatten everything except last dim
        x = tf.reshape(inputs, (-1, 3))

        x = self.dropout1(x, training=training)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        x = self.dense2(x)

        # Reshape back to (batch, nb_cells, nb_altitudes)
        return tf.reshape(x, (-1, self.nb_cells, self.nb_altitudes))


class HumidityFlyabilityBlock(tf.keras.Model):
    """
    Humidity/Rain flyability prediction block.

    Predicts flyability based on humidity/rain data.
    """

    def __init__(self, nb_altitudes: int, nb_cells: int, humidity_dim: int, name: str = "humidity_flyability_block"):
        super().__init__(name=name)
        self.nb_altitudes = nb_altitudes
        self.nb_cells = nb_cells
        self.humidity_dim = humidity_dim
        self.dropout_rate = 0.05

        self.reshape = tf.keras.layers.Lambda(lambda x: tf.reshape(x, (-1, 3 * humidity_dim)))
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense1 = tf.keras.layers.Dense(4, activation="tanh", name="RainFlyability_1")
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense2 = tf.keras.layers.Dense(1, activation="sigmoid", name="RainFlyability_2")
        self.tile = tf.keras.layers.Lambda(
            lambda x: tf.tile(tf.reshape(x, (-1, nb_cells, 1)), (1, 1, nb_altitudes))
        )

    def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
        x = self.reshape(inputs)
        x = self.dropout1(x, training=training)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)
        x = self.dense2(x)
        return self.tile(x)


class FlyabilityBlock(tf.keras.Model):
    """
    Main flyability prediction block.

    Combines wind, other weather, and rain data to predict flyability.
    """

    def __init__(self, other_dim: int, humidity_dim: int, name: str = "flyability_block"):
        super().__init__(name=name)
        self.other_dim = other_dim
        self.humidity_dim = humidity_dim
        self.batch_normalization = True
        self.dropout_rate = 0.0

        self.concat = tf.keras.layers.Concatenate(name="concatenate_flyability")
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense1 = tf.keras.layers.Dense(32, use_bias=not self.batch_normalization, name="Flyability_1A")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation("tanh")
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense2 = tf.keras.layers.Dense(16, use_bias=not self.batch_normalization, name="Flyability_1B")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.Activation("tanh")
        self.dropout3 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense3 = tf.keras.layers.Dense(1, activation="sigmoid", name="Flyability_2")

    def call(self, inputs: list[tf.Tensor], training: bool = False) -> tf.Tensor:
        wind, other, rain = inputs
        x = self.concat([wind, other, rain])
        x = self.dropout1(x, training=training)
        x = self.dense1(x)
        if self.batch_normalization:
            x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.dropout2(x, training=training)
        x = self.dense2(x)
        if self.batch_normalization:
            x = self.bn2(x, training=training)
        x = self.act2(x)

        x = self.dropout3(x, training=training)
        return self.dense3(x)


class CrossabilityBlock(tf.keras.Model):
    """
    Cross-country flyability prediction block (fufu).

    Predictes cross-country potential based on flyability and weather data.
    """

    def __init__(
        self,
        other_dim: int,
        humidity_dim: int,
        nb_altitudes: int,
        nb_cells: int,
        name: str = "crossability_block",
    ):
        super().__init__(name=name)
        self.other_dim = other_dim
        self.humidity_dim = humidity_dim
        self.nb_altitudes = nb_altitudes
        self.nb_cells = nb_cells
        self.nbH = 3
        self.batch_normalization = True
        self.dropout_rate = 0.0

        self.reshape = tf.keras.layers.Lambda(
            lambda x: [
                tf.reshape(x[0], (-1, nb_altitudes)),  # flyability
                tf.reshape(x[1], (-1, nb_altitudes * self.nbH)),  # wind
                tf.reshape(x[2], (-1, self.nbH * other_dim)),  # other
                tf.reshape(x[3], (-1, self.nbH * humidity_dim)),  # rain
            ]
        )
        self.concat = tf.keras.layers.Concatenate(name="concatenate_fufu", axis=-1)
        self.dropout1 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense1 = tf.keras.layers.Dense(32, use_bias=not self.batch_normalization, name="Fufu_1A")
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.act1 = tf.keras.layers.Activation("tanh")
        self.dropout2 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense2 = tf.keras.layers.Dense(16, use_bias=not self.batch_normalization, name="Fufu_1B")
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.act2 = tf.keras.layers.Activation("tanh")
        self.dropout3 = tf.keras.layers.Dropout(self.dropout_rate)
        self.dense3 = tf.keras.layers.Dense(1, activation="sigmoid", name="Fufu_2")
        self.tile = tf.keras.layers.Lambda(
            lambda x: tf.tile(tf.reshape(x, (-1, nb_cells, 1)), (1, 1, nb_altitudes))
        )

    def call(self, inputs: list[tf.Tensor], training: bool = False) -> tf.Tensor:
        flyability, fufu_wind, fufu_other, fufu_rain = inputs
        x = self.reshape([flyability, fufu_wind, fufu_other, fufu_rain])
        x = self.concat(x)
        x = self.dropout1(x, training=training)
        x = self.dense1(x)
        if self.batch_normalization:
            x = self.bn1(x, training=training)
        x = self.act1(x)

        x = self.dropout2(x, training=training)
        x = self.dense2(x)
        if self.batch_normalization:
            x = self.bn2(x, training=training)
        x = self.act2(x)

        x = self.dropout3(x, training=training)
        x = self.dense3(x)
        return self.tile(x)


class WindBlockCells(tf.keras.layers.Layer):
    """
    Wind processing block for CELLS model.

    Adjusts wind prediction based on mountainous terrain.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape: list[tf.TensorShape]) -> None:
        """Build the layer weights."""
        # Mountainous terrain adjustment factor
        self.mountainess_factor = self.add_weight(
            name="mountainess_factor",
            shape=(1,),
            trainable=True,
            initializer=tf.keras.initializers.Constant(value=np.zeros((1,))),
            dtype=np.float32,
        )
        super().build(input_shape)

    def call(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        """
        Process wind data with mountainous terrain adjustment.

        Args:
            inputs: [mountainess, wind]
                - mountainess: (batch, nbCells, nbWindAltitudes)
                - wind: (batch, nbCells, nbWindAltitudes, nbHours, nbWindDims)

        Returns:
            Wind prediction adjusted for terrain: (batch, nbCells, nbWindAltitudes, nbHours)
        """
        mountainess, wind = inputs

        # Expand mountainess to match wind dimensions (nbHours)
        mountainess_expanded = tf.tile(tf.expand_dims(mountainess, -1), (1, 1, 1, tf.shape(wind)[-2]))

        # Sum wind over wind direction dimension
        wind_norm = tf.reduce_sum(wind, axis=-1)

        # Apply mountainous terrain factor
        wind_prediction = (1.0 + mountainess_expanded * self.mountainess_factor[0]) * wind_norm

        return wind_prediction


class WindBlockSpots(tf.keras.layers.Layer):
    """
    Wind processing block for SPOTS model.

    Estimates spot-specific wind values with learned:
    - Wind direction weights (8 directions)
    - Relevant altitude
    """

    nb_hours = 3
    wind_dim = 8

    def __init__(self, nb_spots: int, **kwargs):
        super().__init__(**kwargs)

        self.nb_spots = nb_spots

        initial_wind_weight_value = 1.0
        initial_alt_value = 2.0

        self.initial_weights = [
            tf.keras.initializers.Constant(
                value=initial_wind_weight_value * np.ones((nb_spots, self.wind_dim))
            ),
            tf.keras.initializers.Constant(value=initial_alt_value),
        ]

    def build(self, input_shape: tf.TensorShape) -> None:
        """Build the layer weights."""
        nb_altitudes = int(input_shape[1])

        # Wind direction weights for each spot
        self.windWeights = self.add_weight(
            name="windWeights",
            shape=(self.nb_spots, self.wind_dim),
            trainable=True,
            initializer=self.initial_weights[0],
        )

        # Relevant altitude for each spot (constrained to valid range)
        self.alt = self.add_weight(
            name="alt",
            shape=(self.nb_spots, 1),
            trainable=True,
            initializer=self.initial_weights[1],
            constraint=tf.keras.constraints.MinMaxNorm(min_value=0.0, max_value=nb_altitudes - 1.0),
        )

        super().build(input_shape)

    def call(self, x: tf.Tensor) -> tf.Tensor:
        """
        Process wind data for each spot.

        Args:
            x: Wind tensor (batch, nbAltitudes, nbHours, nbWindDirections)

        Returns:
            Spot-specific wind values (batch, nb_spots, nb_hours)
        """
        nb_altitudes = tf.shape(x)[1]
        wind_dim = tf.shape(x)[3]

        # Permute and reshape for processing
        x_permuted = tf.transpose(x, [0, 2, 1, 3])  # (batch, nbHours, nbAltitudes, nbWindDirections)
        x_reshaped = tf.reshape(x_permuted, (-1, wind_dim))

        results = []

        # Process each spot
        for s in range(self.nb_spots):
            # Create interpolation kernel for altitude selection
            altitude_range = tf.cast(tf.range(nb_altitudes), tf.float32)
            interpolation_kernel = tf.clip_by_value(altitude_range - self.alt[s, :] + 1.0, 0.0, 1.0) - tf.clip_by_value(
                altitude_range - self.alt[s, :], 0.0, 1.0
            )

            # Compute wind factor using dot product
            wind_factor = tf.einsum("bi,i->b", x_reshaped, tf.reshape(self.windWeights[s, :], (self.wind_dim,)))
            wind_factor_each_alt = tf.reshape(wind_factor, (-1, nb_altitudes))
            wind_factor_relevant_alt = tf.einsum("ba,a->b", wind_factor_each_alt, tf.squeeze(interpolation_kernel))
            wind_factor_relevant_alt = tf.reshape(wind_factor_relevant_alt, (-1, 3))

            results.append(wind_factor_relevant_alt)

        # Stack results for all spots
        result = tf.stack(results, axis=1)  # (batch, nb_spots, 3)
        return result


class PopulationBlock(tf.keras.layers.Layer):
    """
    Population block modeling pilot behavior.

    Models the probability that at least one person flies given:
    - Flyability prediction
    - Pilot population
    - Date factor (seasonality)
    - Day-of-week factor
    """

    def __init__(
        self,
        problem_formulation: ProblemFormulation,
        var_date_factor: tf.Tensor,
        var_dow_factor: tf.Tensor,
        super_resolution: int,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.frozen_date_factor = not isinstance(var_date_factor, tf.Variable)
        self.frozen_dow_factor = not isinstance(var_dow_factor, tf.Variable)
        self.problem_formulation = problem_formulation
        self.var_date_factor = var_date_factor
        self.var_dow_factor = var_dow_factor
        self.super_resolution = super_resolution

    def build(self, input_shape: list[tf.TensorShape]) -> None:
        """Build the layer weights."""
        shape_prediction, shape_date, shape_dow = input_shape

        # Population weights for each cell and altitude
        self.popu = self.add_weight(
            name="kernel",
            shape=(
                int(shape_prediction[-2]) * self.super_resolution * self.super_resolution,
                int(shape_prediction[-1]),
            ),  # (nbCells*super_resolution^2, nbAltitudes)
            trainable=True,
            initializer=tf.keras.initializers.Constant(value=0.5),
            dtype=np.float32,
            constraint=tf.keras.constraints.NonNeg(),
        )

        # Add trainable variables for date and dow factors
        trainable_weights = []
        if not self.frozen_date_factor:
            trainable_weights.append(self.var_date_factor)
        if not self.frozen_dow_factor:
            trainable_weights.append(self.var_dow_factor)
        if trainable_weights:
            self.trainable_weights.extend(trainable_weights)

        super().build(input_shape)

    def call(self, inputs: list[tf.Tensor]) -> tf.Tensor:
        """
        Apply population model to predictions.

        Args:
            inputs: [prediction, date, dow]
                - prediction: (batch, nbCells, nbAltitudes)
                - date: (batch, 1)
                - dow: (batch, 7)

        Returns:
            Prediction with population applied
        """
        prediction, date, dow = inputs

        # Expand prediction to super resolution (use repeat_elements like original)
        prediction = tf.keras.backend.repeat_elements(
            prediction, self.super_resolution * self.super_resolution, axis=1
        )  # (batch, nbCells*super_resolution^2, nbAltitudes)

        # Tile population weights
        popu_reshaped = tf.reshape(
            self.popu, (1, int(self.popu.shape[0]), int(self.popu.shape[1]))
        )  # (1, nbCells*super_resolution^2, nbAltitudes)
        tiled_popu = tf.tile(popu_reshaped, (tf.shape(prediction)[0], 1, 1))

        # Compute day factor (seasonality * day of week)
        # Use batch_dot like original: dow (batch, 7) x dow_factor (7, 1) -> (batch, 1)
        dow_factor_reshaped = tf.reshape(self.var_dow_factor, (-1, 1))  # (7, 1)
        dow_factor_dot = tf.matmul(dow, dow_factor_reshaped)  # (batch, 1)
        day_factor_scalar = (1.0 + self.var_date_factor * date) * dow_factor_dot  # (batch, 1)
        day_factor_vector = tf.reshape(day_factor_scalar, (-1, 1, 1))  # (batch, 1, 1)
        # KEY FIX: first element is 1 (batch dim), not tf.shape(tiled_popu)[0]
        day_factor_vector = tf.tile(
            day_factor_vector, (1, int(tiled_popu.shape[1]), int(tiled_popu.shape[2]))
        )  # (batch, nbCells*super_resolution^2, nbAlts)
        tiled_popu = day_factor_vector * tiled_popu

        # Apply population model
        if self.problem_formulation == ProblemFormulation.CLASSIFICATION:
            pred_with_popu = tf.where(
                tiled_popu > 1.0,
                (1.0 - tf.pow(1.0 - prediction, tf.clip_by_value(tiled_popu, 0.0, 100.0))),
                tiled_popu * prediction,
            )
        else:  # REGRESSION
            pred_with_popu = tiled_popu * prediction

        return pred_with_popu
