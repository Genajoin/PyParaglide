"""
Model trainer for PyParaglide.

Handles training of both CELLS and SPOTS models.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import tensorflow as tf

from pyparaglide.data import Dataset, Normalization
from pyparaglide.data.normalization import apply_normalization, compute_normalization_coeffs
from pyparaglide.models import ModelCells, ModelSpots, ModelType, ProblemFormulation
from pyparaglide.training.callbacks import LearningRateScheduler, TrainingLogger


class Trainer:
    """
    Main training class for PyParaglide models.

    Supports training both CELLS (grid-based) and SPOTS (location-specific) models.
    """

    def __init__(
        self,
        data_dir: Path | str,
        model_type: ModelType,
        problem_formulation: ProblemFormulation = ProblemFormulation.CLASSIFICATION,
        models_dir: Path | str = "data/models",
    ):
        """
        Initialize trainer.

        Args:
            data_dir: Directory containing PKL files
            model_type: CELLS or SPOTS
            problem_formulation: CLASSIFICATION or REGRESSION
            models_dir: Directory to save/load model weights
        """
        self.data_dir = Path(data_dir)
        self.model_type = model_type
        self.problem_formulation = problem_formulation
        self.models_dir = Path(models_dir)

        # Model parameters
        self.wind_dim = 8
        self.nb_altitudes = 5

        # Load dataset
        self.dataset = Dataset(data_dir)
        self.nb_cells = self.dataset.nb_cells
        self.nb_days = self.dataset.nb_days

        # Model will be created later
        self.model: tf.keras.Model | None = None
        self.normalization: Normalization | None = None

    def prepare_data(self, cells: list[int] | None = None, super_resolution: int = 1) -> tuple[list, list]:
        """
        Prepare training data for given cells.

        Args:
            cells: List of cell indices to train on (None = all cells)
            super_resolution: Super-resolution factor (1 = normal)

        Returns:
            (X, Y) tuple of input and output data
        """
        if cells is None:
            cells = list(range(self.nb_cells))

        # Load weather data
        X_other = [self.dataset.get_meteo_matrix(cells, Dataset.METEO_PARAMS_OTHER[h]) for h in range(3)]
        X_wind = [self._convert_wind(self.dataset.get_meteo_matrix(cells, Dataset.METEO_PARAMS_WIND[h])) for h in range(3)]
        X_humidity = [self.dataset.get_meteo_matrix(cells, Dataset.METEO_PARAMS_HUMIDITY[h]) for h in range(3)]

        # Compute and save normalization (based on hour 1 = 12h)
        norm_mean_other, norm_std_other = compute_normalization_coeffs(X_other[1])
        norm_mean_hum, norm_std_hum = compute_normalization_coeffs(X_humidity[1])

        self.normalization = Normalization(
            other_mean=norm_mean_other,
            other_std=norm_std_other,
            humidity_mean=norm_mean_hum,
            humidity_std=norm_std_hum,
        )

        # Apply normalization
        for h in range(3):
            X_other[h] = apply_normalization(X_other[h], norm_mean_other, norm_std_other)
            X_humidity[h] = apply_normalization(X_humidity[h], norm_mean_hum, norm_std_hum)

        # Prepare inputs
        X = self._prepare_inputs(cells, X_other, X_wind, X_humidity, super_resolution)
        Y = self._prepare_outputs(cells, super_resolution)

        return X, Y

    def _convert_wind(self, wind_matrix: np.ndarray) -> np.ndarray:
        """Convert wind matrix to direction encoding."""
        # Simplified wind encoding (8 directions × 5 altitudes)
        nb_samples = wind_matrix.shape[0]
        result = np.zeros((nb_samples, self.nb_altitudes, self.wind_dim), dtype=np.float32)

        # Encode U/V components into direction bins
        for i in range(nb_samples):
            for alt in range(self.nb_altitudes):
                if alt * 2 + 1 < wind_matrix.shape[1]:
                    u = wind_matrix[i, alt * 2]
                    v = wind_matrix[i, alt * 2 + 1]
                    # Simple encoding: magnitude in first bin
                    result[i, alt, 0] = np.sqrt(u * u + v * v)
                    # Direction could be computed from arctan2(v, u)

        return result.reshape(nb_samples, self.nb_altitudes * self.wind_dim)

    def _prepare_inputs(
        self,
        cells: list[int],
        X_other: list[np.ndarray],
        X_wind: list[np.ndarray],
        X_humidity: list[np.ndarray],
        super_resolution: int,
    ) -> list:
        """Prepare input tensors for model."""
        nb_cells_model = len(cells)

        # Date (nb_days,)
        X_date = self.dataset.get_date()

        # Day of week (nb_days, 7)
        X_dow = self.dataset.get_dow()

        # Mountainess (nb_days, nb_cells, nb_altitudes)
        mountainess = self.dataset.get_mountainess(cells, self.nb_altitudes)
        X_mountainess = np.repeat(mountainess[np.newaxis, :, :], self.nb_days, axis=0)

        # Other weather (nb_days, nb_cells, 3, dim_other)
        dim_other = X_other[0].shape[1]
        X_other_stacked = np.zeros((self.nb_days, nb_cells_model, 3, dim_other), dtype=np.float32)
        for h in range(3):
            for i, cell in enumerate(cells):
                start_idx = cell * self.nb_days
                X_other_stacked[:, i, h, :] = X_other[h][start_idx : start_idx + self.nb_days, :]

        # Humidity (nb_days, nb_cells, 3, dim_humidity)
        dim_humidity = X_humidity[0].shape[1]
        X_humidity_stacked = np.zeros((self.nb_days, nb_cells_model, 3, dim_humidity), dtype=np.float32)
        for h in range(3):
            for i, cell in enumerate(cells):
                start_idx = cell * self.nb_days
                X_humidity_stacked[:, i, h, :] = X_humidity[h][start_idx : start_idx + self.nb_days, :]

        # Wind (nb_days, nb_cells, nb_altitudes, 3, wind_dim)
        X_wind_stacked = np.zeros((self.nb_days, nb_cells_model, self.nb_altitudes, 3, self.wind_dim), dtype=np.float32)
        for h in range(3):
            wind_reshaped = X_wind[h].reshape(self.nb_days, nb_cells_model, self.nb_altitudes, self.wind_dim)
            X_wind_stacked[:, :, :, h, :] = wind_reshaped

        return [X_date, X_dow, X_mountainess, X_other_stacked, X_humidity_stacked, X_wind_stacked]

    def _prepare_outputs(self, cells: list[int], super_resolution: int) -> list:
        """Prepare output tensors for model."""
        if self.model_type == ModelType.CELLS:
            return self.dataset.get_flights_by_altitude(cells, self.nb_altitudes, super_resolution, self.problem_formulation == ProblemFormulation.REGRESSION)
        else:
            # SPOTS model
            spots_data = self.dataset.get_flights_by_spots(cells)
            return [np.stack([spots_data[c][s] for s in range(len(spots_data[c]))], axis=1) for c in cells if len(spots_data[c]) > 0]

    def create_model(
        self,
        cells: list[int],
        super_resolution: int = 1,
        load_weights: bool = False,
    ) -> None:
        """
        Create and optionally load model weights.

        Args:
            cells: List of cell indices
            super_resolution: Super-resolution factor
            load_weights: Whether to load existing weights
        """
        tf.keras.backend.clear_session()

        if self.model_type == ModelType.CELLS:
            self.model = ModelCells.create_model(
                problem_formulation=self.problem_formulation,
                nb_cells=len(cells),
                wind_dim=self.wind_dim,
                other_dim=45,  # Will be updated from data
                humidity_dim=2,
                nb_altitudes=self.nb_altitudes,
                super_resolution=super_resolution,
            )
        else:
            # SPOTS model - create cells data structure
            cells_data = {}
            for cell in cells:
                cells_data[cell] = {"spots": list(range(10))}  # Placeholder

            self.model = ModelSpots.create_model(
                problem_formulation=self.problem_formulation,
                cells_data=cells_data,
                wind_dim=self.wind_dim,
                other_dim=45,
                humidity_dim=2,
                nb_altitudes=self.nb_altitudes,
                initialization={"date_factor": np.array([[1.275]]), "dow_factor": np.array([[1.0] * 7])},
            )

        if load_weights:
            self._load_weights()

    def _load_weights(self) -> None:
        """Load model weights from directory."""
        # Try to load weights
        weight_path = self.models_dir / f"{self.model_type.name.lower()}_weights.h5"
        if weight_path.exists():
            self.model.load_weights(weight_path)
            print(f"[INFO] Loaded weights from {weight_path}")

    def save_weights(self) -> None:
        """Save model weights to directory."""
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Save normalization
        if self.normalization:
            norm_path = self.models_dir / f"normalization_{self.model_type.name.lower()}.pkl"
            self.normalization.save(norm_path)

        # Save model weights
        weight_path = self.models_dir / f"{self.model_type.name.lower()}_weights.h5"
        self.model.save_weights(weight_path)
        print(f"[INFO] Saved weights to {weight_path}")

    def train(
        self,
        X: list,
        Y: list,
        lr_init: float = 0.008,
        lr_end: float = 7e-4,
        nb_epochs: int = 55,
        batch_size: int = 32,
        validation_split: float = 0.0,
        use_validation_set: bool = False,
        freeze_crossability: bool = False,
    ) -> dict:
        """
        Train the model.

        Args:
            X: Input data
            Y: Output data
            lr_init: Initial learning rate
            lr_end: Final learning rate
            nb_epochs: Number of epochs
            batch_size: Batch size
            validation_split: Fraction of data to use for validation
            use_validation_set: Whether to use a specific validation set
            freeze_crossability: Only train crossability block (CELLS only)

        Returns:
            Training history
        """
        # Compile model
        self.model.compile(optimizer="adam", loss="binary_crossentropy" if self.problem_formulation == ProblemFormulation.CLASSIFICATION else "mse")

        # Prepare validation split
        if use_validation_set:
            # Use every 2nd day for validation
            val_indices = np.arange(0, self.nb_days, 2)
            train_indices = np.arange(1, self.nb_days, 2)
        else:
            val_indices = np.array([])
            train_indices = np.arange(self.nb_days)

        # Prepare callbacks
        log_file = self.models_dir / f"{self.model_type.name.lower()}.log"
        callbacks = [
            LearningRateScheduler.create(lr_init=lr_init, lr_end=lr_end, nb_epochs=nb_epochs),
            TrainingLogger(self.model_type, log_file),
        ]

        # Prepare data
        X_train = [x[train_indices] for x in X]
        Y_train = [y[train_indices] for y in Y]

        val_data = None
        if len(val_indices) > 0:
            X_val = [x[val_indices] for x in X]
            Y_val = [y[val_indices] for y in Y]
            val_data = (X_val, Y_val)

        # Train
        history = self.model.fit(
            X_train,
            Y_train,
            validation_data=val_data,
            epochs=nb_epochs,
            batch_size=batch_size,
            shuffle=True,
            verbose=0,
            callbacks=callbacks,
        )

        return history.history

    def evaluate(self, X: list, Y: list) -> dict:
        """
        Evaluate the model.

        Args:
            X: Input data
            Y: Output data

        Returns:
            Dictionary of metrics
        """
        results = self.model.evaluate(X, Y, verbose=0)
        if not isinstance(results, dict):
            results = {"loss": results}
        return results
