"""
Model trainer for PyParaglide.

Handles training of CELLS model for grid-based flyability prediction.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import tensorflow as tf

from pyparaglide.data import Dataset, Normalization
from pyparaglide.data.dataset import convert_wind_matrix
from pyparaglide.data.normalization import apply_normalization, compute_normalization_coeffs
from pyparaglide.models import ModelCells, ModelType, ProblemFormulation
from pyparaglide.training.callbacks import LearningRateScheduler, TrainingLogger


class Trainer:
    """
    Main training class for PyParaglide CELLS model.

    Trains grid-based flyability prediction for 1°×1° cells.
    """

    def __init__(
        self,
        data_dir: Path | str,
        model_type: ModelType = ModelType.CELLS,
        problem_formulation: ProblemFormulation = ProblemFormulation.CLASSIFICATION,
        models_dir: Path | str = "data/models",
        thermo_dim: int = 0,  # NEW: number of thermo parameters (0 or 4)
    ):
        """
        Initialize trainer.

        Args:
            data_dir: Directory containing PKL files
            model_type: Must be ModelType.CELLS
            problem_formulation: CLASSIFICATION or REGRESSION
            models_dir: Directory to save/load model weights
            thermo_dim: Number of thermo parameters (0 for baseline, 4 for thermo-enhanced)
        """
        self.data_dir = Path(data_dir)
        self.model_type = model_type
        self.problem_formulation = problem_formulation
        self.models_dir = Path(models_dir)
        self.thermo_dim = thermo_dim  # NEW

        # Model parameters
        self.wind_dim = 8
        self.nb_altitudes = 1  # Changed from 5 - altitude binning removed

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

        # NEW: Validate dataset integrity BEFORE loading any data
        expected_cols = len(self.dataset.meteo_params)
        actual_cols = self.dataset.meteo_content.shape[1]
        if actual_cols != expected_cols:
            import sys
            if self.thermo_dim > 0 and actual_cols < expected_cols:
                print("\n" + "=" * 70)
                print("ERROR: Dataset out of sync - rebuild required!")
                print("=" * 70)
                print(f"\nDataset metadata expects {expected_cols} parameters")
                print(f"Dataset data has {actual_cols} parameters")
                print("\nThe dataset was partially updated but not fully rebuilt.")
                print("You need to rebuild the dataset to include thermo parameters:\n")
                print("  pyparaglide build-dataset --rebuild-cache")
                print("\n" + "=" * 70)
                sys.exit(1)

        # Load weather data for CELLS: meteo data is organized as (nb_cells * nb_days, dim)
        X_other = [self.dataset.get_meteo_matrix(cells, self.dataset.params_other[h]) for h in range(3)]
        X_wind = [convert_wind_matrix(self.dataset.get_meteo_matrix(cells, self.dataset.params_wind[h]), self.wind_dim) for h in range(3)]
        X_humidity = [self.dataset.get_meteo_matrix(cells, self.dataset.params_humidity[h]) for h in range(3)]

        # Load thermo data if enabled (NEW)
        X_thermo = None
        if self.thermo_dim > 0:
            # Now safe to access thermo parameters
            X_thermo = [self.dataset.get_meteo_matrix(cells, self.dataset.params_thermo[h]) for h in range(3)]

        # Compute normalization (based on hour 1 = 12h)
        print("[INFO] Computing normalization from data")
        norm_mean_other, norm_std_other = compute_normalization_coeffs(X_other[1])
        norm_mean_hum, norm_std_hum = compute_normalization_coeffs(X_humidity[1])

        # Compute thermo normalization if enabled (NEW)
        norm_mean_thermo, norm_std_thermo = None, None
        if X_thermo is not None:
            norm_mean_thermo, norm_std_thermo = compute_normalization_coeffs(X_thermo[1])

        self.normalization = Normalization(
            other_mean=norm_mean_other,
            other_std=norm_std_other,
            humidity_mean=norm_mean_hum,
            humidity_std=norm_std_hum,
            thermo_mean=norm_mean_thermo,  # NEW
            thermo_std=norm_std_thermo,    # NEW
        )

        # Apply normalization
        for h in range(3):
            X_other[h] = apply_normalization(X_other[h], norm_mean_other, norm_std_other)
            X_humidity[h] = apply_normalization(X_humidity[h], norm_mean_hum, norm_std_hum)
            if X_thermo is not None:  # NEW
                X_thermo[h] = apply_normalization(X_thermo[h], norm_mean_thermo, norm_std_thermo)

        # Prepare inputs
        X = self._prepare_inputs(cells, X_other, X_wind, X_humidity, X_thermo, super_resolution)  # NEW: added X_thermo
        Y = self._prepare_outputs(cells, super_resolution)

        return X, Y

    def _prepare_inputs(
        self,
        cells: list[int],
        X_other: list[np.ndarray],
        X_wind: list[np.ndarray],
        X_humidity: list[np.ndarray],
        X_thermo: list[np.ndarray] | None,  # NEW
        super_resolution: int,
    ) -> list:
        """Prepare input tensors for CELLS model."""
        nb_cells_model = len(cells)

        # Date (nb_days,)
        X_date = self.dataset.get_date()

        # Day of week (nb_days, 7)
        X_dow = self.dataset.get_dow()

        # Mountainess (nb_days, nb_cells, 1) - averaged over altitudes
        mountainess = self.dataset.get_mountainess(cells, self.nb_altitudes)  # Shape: (nb_cells, 1)
        X_mountainess = np.repeat(mountainess[np.newaxis, :, :], self.nb_days, axis=0)  # (nb_days, nb_cells, 1)

        # Initialize stacked arrays
        dim_other = X_other[0].shape[1]
        X_other_stacked = np.zeros((self.nb_days, nb_cells_model, 3, dim_other), dtype=np.float32)

        dim_humidity = X_humidity[0].shape[1]
        X_humidity_stacked = np.zeros((self.nb_days, nb_cells_model, 3, dim_humidity), dtype=np.float32)

        X_wind_stacked = np.zeros((self.nb_days, nb_cells_model, 1, 3, self.wind_dim), dtype=np.float32)

        # Initialize thermo stacked array (ALWAYS create, even for baseline with thermo_dim=0)
        # This ensures the model always receives 7 inputs matching its expected input signature
        dim_thermo = self.thermo_dim  # 0 for baseline, 4 for thermo
        X_thermo_stacked = np.zeros((self.nb_days, nb_cells_model, 3, dim_thermo), dtype=np.float32)

        # Fill with actual thermo data if available
        if X_thermo is not None:
            for h in range(3):
                for i, cell in enumerate(cells):
                    start_idx = i * self.nb_days
                    end_idx = start_idx + self.nb_days
                    X_thermo_stacked[:, i, h, :] = X_thermo[h][start_idx:end_idx, :]

        for h in range(3):
            for i, cell in enumerate(cells):
                # CELLS: data loaded for requested `cells` only -> index i
                start_idx = i * self.nb_days
                end_idx = start_idx + self.nb_days

                # Safety check for data availability
                if end_idx > X_other[h].shape[0]:
                    raise ValueError(f"Not enough weather data for cell {cell}. Expected range {start_idx}:{end_idx}, but data has only {X_other[h].shape[0]} rows.")

                # Other weather
                X_other_stacked[:, i, h, :] = X_other[h][start_idx : end_idx, :]

                # Humidity
                X_humidity_stacked[:, i, h, :] = X_humidity[h][start_idx : end_idx, :]

                # Wind
                # Extract wind for this cell (nb_days, alt*dim) - average over altitudes
                wind_cell = X_wind[h][start_idx : end_idx, :]
                # Reshape to (nb_days, 1, dim) - average over 5 altitudes to single value
                wind_reshaped = wind_cell.reshape(self.nb_days, 5, self.wind_dim).mean(axis=1, keepdims=True)
                # Assign to stacked array
                X_wind_stacked[:, i, 0, h, :] = wind_reshaped[:, 0, :]

        # Build input list (ALWAYS include thermo - even when dim_thermo=0)
        # Model expects 7 inputs matching its signature
        inputs = [X_date, X_dow, X_mountainess, X_other_stacked, X_humidity_stacked, X_wind_stacked, X_thermo_stacked]

        return inputs

    def _prepare_outputs(self, cells: list[int], super_resolution: int) -> list:
        """Prepare output tensors for CELLS model."""
        # Get data with shape (len(cells) * super_resolution^2 * nb_days, 1)
        outputs = self.dataset.get_flights_by_altitude(
            cells, self.nb_altitudes, super_resolution, self.problem_formulation == ProblemFormulation.REGRESSION
        )
        # Reshape each output to (nb_days, len(cells) * super_resolution^2, 1)
        # Use F-order because output is organized as [cell0_day0, cell0_day1, ..., cell1_day0, ...]
        return [
            out.reshape((self.nb_days, len(cells) * super_resolution * super_resolution, 1), order='F')
            for out in outputs
        ]

    def create_model(
        self,
        cells: list[int] | None = None,
        super_resolution: int = 1,
        load_weights: bool = False,
        weight_suffix: str = "",
    ) -> None:
        """
        Create and optionally load model weights.

        Args:
            cells: List of cell indices (None = all cells)
            super_resolution: Super-resolution factor
            load_weights: Whether to load existing weights
            weight_suffix: Optional suffix for weight file
        """
        tf.keras.backend.clear_session()

        # Use all cells if not specified
        if cells is None:
            cells = list(range(self.nb_cells))

        self.model = ModelCells.create_model(
            problem_formulation=self.problem_formulation,
            nb_cells=len(cells),
            wind_dim=self.wind_dim,
            other_dim=45,  # Will be updated from data
            humidity_dim=2,
            nb_altitudes=self.nb_altitudes,
            thermo_dim=self.thermo_dim,  # NEW
            super_resolution=super_resolution,
        )

        if load_weights:
            self._load_weights(suffix=weight_suffix)

    def _load_weights(self, suffix: str = "") -> None:
        """Load model weights from directory."""
        weight_path = self.models_dir / f"cells{suffix}.weights.h5"
        if weight_path.exists():
            self.model.load_weights(weight_path)
            print(f"[INFO] Loaded weights from {weight_path}")

    def save_weights(self, suffix: str = "") -> Path:
        """
        Save model weights to directory.

        Args:
            suffix: Optional suffix for filename (e.g., "_cell_0")

        Returns:
            Path where weights were saved
        """
        self.models_dir.mkdir(parents=True, exist_ok=True)

        # Save normalization
        if self.normalization:
            norm_path = self.models_dir / f"normalization_cells{suffix}.pkl"
            self.normalization.save(norm_path)

        # Save model weights
        weight_path = self.models_dir / f"cells{suffix}.weights.h5"
        self.model.save_weights(weight_path)
        print(f"[INFO] Saved weights to {weight_path}")
        return weight_path

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
        early_stopping_patience: int = 0,
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
        # If validation_split > 0, use Keras random split (by samples)
        # Otherwise, use alternating days (by days) for time-series consistency
        use_keras_val_split = validation_split > 0.0

        if use_keras_val_split:
            # Use Keras validation_split: random split by samples
            # Train on all data, Keras will handle the split internally
            train_indices = np.arange(self.nb_days)
            val_indices = np.array([])
            keras_validation_split = validation_split
            print(f"[INFO] Using Keras validation_split={validation_split} (random by samples)")
        elif use_validation_set:
            # Use alternating days for validation (time-series consistency)
            val_indices = np.arange(0, self.nb_days, 2)
            train_indices = np.arange(1, self.nb_days, 2)
            keras_validation_split = 0.0
            print(f"[INFO] Using alternating days for validation (days {val_indices[:5].tolist()}... for val)")
        else:
            # No validation
            val_indices = np.array([])
            train_indices = np.arange(self.nb_days)
            keras_validation_split = 0.0

        # Prepare callbacks
        log_file = self.models_dir / "cells.log"
        callbacks = [
            LearningRateScheduler.create(lr_init=lr_init, lr_end=lr_end, nb_epochs=nb_epochs),
            TrainingLogger(ModelType.CELLS, log_file),
        ]

        # Add EarlyStopping if validation is enabled
        # Works with both alternating days (val_indices) and Keras validation_split
        has_validation = len(val_indices) > 0 or use_keras_val_split
        if early_stopping_patience > 0 and has_validation:
            from pyparaglide.training.early_stopping import EarlyStopping
            callbacks.append(
                EarlyStopping(
                    monitor="val_loss",
                    patience=early_stopping_patience,
                    min_delta=0.001,
                    restore_best_weights=True,
                    verbose=1,
                )
            )
            print(f"[INFO] EarlyStopping enabled: patience={early_stopping_patience}")

        # Prepare data
        if use_keras_val_split:
            # For Keras validation_split, pass all data
            # Keras will randomly split by samples
            X_train = X
            Y_train = Y
            val_data = None
        else:
            # For alternating days or no validation, manually split
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
            validation_split=keras_validation_split if use_keras_val_split else None,
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
