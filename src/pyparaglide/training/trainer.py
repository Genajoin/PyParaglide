"""
Model trainer for PyParaglide.

Handles training of both CELLS and SPOTS models.
"""

from pathlib import Path
from typing import Literal

import numpy as np
import tensorflow as tf

from pyparaglide.data import Dataset, Normalization
from pyparaglide.data.dataset import convert_wind_matrix
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

        # Load weather data
        # For SPOTS: meteo data is not organized by cells, use all data (nb_days, dim)
        # For CELLS: meteo data is organized as (nb_cells * nb_days, dim)
        if self.model_type == ModelType.SPOTS:
            # Use range(nb_cells) to get all data, then we'll filter in _prepare_inputs
            X_other = [self.dataset.get_meteo_matrix(list(range(self.nb_cells)), self.dataset.params_other[h]) for h in range(3)]
            X_wind = [convert_wind_matrix(self.dataset.get_meteo_matrix(list(range(self.nb_cells)), self.dataset.params_wind[h]), self.wind_dim) for h in range(3)]
            X_humidity = [self.dataset.get_meteo_matrix(list(range(self.nb_cells)), self.dataset.params_humidity[h]) for h in range(3)]
        else:
            X_other = [self.dataset.get_meteo_matrix(cells, self.dataset.params_other[h]) for h in range(3)]
            X_wind = [convert_wind_matrix(self.dataset.get_meteo_matrix(cells, self.dataset.params_wind[h]), self.wind_dim) for h in range(3)]
            X_humidity = [self.dataset.get_meteo_matrix(cells, self.dataset.params_humidity[h]) for h in range(3)]

        # Compute or load normalization
        # For SPOTS, we MUST use the normalization from the CELLS model
        norm_path = self.models_dir / "normalization_cells.pkl"
        if self.model_type == ModelType.SPOTS and norm_path.exists():
            print(f"[INFO] Loading normalization from {norm_path}")
            self.normalization = Normalization.load(norm_path)
            norm_mean_other = self.normalization.other_mean
            norm_std_other = self.normalization.other_std
            norm_mean_hum = self.normalization.humidity_mean
            norm_std_hum = self.normalization.humidity_std
        else:
            # Compute normalization (based on hour 1 = 12h)
            print("[INFO] Computing normalization from data")
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

        # Mountainess (nb_days, nb_cells, 1) - only used for CELLS (averaged over altitudes)
        if self.model_type == ModelType.CELLS:
            mountainess = self.dataset.get_mountainess(cells, self.nb_altitudes)  # Shape: (nb_cells, 1)
            X_mountainess = np.repeat(mountainess[np.newaxis, :, :], self.nb_days, axis=0)  # (nb_days, nb_cells, 1)

        # Initialize stacked arrays
        dim_other = X_other[0].shape[1]
        X_other_stacked = np.zeros((self.nb_days, nb_cells_model, 3, dim_other), dtype=np.float32)
        
        dim_humidity = X_humidity[0].shape[1]
        X_humidity_stacked = np.zeros((self.nb_days, nb_cells_model, 3, dim_humidity), dtype=np.float32)
        
        X_wind_stacked = np.zeros((self.nb_days, nb_cells_model, 1, 3, self.wind_dim), dtype=np.float32)

        for h in range(3):
            for i, cell in enumerate(cells):
                # Calculate start index based on how data was loaded
                # CELLS: data loaded for requested `cells` only -> index i
                # SPOTS: data loaded for ALL cells (range(nb_cells)) -> index cell
                if self.model_type == ModelType.CELLS:
                    start_idx = i * self.nb_days
                else:
                    start_idx = cell * self.nb_days
                
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

        # For SPOTS single-cell training: squeeze cell dimension to match model input shapes
        if self.model_type == ModelType.SPOTS:
            if nb_cells_model == 1:
                X_other_stacked = X_other_stacked[:, 0, :, :]  # (nb_days, 1, 3, dim) -> (nb_days, 3, dim)
                X_humidity_stacked = X_humidity_stacked[:, 0, :, :]  # (nb_days, 1, 3, dim) -> (nb_days, 3, dim)
                X_wind_stacked = X_wind_stacked[:, 0, :, :, :]  # (nb_days, 1, 1, 3, wind_dim) -> (nb_days, 1, 3, wind_dim)
            # Note: SPOTS model doesn't use mountainess, so don't include it
            return [X_date, X_dow, X_other_stacked, X_humidity_stacked, X_wind_stacked]

        return [X_date, X_dow, X_mountainess, X_other_stacked, X_humidity_stacked, X_wind_stacked]

    def _prepare_outputs(self, cells: list[int], super_resolution: int) -> list:
        """Prepare output tensors for model."""
        if self.model_type == ModelType.CELLS:
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
        else:
            # SPOTS model
            spots_data = self.dataset.get_flights_by_spots(cells)
            # Stack along axis=1 gives (nb_days, n_spots)
            return [np.stack([spots_data[c][s] for s in range(len(spots_data[c]))], axis=1, dtype=np.float32) for c in cells if len(spots_data[c]) > 0]

    def create_model(
        self,
        cells: list[int],
        super_resolution: int = 1,
        load_weights: bool = False,
        weight_suffix: str = "",
    ) -> None:
        """
        Create and optionally load model weights.

        Args:
            cells: List of cell indices
            super_resolution: Super-resolution factor
            load_weights: Whether to load existing weights
            weight_suffix: Optional suffix for weight file
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
            # SPOTS model - create cells data structure with real spots
            # For per-cell training, use sequential indices (0, 1, ...) instead of actual cell IDs
            spots_by_cell = self.dataset.get_spots()
            cells_data = {}
            for sequential_idx, cell in enumerate(cells):
                cell_spots = spots_by_cell.get(cell, [])
                if not cell_spots:
                    raise ValueError(f"No spots found for cell {cell}")
                # Use sequential index as key to match input tensor indexing
                cells_data[sequential_idx] = {"spots": cell_spots}

            # Load initialization factors from trained CELLS model
            initialization = self._load_initialization_from_cells()

            print(f"[DEBUG] Creating SPOTS model with initialization:")
            print(f"  date_factor = {initialization['date_factor'].flatten()}")
            print(f"  dow_factor = {initialization['dow_factor'].flatten()}")

            self.model = ModelSpots.create_model(
                problem_formulation=self.problem_formulation,
                cells_data=cells_data,
                wind_dim=self.wind_dim,
                other_dim=45,
                humidity_dim=2,
                nb_altitudes=self.nb_altitudes,
                initialization=initialization,
            )

            # Transfer FlyabilityBlock weights from CELLS model
            self._transfer_flyability_weights()

        if load_weights:
            self._load_weights(suffix=weight_suffix)

    def _load_weights(self, suffix: str = "") -> None:
        """Load model weights from directory."""
        # Try to load weights
        model_name = self.model_type.name.lower()
        weight_path = self.models_dir / f"{model_name}{suffix}.weights.h5"
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
            norm_path = self.models_dir / f"normalization_{self.model_type.name.lower()}{suffix}.pkl"
            self.normalization.save(norm_path)

        # Save model weights
        model_name = self.model_type.name.lower()
        weight_path = self.models_dir / f"{model_name}{suffix}.weights.h5"
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
        log_file = self.models_dir / f"{self.model_type.name.lower()}.log"
        callbacks = [
            LearningRateScheduler.create(lr_init=lr_init, lr_end=lr_end, nb_epochs=nb_epochs),
            TrainingLogger(self.model_type, log_file),
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

    def _load_initialization_from_cells(self) -> dict:
        """
        Load date_factor and dow_factor from trained CELLS model.

        This method extracts the learned date and day-of-week factors from
        a pre-trained CELLS model to initialize the SPOTS model with the
        correct values instead of hardcoded defaults.

        Returns:
            Dict with 'date_factor' and 'dow_factor' arrays

        Raises:
            FileNotFoundError: If CELLS weights file is not found
        """
        cells_weight_path = self.models_dir / "cells.weights.h5"

        if not cells_weight_path.exists():
            print("[WARNING] CELLS weights not found, using default initialization")
            print(f"[WARNING] Expected at: {cells_weight_path}")
            return {
                "date_factor": np.array([[1.275]], dtype=np.float32),
                "dow_factor": np.array([[1.0] * 7], dtype=np.float32).T,
            }

        print(f"[INFO] Loading initialization from CELLS model: {cells_weight_path}")

        # Determine nb_cells from the weights file
        import h5py

        with h5py.File(cells_weight_path, "r") as f:
            # Find PopulationBlock layer and read its kernel shape
            # NOTE: kernel is the largest variable (not var_0 which may be date_factor)
            kernel_shape = None
            for key in f["layers"].keys():
                if "population_block" in key and "vars" in f[f"layers/{key}"]:
                    # Find kernel variable (largest shape)
                    vars_list = list(f[f"layers/{key}/vars"].keys())
                    shapes = [f[f"layers/{key}/vars/{v}"].shape for v in vars_list]
                    # kernel has shape (nb_cells, nb_altitudes), find it by max size
                    kernel_idx = max(range(len(shapes)), key=lambda i: shapes[i][0] * shapes[i][1] if len(shapes[i]) == 2 else 0)
                    kernel_shape = shapes[kernel_idx]
                    break

            if kernel_shape is None:
                raise ValueError(
                    "Could not find PopulationBlock kernel in CELLS weights file"
                )

            # kernel_shape = (nb_cells * super_resolution^2, nb_altitudes)
            # Assuming super_resolution=1 for CELLS model
            nb_cells_from_weights = kernel_shape[0]
            print(
                f"[INFO] Detected {nb_cells_from_weights} cells from CELLS weights file (kernel shape: {kernel_shape})"
            )

        # Create temporary CELLS model to load weights
        from pyparaglide.models.enums import ModelType

        temp_trainer = Trainer(
            data_dir=self.data_dir,
            model_type=ModelType.CELLS,
            problem_formulation=self.problem_formulation,
            models_dir=self.models_dir,
        )

        # Create temp model with the same nb_cells as in the weights file
        cells_to_load = list(range(nb_cells_from_weights))
        temp_trainer.create_model(cells=cells_to_load)

        # Load weights - shapes will now match
        temp_trainer.model.load_weights(str(cells_weight_path))

        # Extract factors from PopulationBlock
        try:
            cells_pop = temp_trainer.model.get_layer("population_block")
        except ValueError:
            # Try alternative name (for multi-output models)
            cells_pop = temp_trainer.model.get_layer("population_block_flown")

        date_factor = cells_pop.var_date_factor.numpy()
        dow_factor = cells_pop.var_dow_factor.numpy()

        print(f"[INFO] Loaded initialization from CELLS model:")
        print(f"  date_factor: {date_factor.flatten()}")
        print(f"  dow_factor: {dow_factor.flatten()}")

        # Clear the temporary model from memory
        tf.keras.backend.clear_session()

        return {
            "date_factor": date_factor,
            "dow_factor": dow_factor,
        }

    def _transfer_flyability_weights(self) -> None:
        """
        Transfer FlyabilityBlock weights from CELLS to SPOTS model.

        This method loads the trained FlyabilityBlock weights from a CELLS model
        and applies them to the SPOTS model. This provides a better starting point
        than random initialization.

        Note: This method should be called after creating the SPOTS model but before
        training it.

        Raises:
            FileNotFoundError: If CELLS weights file is not found
        """
        cells_weight_path = self.models_dir / "cells.weights.h5"

        if not cells_weight_path.exists():
            print(
                "[WARNING] CELLS weights not found, FlyabilityBlock uses random initialization"
            )
            print(f"[WARNING] Expected at: {cells_weight_path}")
            return

        print(
            f"[INFO] Transferring FlyabilityBlock weights from: {cells_weight_path}"
        )

        # Reuse the code from _load_initialization_from_cells to avoid duplication
        import h5py

        with h5py.File(cells_weight_path, "r") as f:
            # Find PopulationBlock kernel to determine nb_cells
            # NOTE: Use kernel (largest variable), not var_0 which may be date_factor
            kernel_shape = None
            for key in f["layers"].keys():
                if "population_block" in key and "vars" in f[f"layers/{key}"]:
                    # Find kernel variable (largest shape)
                    vars_list = list(f[f"layers/{key}/vars"].keys())
                    shapes = [f[f"layers/{key}/vars/{v}"].shape for v in vars_list]
                    kernel_idx = max(range(len(shapes)), key=lambda i: shapes[i][0] * shapes[i][1] if len(shapes[i]) == 2 else 0)
                    kernel_shape = shapes[kernel_idx]
                    break

            if kernel_shape is None:
                print("[WARNING] Could not detect nb_cells from weights file")
                return

            nb_cells_from_weights = kernel_shape[0]

        # Create temporary CELLS model
        from pyparaglide.models.enums import ModelType

        temp_trainer = Trainer(
            data_dir=self.data_dir,
            model_type=ModelType.CELLS,
            problem_formulation=self.problem_formulation,
            models_dir=self.models_dir,
        )

        cells_to_load = list(range(nb_cells_from_weights))
        temp_trainer.create_model(cells=cells_to_load)
        temp_trainer.model.load_weights(str(cells_weight_path))

        # Transfer FlyabilityBlock weights
        try:
            cells_flyability = temp_trainer.model.get_layer("flyability_block")
            spots_flyability = self.model.get_layer("flyability_block")

            spots_flyability.set_weights(cells_flyability.get_weights())
            print("[INFO] Transferred FlyabilityBlock weights from CELLS")

        except ValueError as e:
            print(f"[WARNING] Could not transfer FlyabilityBlock weights: {e}")

        # Clear the temporary model from memory
        tf.keras.backend.clear_session()

    def prepare_data_for_cell(self, cell_index: int) -> tuple:
        """
        Prepare training data for a specific cell (SPOTS model).

        Args:
            cell_index: Index of the cell to prepare data for

        Returns:
            (X, Y) tuple where X is list of inputs, Y is list of outputs

        Raises:
            ValueError: If model_type is not SPOTS
        """
        if self.model_type != ModelType.SPOTS:
            raise ValueError("Per-cell data preparation only for SPOTS model")

        print(f"[INFO] Preparing data for cell {cell_index}")
        return self.prepare_data(cells=[cell_index])

    def get_spot_count_for_cell(self, cell_index: int) -> int:
        """
        Get number of spots in a specific cell.

        Args:
            cell_index: Index of the cell

        Returns:
            Number of spots in the cell

        Raises:
            ValueError: If cell has no spots or spots data not available
        """
        spots_by_cell = self.dataset.get_spots()
        if cell_index not in spots_by_cell:
            raise ValueError(f"No spots found for cell {cell_index}")
        return len(spots_by_cell[cell_index])
