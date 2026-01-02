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

        # Mountainess (nb_days, nb_cells, nb_altitudes) - only used for CELLS
        if self.model_type == ModelType.CELLS:
            mountainess = self.dataset.get_mountainess(cells, self.nb_altitudes)
            X_mountainess = np.repeat(mountainess[np.newaxis, :, :], self.nb_days, axis=0)

        # Initialize stacked arrays
        dim_other = X_other[0].shape[1]
        X_other_stacked = np.zeros((self.nb_days, nb_cells_model, 3, dim_other), dtype=np.float32)
        
        dim_humidity = X_humidity[0].shape[1]
        X_humidity_stacked = np.zeros((self.nb_days, nb_cells_model, 3, dim_humidity), dtype=np.float32)
        
        X_wind_stacked = np.zeros((self.nb_days, nb_cells_model, self.nb_altitudes, 3, self.wind_dim), dtype=np.float32)

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
                # Extract wind for this cell (nb_days, alt*dim)
                wind_cell = X_wind[h][start_idx : end_idx, :]
                # Reshape to (nb_days, alt, dim)
                wind_reshaped = wind_cell.reshape(self.nb_days, self.nb_altitudes, self.wind_dim)
                # Assign to stacked array
                X_wind_stacked[:, i, :, h, :] = wind_reshaped

        # For SPOTS single-cell training: squeeze cell dimension to match model input shapes
        if self.model_type == ModelType.SPOTS:
            if nb_cells_model == 1:
                X_other_stacked = X_other_stacked[:, 0, :, :]  # (nb_days, 1, 3, dim) -> (nb_days, 3, dim)
                X_humidity_stacked = X_humidity_stacked[:, 0, :, :]  # (nb_days, 1, 3, dim) -> (nb_days, 3, dim)
                X_wind_stacked = X_wind_stacked[:, 0, :, :, :]  # (nb_days, 1, nb_altitudes, 3, wind_dim) -> (nb_days, nb_altitudes, 3, wind_dim)
            # Note: SPOTS model doesn't use mountainess, so don't include it
            return [X_date, X_dow, X_other_stacked, X_humidity_stacked, X_wind_stacked]

        return [X_date, X_dow, X_mountainess, X_other_stacked, X_humidity_stacked, X_wind_stacked]

    def _prepare_outputs(self, cells: list[int], super_resolution: int) -> list:
        """Prepare output tensors for model."""
        if self.model_type == ModelType.CELLS:
            # Get data with shape (len(cells) * super_resolution^2 * nb_days, nb_altitudes)
            outputs = self.dataset.get_flights_by_altitude(
                cells, self.nb_altitudes, super_resolution, self.problem_formulation == ProblemFormulation.REGRESSION
            )
            # Reshape each output to (nb_days, len(cells) * super_resolution^2, nb_altitudes)
            return [
                out.reshape((self.nb_days, len(cells) * super_resolution * super_resolution, self.nb_altitudes))
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

            self.model = ModelSpots.create_model(
                problem_formulation=self.problem_formulation,
                cells_data=cells_data,
                wind_dim=self.wind_dim,
                other_dim=45,
                humidity_dim=2,
                nb_altitudes=self.nb_altitudes,
                initialization={"date_factor": np.array([[1.275]], dtype=np.float32), "dow_factor": np.array([[1.0] * 7], dtype=np.float32).T},
            )

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

    def load_weights_from_cells(
        self,
        cells_weight_path: Path,
        freeze_transferred: bool = True,
    ) -> None:
        """
        Load shared weights from trained CELLS model into SPOTS model.

        Transfers:
        - flyability_block (entire submodel)
        - date_factor (DateBlock)
        - dow_factor (DayOfWeekBlock)

        Args:
            cells_weight_path: Path to CELLS.weights.h5
            freeze_transferred: Whether to freeze transferred layers

        Raises:
            ValueError: If model_type is not SPOTS or model not created
        """
        if self.model_type != ModelType.SPOTS:
            raise ValueError("Weight transfer only applies to SPOTS model")

        if self.model is None:
            raise ValueError("Model must be created before loading weights")

        print(f"[INFO] Loading weights from CELLS model: {cells_weight_path}")

        # Determine nb_cells from the weights file (PopulationBlock shape)
        # PopulationBlock kernel shape = (nb_cells * super_resolution^2, nb_altitudes)
        import h5py
        with h5py.File(cells_weight_path, 'r') as f:
            # Find any population_block layer and read its kernel shape
            popu_key = None
            for key in f['layers'].keys():
                if 'population_block' in key and 'vars' in f[f'layers/{key}']:
                    popu_key = f'layers/{key}/vars/0'
                    break
            if popu_key is None:
                raise ValueError("Could not find PopulationBlock in weights file")
            popu_shape = f[popu_key].shape
            # popu_shape = (nb_cells * super_resolution^2, nb_altitudes)
            # Assuming super_resolution=1 for CELLS model
            nb_cells_from_weights = popu_shape[0]
            print(f"[INFO] Detected {nb_cells_from_weights} cells from weights file")

        # Create temporary CELLS model to load weights
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

        # Transfer weights for shared layers
        shared_layer_names = [
            "flyability_block",
            "date_factor",
            "dow_factor",
        ]

        for layer_name in shared_layer_names:
            try:
                if layer_name == "flyability_block":
                    cells_layer = temp_trainer.model.get_layer(layer_name)
                    spots_layer = self.model.get_layer(layer_name)
                    spots_layer.set_weights(cells_layer.get_weights())
                    print(f"[INFO] Transferred weights for layer: {layer_name}")
                elif layer_name in ["date_factor", "dow_factor"]:
                    # These are inside PopulationBlock in both models
                    # Extract from first population block in CELLS model
                    cells_pop_block = temp_trainer.model.get_layer("population_block_flown")
                    
                    # Find corresponding layer in SPOTS model (it's named population__cell_{id})
                    # For weight transfer during creation, we usually have only one cell in the list
                    spots_pop_layer = None
                    for layer in self.model.layers:
                        if layer.name.startswith("population__cell_"):
                            spots_pop_layer = layer
                            break
                    
                    if spots_pop_layer and hasattr(cells_pop_block, layer_name) and hasattr(spots_pop_layer, layer_name):
                        val = getattr(cells_pop_block, layer_name)
                        # If it's a variable, get value
                        if hasattr(val, "numpy"):
                            val = val.numpy()
                        
                        # Set in SPOTS layer (it might be a constant or variable)
                        # Since we can't easily set_weights on individual attributes if they are not tracked as weights,
                        # we check if they are in trainable_weights
                        target_attr = getattr(spots_pop_layer, layer_name)
                        if hasattr(target_attr, "assign"):
                            target_attr.assign(val)
                        else:
                            setattr(spots_pop_layer, layer_name, tf.constant(val))
                        
                        print(f"[INFO] Transferred {layer_name} from CELLS population_block to SPOTS {spots_pop_layer.name}")

                # Optionally freeze
                if freeze_transferred:
                    try:
                        layer_to_freeze = self.model.get_layer(layer_name)
                        layer_to_freeze.trainable = False
                        print(f"[INFO] Frozen layer: {layer_name}")
                    except ValueError:
                        # date_factor/dow_factor are handled inside population block
                        pass
            except Exception as e:
                print(f"[WARNING] Could not transfer {layer_name}: {e}")

        print("[INFO] Weight transfer complete")

        # Recompile model after freezing layers
        if freeze_transferred:
            self.model.compile(
                optimizer="adam",
                loss=(
                    "binary_crossentropy"
                    if self.problem_formulation == ProblemFormulation.CLASSIFICATION
                    else "mse"
                ),
            )

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
