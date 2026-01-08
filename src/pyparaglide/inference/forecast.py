"""
Forecaster for PyParaglide.

Generates paragliding flyability forecasts using trained CELLS models and GFS weather data.
"""

import datetime as dt
import json
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from pyparaglide.config import get_settings
from pyparaglide.data.normalization import Normalization
from pyparaglide.inference.grib_reader import GribReader
from pyparaglide.models import ModelCells, ProblemFormulation


class Forecaster:
    """
    Main forecaster for paragliding flyability prediction.

    Loads trained CELLS models and generates grid-based forecasts.
    """

    def __init__(
        self,
        models_dir: Path | str,
        problem_formulation: ProblemFormulation = ProblemFormulation.CLASSIFICATION,
    ):
        """
        Initialize forecaster.

        Args:
            models_dir: Directory containing trained model weights
            problem_formulation: CLASSIFICATION or REGRESSION
        """
        self.models_dir = Path(models_dir)
        self.problem_formulation = problem_formulation

        # Model parameters
        self.wind_dim = 8
        self.nb_altitudes = 1  # Changed from 5 - altitude binning removed
        self.nb_cells = 97  # Alps region default

        # Model and normalization (loaded later)
        self.model: tf.keras.Model | None = None
        self.normalization: Normalization | None = None

        # Mountainess data (loaded from models_dir)
        self.mountainess_data: np.ndarray | None = None

    def _detect_nb_cells_from_weights(self) -> int:
        """
        Detect nb_cells from the weights file by reading PopulationBlock kernel shape.

        Returns:
            Number of cells detected from weights
        """
        # Always use default cells.weights.h5 (no suffix)
        weight_path = self.models_dir / "cells.weights.h5"

        import h5py

        with h5py.File(weight_path, "r") as f:
            # Find PopulationBlock layer and read its kernel shape
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
                    f"Could not find PopulationBlock kernel in weights file: {weight_path}"
                )

            # kernel_shape = (nb_cells * super_resolution^2, nb_altitudes)
            nb_cells_from_weights = kernel_shape[0]
            print(
                f"[INFO] Detected {nb_cells_from_weights} cells from weights file (kernel shape: {kernel_shape})"
            )
            return nb_cells_from_weights

    def load_model(self) -> None:
        """Load trained CELLS model and normalization coefficients."""
        # Always use default files (no suffix)
        norm_path = self.models_dir / "normalization_cells.pkl"

        if norm_path.exists():
            self.normalization = Normalization.load(norm_path)
        else:
            raise FileNotFoundError(f"Normalization file not found: {norm_path}")

        # Detect nb_cells from weights file before creating model
        self.nb_cells = self._detect_nb_cells_from_weights()

        # Detect thermo_dim from normalization (auto-detected)
        thermo_dim = 4 if self.normalization.thermo_mean is not None else 0

        # Create and load CELLS model
        self.model = ModelCells.create_model(
            problem_formulation=self.problem_formulation,
            nb_cells=self.nb_cells,
            wind_dim=self.wind_dim,
            other_dim=self.normalization.other_mean.shape[0],
            humidity_dim=self.normalization.humidity_mean.shape[0],
            nb_altitudes=self.nb_altitudes,
            thermo_dim=thermo_dim,
            super_resolution=1,
        )

        # Load weights (no suffix)
        weight_path = self.models_dir / "cells.weights.h5"
        if weight_path.exists():
            self.model.load_weights(weight_path)
            print(f"[INFO] Loaded model from {weight_path}")
        else:
            raise FileNotFoundError(f"Model weights not found: {weight_path}")

        # Load mountainess data
        self._load_mountainess_data()

    def predict_day(
        self,
        grib_files: list[Path | str],
        target_date: dt.date,
        bbox: tuple[float, float, float, float] | None = None,
    ) -> dict[str, Any]:
        """
        Generate forecast for a single day.

        Args:
            grib_files: List of GRIB files (6h, 12h, 18h forecasts)
            target_date: Target date for forecast
            bbox: Bounding box following GeoJSON RFC 7946 (lon_min, lat_min, lon_max, lat_max)

        Returns:
            Dictionary with forecast results
        """
        if self.model is None:
            self.load_model()

        settings = get_settings()
        if bbox is None:
            bbox = settings.parse_bbox()

        # Read GRIB files and extract weather data
        readers = [GribReader(f) for f in grib_files]

        # Prepare input data
        X = self._prepare_inputs(readers, target_date, bbox)

        # Run prediction
        predictions = self.model.predict(X, verbose=0)

        # Format results
        results = self._format_results(predictions, target_date, bbox)

        return results

    def _prepare_inputs(
        self, readers: list[GribReader], target_date: dt.date, bbox: tuple[float, float, float, float]
    ) -> list[np.ndarray]:
        """Prepare input tensors for model prediction."""
        # Date inputs
        day_of_year = target_date.timetuple().tm_yday / 365.0  # Normalized 0-1
        X_date = np.array([[day_of_year]], dtype=np.float32)

        # Day of week
        dow = np.zeros((1, 7), dtype=np.float32)
        dow[0, target_date.weekday()] = 1.0
        X_dow = dow

        # Weather data from GRIB files
        # This is simplified - full implementation needs proper grid extraction
        X_other, X_wind, X_humidity, X_thermo = self._extract_weather_data(readers, bbox)

        # Mountainess data from elevation analysis
        X_mountainess = self._get_mountainess_inputs()

        # Combine all inputs (ALWAYS include thermo, even if empty for baseline models)
        # Model always expects 7 inputs matching its signature
        inputs = [X_date, X_dow, X_mountainess, X_other, X_humidity, X_wind, X_thermo]

        return inputs

    def _extract_weather_data(
        self, readers: list[GribReader], bbox: tuple[float, float, float, float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract weather data from GRIB files.

        Returns:
            (X_other, X_wind, X_humidity, X_thermo) tuple
            X_thermo has shape (1, nb_cells, 3, thermo_dim) where thermo_dim=0 for baseline models
        """
        nb_hours = 3
        nb_cells = self.nb_cells

        # Check if model expects thermo data and determine thermo_dim
        thermo_dim = 4 if self.normalization.thermo_mean is not None else 0

        # Extract for each forecast hour
        X_other = np.zeros((1, nb_cells, nb_hours, 45), dtype=np.float32)  # 5 params × 9 levels
        X_humidity = np.zeros((1, nb_cells, nb_hours, 2), dtype=np.float32)  # PWAT, CWAT
        X_wind = np.zeros((1, nb_cells, 1, nb_hours, self.wind_dim), dtype=np.float32)  # nb_altitudes=1
        # Always create thermo array (empty for baseline models with thermo_dim=0)
        X_thermo = np.zeros((1, nb_cells, nb_hours, thermo_dim), dtype=np.float32)

        for h, reader in enumerate(readers):
            # This is simplified - proper implementation needs:
            # 1. Grid cell center coordinates
            # 2. Proper interpolation to cell centers
            # 3. Normalization using loaded coefficients

            # Placeholder: extract bbox mean values
            for cell_idx in range(nb_cells):
                # Extract "other" parameters
                for param_idx, (name, _, levels) in enumerate(GribReader.PARAMS_OTHER):
                    for level_idx, level in enumerate(levels):
                        data = reader.get_bbox_data(name, bbox, level)
                        if data is not None:
                            X_other[0, cell_idx, h, param_idx * 9 + level_idx] = np.mean(data)

                # Extract humidity parameters
                for param_idx, (name, _, _) in enumerate(GribReader.PARAMS_HUMIDITY):
                    data = reader.get_bbox_data(name, bbox, None)
                    if data is not None:
                        X_humidity[0, cell_idx, h, param_idx] = np.mean(data)

                # Extract thermo parameters (NEW) - 4 params (PBLH not available)
                if thermo_dim > 0:
                    thermo_params = [
                        ("Total Cloud Cover", "atmosphere"),
                        ("Convective available potential energy", "surface"),
                        ("Surface lifted index", "surface"),
                        ("Convective inhibition", "surface"),
                    ]
                    for param_idx, (name, level_type) in enumerate(thermo_params):
                        data = reader.get_bbox_data(name, bbox, level_type)
                        if data is not None:
                            X_thermo[0, cell_idx, h, param_idx] = np.mean(data)

                # Extract wind and convert to direction bins (averaged over 5 altitudes)
                wind_dirs_avg = np.zeros(self.wind_dim, dtype=np.float32)
                for level in [1000, 900, 800, 700, 600]:
                    u = reader.get_bbox_data("u", bbox, level)
                    v = reader.get_bbox_data("v", bbox, level)
                    if u is not None and v is not None:
                        u_mean = np.mean(u)
                        v_mean = np.mean(v)
                        wind_dirs = GribReader.wind_to_directions(
                            np.array([[u_mean]]), np.array([[v_mean]]), self.wind_dim
                        )
                        wind_dirs_avg += wind_dirs[0, 0, :]

                # Average over 5 altitudes
                X_wind[0, cell_idx, 0, h, :] = wind_dirs_avg / 5.0

        # Apply normalization (if loaded)
        if self.normalization is not None:
            for h in range(nb_hours):
                # Normalize other parameters
                other_flat = X_other[0, :, h, :].reshape(-1, X_other.shape[-1])
                # Apply normalization per feature
                for i in range(other_flat.shape[1]):
                    if i < len(self.normalization.other_mean):
                        other_flat[:, i] = (other_flat[:, i] - self.normalization.other_mean[i]) / self.normalization.other_std[i]
                X_other[0, :, h, :] = other_flat.reshape(nb_cells, -1)

                # Normalize humidity
                hum_flat = X_humidity[0, :, h, :].reshape(-1, X_humidity.shape[-1])
                for i in range(hum_flat.shape[1]):
                    if i < len(self.normalization.humidity_mean):
                        hum_flat[:, i] = (hum_flat[:, i] - self.normalization.humidity_mean[i]) / self.normalization.humidity_std[i]
                X_humidity[0, :, h, :] = hum_flat.reshape(nb_cells, -1)

                # Normalize thermo (NEW) - only if thermo_dim > 0 and normalization exists
                if X_thermo.shape[-1] > 0 and self.normalization.thermo_mean is not None:
                    thermo_flat = X_thermo[0, :, h, :].reshape(-1, X_thermo.shape[-1])
                    for i in range(thermo_flat.shape[1]):
                        if i < len(self.normalization.thermo_mean):
                            thermo_flat[:, i] = (thermo_flat[:, i] - self.normalization.thermo_mean[i]) / self.normalization.thermo_std[i]
                    X_thermo[0, :, h, :] = thermo_flat.reshape(nb_cells, -1)

        return X_other, X_wind, X_humidity, X_thermo  # NEW: added X_thermo

    def _format_results(self, predictions: list[np.ndarray], target_date: dt.date, bbox: tuple[float, float, float, float]) -> dict[str, Any]:
        """Format CELLS prediction results into output dictionary."""
        results = {
            "date": target_date.isoformat(),
            "bbox": {"lon_min": bbox[0], "lat_min": bbox[1], "lon_max": bbox[2], "lat_max": bbox[3]},
            "model_type": "CELLS",
            "predictions": [],
        }

        # 2 outputs: flown, crossed
        flown, crossed = predictions

        for cell_idx in range(self.nb_cells):
            # Single aggregated prediction per cell (altitude binning removed)
            cell_result = {
                "cell_id": cell_idx,
                "flyability": float(flown[0, cell_idx, 0]),
                "crossability": float(crossed[0, cell_idx, 0]),
            }

            results["predictions"].append(cell_result)

        return results

    def _load_mountainess_data(self) -> None:
        """
        Load mountainess data from mountainess_by_cell_alt.pkl file.

        This file contains terrain mountainess values for each grid cell,
        computed from elevation data during model training.
        """
        settings = get_settings()

        # Try pkl_dir first (where build-dataset saves it), then models_dir for backwards compatibility
        for base_dir, desc in [(Path(settings.pkl_dir), "PKL directory"), (self.models_dir, "models directory")]:
            mountainess_path = base_dir / "mountainess_by_cell_alt.pkl"

            if mountainess_path.exists():
                try:
                    with open(mountainess_path, "rb") as f:
                        self.mountainess_data = pickle.load(f)
                    print(f"[INFO] Loaded mountainess data from {mountainess_path}")

                    # Validate shape
                    if self.mountainess_data is not None and self.mountainess_data.shape[0] != self.nb_cells:
                        print(f"[WARNING] Mountainess data has {self.mountainess_data.shape[0]} cells, "
                              f"but model expects {self.nb_cells}. Using first {self.nb_cells} cells.")
                        self.mountainess_data = self.mountainess_data[:self.nb_cells]

                    return  # Success, exit early

                except Exception as e:
                    print(f"[WARNING] Failed to load mountainess data from {mountainess_path}: {e}")

        # If we get here, no file was found or loaded successfully
        print("[INFO] Using default mountainess values (zeros)")
        self.mountainess_data = None
    
    def _get_mountainess_inputs(self) -> np.ndarray:
        """
        Get mountainess input tensor for model prediction.
        
        Returns:
            Array of shape (1, nb_cells, 1) with mountainess values
        """
        nb_cells = self.nb_cells
        
        if self.mountainess_data is not None:
            # Use loaded mountainess data
            # Average over 5 altitude levels (altitude binning removed)
            cell_avg = np.mean(self.mountainess_data, axis=1)  # Shape: (nb_cells,)
            return cell_avg.reshape(1, nb_cells, 1).astype(np.float32)
        else:
            # Fallback to zeros if data not available
            print("[DEBUG] Using zero mountainess values")
            return np.zeros((1, nb_cells, 1), dtype=np.float32)

    def save_forecast(self, results: dict[str, Any], output_path: Path | str) -> None:
        """
        Save forecast results to JSON file.

        Args:
            results: Forecast results dictionary
            output_path: Output file path
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"[INFO] Saved forecast to {output_path}")
