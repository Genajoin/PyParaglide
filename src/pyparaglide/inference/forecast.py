"""
Forecaster for PyParaglide.

Generates paragliding flyability forecasts using trained models and GFS weather data.
"""

import datetime as dt
import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from pyparaglide.config import get_settings
from pyparaglide.data.normalization import Normalization
from pyparaglide.inference.grib_reader import GribReader
from pyparaglide.models import ModelCells, ModelSpots, ModelType, ProblemFormulation


class Forecaster:
    """
    Main forecaster for paragliding flyability prediction.

    Loads trained models and generates forecasts for spots or grid cells.
    """

    def __init__(
        self,
        models_dir: Path | str,
        model_type: ModelType = ModelType.CELLS,
        problem_formulation: ProblemFormulation = ProblemFormulation.CLASSIFICATION,
    ):
        """
        Initialize forecaster.

        Args:
            models_dir: Directory containing trained model weights
            model_type: CELLS or SPOTS
            problem_formulation: CLASSIFICATION or REGRESSION
        """
        self.models_dir = Path(models_dir)
        self.model_type = model_type
        self.problem_formulation = problem_formulation

        # Model parameters
        self.wind_dim = 8
        self.nb_altitudes = 5
        self.nb_cells = 97  # Alps region default

        # Model and normalization (loaded later)
        self.model: tf.keras.Model | None = None
        self.normalization: Normalization | None = None

    def load_model(self) -> None:
        """Load trained model and normalization coefficients."""
        # Load normalization
        norm_path = self.models_dir / f"normalization_{self.model_type.name.lower()}.pkl"
        if norm_path.exists():
            self.normalization = Normalization.load(norm_path)
        else:
            raise FileNotFoundError(f"Normalization file not found: {norm_path}")

        # Create and load model
        if self.model_type == ModelType.CELLS:
            self.model = ModelCells.create_model(
                problem_formulation=self.problem_formulation,
                nb_cells=self.nb_cells,
                wind_dim=self.wind_dim,
                other_dim=self.normalization.other_mean.shape[0],
                humidity_dim=self.normalization.humidity_mean.shape[0],
                nb_altitudes=self.nb_altitudes,
                super_resolution=1,
            )
        else:
            # SPOTS model requires cells data structure
            cells_data = {i: {"spots": list(range(5))} for i in range(self.nb_cells)}
            self.model = ModelSpots.create_model(
                problem_formulation=self.problem_formulation,
                cells_data=cells_data,
                wind_dim=self.wind_dim,
                other_dim=self.normalization.other_mean.shape[0],
                humidity_dim=self.normalization.humidity_mean.shape[0],
                nb_altitudes=self.nb_altitudes,
                initialization={"date_factor": np.array([[1.275]]), "dow_factor": np.array([[1.0] * 7])},
            )

        # Load weights
        weight_path = self.models_dir / f"{self.model_type.name.lower()}_weights.h5"
        if weight_path.exists():
            self.model.load_weights(weight_path)
            print(f"[INFO] Loaded model from {weight_path}")
        else:
            raise FileNotFoundError(f"Model weights not found: {weight_path}")

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
            bbox: Bounding box (lat_min, lat_max, lon_min, lon_max)

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
        X_other, X_wind, X_humidity = self._extract_weather_data(readers, bbox)

        # Mountainess (simplified - use elevation data if available)
        nb_cells = self.nb_cells
        X_mountainess = np.zeros((1, nb_cells, self.nb_altitudes), dtype=np.float32)

        # Combine all inputs
        return [X_date, X_dow, X_mountainess, X_other, X_humidity, X_wind]

    def _extract_weather_data(
        self, readers: list[GribReader], bbox: tuple[float, float, float, float]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract weather data from GRIB files.

        Returns:
            (X_other, X_wind, X_humidity) tuple
        """
        nb_hours = 3
        nb_cells = self.nb_cells

        # Extract for each forecast hour
        X_other = np.zeros((1, nb_cells, nb_hours, 45), dtype=np.float32)  # 5 params × 9 levels
        X_humidity = np.zeros((1, nb_cells, nb_hours, 2), dtype=np.float32)  # PWAT, CWAT
        X_wind = np.zeros((1, nb_cells, self.nb_altitudes, nb_hours, self.wind_dim), dtype=np.float32)

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

                # Extract wind and convert to direction bins
                for alt_idx, level in enumerate([1000, 900, 800, 700, 600]):
                    u = reader.get_bbox_data("u", bbox, level)
                    v = reader.get_bbox_data("v", bbox, level)
                    if u is not None and v is not None:
                        # Get mean wind for this cell
                        u_mean = np.mean(u)
                        v_mean = np.mean(v)

                        # Convert to direction bins
                        wind_dirs = GribReader.wind_to_directions(
                            np.array([[u_mean]]), np.array([[v_mean]]), self.wind_dim
                        )
                        X_wind[0, cell_idx, alt_idx, h, :] = wind_dirs[0, 0, :]

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

        return X_other, X_wind, X_humidity

    def _format_results(self, predictions: list[np.ndarray], target_date: dt.date, bbox: tuple[float, float, float, float]) -> dict[str, Any]:
        """Format prediction results into output dictionary."""
        results = {
            "date": target_date.isoformat(),
            "bbox": {"lat_min": bbox[0], "lat_max": bbox[1], "lon_min": bbox[2], "lon_max": bbox[3]},
            "model_type": self.model_type.name,
            "predictions": [],
        }

        if self.model_type == ModelType.CELLS:
            # 4 outputs: flown, crossed, wind_flown, humidity_flown
            flown, crossed, wind_flown, humidity_flown = predictions

            for cell_idx in range(self.nb_cells):
                cell_result = {
                    "cell_id": cell_idx,
                    "altitudes": {},
                }

                for alt_idx, alt_level in enumerate([1000, 900, 800, 700, 600]):
                    cell_result["altitudes"][f"{alt_level}hPa"] = {
                        "flyability": float(flown[0, cell_idx, alt_idx]),
                        "crossability": float(crossed[0, cell_idx, alt_idx]),
                        "wind_flyability": float(wind_flown[0, cell_idx, alt_idx]),
                        "humidity_flyability": float(humidity_flown[0, cell_idx, alt_idx]),
                    }

                results["predictions"].append(cell_result)

        else:  # SPOTS
            # One output per spot
            for cell_idx, cell_predictions in enumerate(predictions):
                for spot_idx, spot_value in enumerate(cell_predictions[0]):
                    results["predictions"].append(
                        {
                            "cell_id": cell_idx,
                            "spot_id": spot_idx,
                            "flyability": float(spot_value),
                        }
                    )

        return results

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
