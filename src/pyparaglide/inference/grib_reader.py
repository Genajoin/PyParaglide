"""
GRIB file reader for GFS weather data.

Reads GRIB2 files downloaded from NOAA and extracts weather parameters
needed for forecasting.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pygrib


class GribReader:
    """
    Reader for GFS GRIB2 files.

    Extracts weather parameters needed for paragliding forecasts.
    """

    # Altitude levels in hPa (hectopascals)
    ALTITUDE_LEVELS = [1000, 900, 800, 700, 600, 500, 400, 300, 200]

    # Forecast hours to extract (6h, 12h, 18h)
    FORECAST_HOURS = [6, 12, 18]

    # Parameters to extract (name, levels)
    # Based on original GfsData parameters
    PARAMS_WIND = [("u", "wind", ALTITUDE_LEVELS[:5]), ("v", "wind", ALTITUDE_LEVELS[:5])]  # 600-1000 hPa
    PARAMS_OTHER = [
        ("w", "vertical_velocity", ALTITUDE_LEVELS),  # VVEL
        ("h", "geopotential_height", ALTITUDE_LEVELS),  # HGT
        ("absv", "absolute_vorticity", ALTITUDE_LEVELS),  # ABSV
        ("t", "temperature", ALTITUDE_LEVELS),  # TMP
        ("r", "relative_humidity", ALTITUDE_LEVELS),  # RH
    ]
    PARAMS_HUMIDITY = [
        ("pwat", "precipitable_water", [0]),  # Entire atmosphere
        ("cwat", "cloud_water", [0]),  # Entire atmosphere
    ]

    def __init__(self, grib_path: Path | str):
        """
        Initialize GRIB reader.

        Args:
            grib_path: Path to GRIB2 file
        """
        self.grib_path = Path(grib_path)
        self._messages: list[pygrib.gribmessage] | None = None

    def _load(self) -> list[pygrib.gribmessage]:
        """Load GRIB file messages."""
        if self._messages is None:
            self._messages = pygrib.open(str(self.grib_path)).read()
        return self._messages

    def get_param(self, name: str, level: int | None = None) -> np.ndarray | None:
        """
        Extract a parameter from the GRIB file.

        Args:
            name: Parameter name (e.g., 't', 'u', 'v', 'r', 'h', 'pwat')
            level: Pressure level in hPa (None for surface/whole-atmosphere params)

        Returns:
            2D numpy array of values, or None if not found
        """
        messages = self._load()

        # Find matching messages
        for msg in messages:
            # Match parameter name
            param_name = msg.getParameterName(False).lower()
            if param_name != name.lower():
                continue

            # Match level (if specified)
            if level is not None:
                msg_level = msg.getLevel()
                if msg_level != level:
                    continue

            # Get data
            data, lats, lons = msg.data()

            # Handle missing values
            if hasattr(msg, "missing"):
                data = np.ma.filled(data, fill_value=0)

            return data

        return None

    def get_bbox_data(
        self, name: str, bbox: tuple[float, float, float, float], level: int | None = None
    ) -> np.ndarray | None:
        """
        Extract parameter data for a bounding box.

        Args:
            name: Parameter name
            bbox: (lat_min, lat_max, lon_min, lon_max)
            level: Pressure level in hPa

        Returns:
            2D numpy array cropped to bbox
        """
        data = self.get_param(name, level)
        if data is None:
            return None

        # Get lat/lon grid from any message
        messages = self._load()
        if len(messages) > 0:
            _, lats, lons = messages[0].data()

            # Find indices for bbox
            lat_min, lat_max, lon_min, lon_max = bbox

            # Handle longitude wrapping
            if lon_min < 0:
                lon_min += 360
            if lon_max < 0:
                lon_max += 360

            lat_mask = (lats >= lat_min) & (lats <= lat_max)
            lon_mask = (lons >= lon_min) & (lons <= lon_max)

            # Apply mask (this is simplified - real implementation needs proper grid slicing)
            return data[lat_mask, :][:, lon_mask] if lon_mask.any() else data[lat_mask, :]

        return data

    @staticmethod
    def wind_to_directions(u: np.ndarray, v: np.ndarray, nb_directions: int = 8) -> np.ndarray:
        """
        Convert U/V wind components to direction bins.

        Args:
            u: U-component array
            v: V-component array
            nb_directions: Number of direction bins (typically 8)

        Returns:
            Array of shape (..., nb_directions) with binned wind data
        """
        # Calculate wind direction and magnitude
        magnitude = np.sqrt(u**2 + v**2)
        direction = np.arctan2(v, u)  # Radians

        # Convert to bins (0 = N, 1 = NE, 2 = E, etc.)
        bin_size = 2 * np.pi / nb_directions
        direction_bins = np.floor((direction + np.pi) / bin_size).astype(int) % nb_directions

        # Create one-hot encoded bins weighted by magnitude
        result = np.zeros(u.shape + (nb_directions,), dtype=np.float32)

        # Iterate over grid points and assign to bins
        it = np.nditer(u.shape, flags=["multi_index"])
        for _ in it:
            idx = it.multi_index
            bin_idx = direction_bins[idx]
            result[idx + (bin_idx,)] = magnitude[idx]

        return result
