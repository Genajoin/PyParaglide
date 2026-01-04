"""
GRIB file reader for PyParaglide.

Provides two classes for reading GRIB files:
- GribReader: Standard reader with pygrib index
- InMemoryGribReader: Loads entire file into memory for fast sequential access

Both classes support extracting meteorological data from GFS GRIB files
for specific parameters and geographic locations.
"""

import logging
import os
from pathlib import Path
from typing import Any, Optional, Tuple

import numpy as np

try:
    import pygrib
    PYGRIB_AVAILABLE = True
except ImportError:
    PYGRIB_AVAILABLE = False

logger = logging.getLogger(__name__)


if not PYGRIB_AVAILABLE:
    raise ImportError(
        "pygrib is required for GRIB reading. "
        "Install it with: pip install pygrib"
    )


class GribReader:
    """
    Standard GRIB file reader with pygrib index for fast parameter selection.

    Usage:
        reader = GribReader("path/to/file.grib2")
        values = reader.getValues(params, cells_lat_lon)
    """

    def __init__(self, grib_file: str | Path) -> None:
        """
        Initialize GRIB reader with index.

        Args:
            grib_file: Path to GRIB file (.grb or .grib2)

        Raises:
            IOError: If file doesn't exist
        """
        self.grib_file = str(grib_file)
        if not os.path.exists(self.grib_file):
            raise IOError(f"GRIB file not found: {self.grib_file}")

        # Create index for fast parameter selection
        self.grb_indx = pygrib.index(self.grib_file, 'name', 'typeOfLevel', 'level')

        # Write index to disk for caching (.idx file alongside .grb2)
        # This makes subsequent opens much faster (index is loaded from disk)
        idx_file = self.grib_file + '.idx'
        if not os.path.exists(idx_file):
            try:
                self.grb_indx.write(idx_file)
            except IOError as e:
                logger.debug(f"Could not write index file: {e}")

    def get_infos(self) -> Tuple[Optional[Any], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get metadata from GRIB file.

        Returns:
            Tuple of (valid_date, distinct_latitudes, distinct_longitudes)
        """
        for grb in pygrib.open(self.grib_file):
            return grb.validDate, grb.distinctLatitudes, grb.distinctLongitudes
        return None, None, None

    def get_grid_structure(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Get grid origin and resolution.

        Returns:
            Tuple of (origin_lat, origin_lon, resolution_lat, resolution_lon)
        """
        for grb in pygrib.open(self.grib_file):
            resolution_lat = abs(grb.distinctLatitudes[1] - grb.distinctLatitudes[0])
            resolution_lon = abs(grb.distinctLongitudes[1] - grb.distinctLongitudes[0])
            origin_lat = grb.distinctLatitudes[-1] - 0.5 * resolution_lat
            origin_lon = grb.distinctLongitudes[0] - 0.5 * resolution_lon
            return origin_lat, origin_lon, resolution_lat, resolution_lon
        return None

    @staticmethod
    def find_closest(val: float, vect: np.ndarray, _lat_or_lon: int) -> int:
        """
        Find index of closest value in vector.

        Args:
            val: Target value
            vect: Array to search
            _lat_or_lon: 0 for latitude, 1 for longitude (unused, for compatibility)

        Returns:
            Index of closest value
        """
        return int((np.abs(vect - val)).argmin())

    def get_values(
        self,
        params: list[tuple[str, list[tuple[str, float]]]],
        cells_lat_lon: list[tuple[float, float]]
    ) -> list[float]:
        """
        Extract values for given parameters and cell coordinates.

        Args:
            params: List of (name, [(typeOfLevel, level), ...]) tuples
                   Example: [('Temperature', [('surface', 0)]), ('Wind', [('heightAboveGround', 10)])]
            cells_lat_lon: List of (lat, lon) tuples for cell centers

        Returns:
            List of values [param1_cell1, param1_cell2, ..., param2_cell1, ...]
            May return fewer values if some parameters are missing
        """
        cells = None
        values = []

        for param in params:
            name, levels = param
            for level_info in levels:
                type_of_level, level = level_info
                try:
                    selected_grbs = self.grb_indx.select(name=name, typeOfLevel=type_of_level, level=level)
                    if len(selected_grbs) != 1:
                        logger.warning(f"Expected 1 GRIB message for {name}/{type_of_level}/{level}, found {len(selected_grbs)}")
                        continue

                    for grb in selected_grbs:
                        # Map cells to grid indices (first time only)
                        if cells is None and cells_lat_lon:
                            cells = [
                                (self.find_closest(lat, grb.distinctLatitudes, 0),
                                 self.find_closest(lon, grb.distinctLongitudes, 1))
                                for lat, lon in cells_lat_lon
                            ]

                        # Extract values for each cell
                        if cells:
                            for cell in cells:
                                values.append(float(grb.values[cell[0], cell[1]]))

                except ValueError as e:
                    logger.debug(f"Parameter {name}/{type_of_level}/{level} not found: {e}")
                    continue
                except (KeyError, IndexError) as e:
                    logger.warning(f"Error extracting {name}/{type_of_level}/{level}: {e}")
                    continue

        expected_count = len(params) * len(cells_lat_lon) if cells_lat_lon else 0
        if len(values) != expected_count:
            logger.debug(f"Partial data: expected {expected_count} values, got {len(values)}")

        return values

    def get_values_array(
        self,
        params: list[tuple[str, list[tuple[str, float]]]],
        crops: list[tuple[int, int, int, int]]
    ) -> Optional[np.ndarray]:
        """
        Extract array values for given parameters and crop regions.

        Args:
            params: List of (name, [(typeOfLevel, level), ...]) tuples
            crops: List of (row_start, row_end, col_start, col_end) tuples

        Returns:
            Stacked numpy array of values, or None if extraction failed
        """
        stacks = []

        for param in params:
            name, levels = param
            for level_info in levels:
                type_of_level, level = level_info
                try:
                    selected_grbs = self.grb_indx.select(name=name, typeOfLevel=type_of_level, level=level)
                    if len(selected_grbs) != 1:
                        logger.warning(f"Expected 1 GRIB message for {name}/{type_of_level}/{level}, found {len(selected_grbs)}")
                        continue

                    for grb in selected_grbs:
                        stack = np.empty(0)
                        for crop in crops:
                            row_start, row_end, col_start, col_end = crop
                            stack = np.concatenate((stack, grb.values[row_start:row_end, col_start:col_end].flatten()))
                        stacks.append(stack)

                except (ValueError, KeyError, IndexError) as e:
                    logger.debug(f"Parameter {name}/{type_of_level}/{level} not found: {e}")
                    continue

        if len(stacks) != len(params):
            logger.warning(f"Partial extraction: got {len(stacks)}/{len(params)} parameters")
            return None

        return np.stack(stacks)


class InMemoryGribReader:
    """
    Load entire GRIB file into memory for fast sequential access.

    This is optimized for HDD storage where random seeks are expensive.
    The file is read sequentially once, then all data is accessed from RAM.

    Memory usage: ~500MB per GRIB file (65 messages × ~8MB each)

    Usage:
        reader = InMemoryGribReader("path/to/file.grib2")
        values = reader.getValues(params, cells_lat_lon)
    """

    def __init__(self, grib_file: str | Path) -> None:
        """
        Load all GRIB messages into memory.

        Args:
            grib_file: Path to GRIB file

        Raises:
            IOError: If file doesn't exist
        """
        self.grib_file = str(grib_file)
        if not os.path.exists(self.grib_file):
            raise IOError(f"GRIB file not found: {self.grib_file}")

        # Storage for messages and grid info
        # Key: (name, typeOfLevel, level) -> values array
        self.messages: dict[tuple[str, str, float], np.ndarray] = {}
        self.distinct_latitudes: Optional[np.ndarray] = None
        self.distinct_longitudes: Optional[np.ndarray] = None

        # Load all messages sequentially (single disk pass)
        self._load_all_messages()

    def _load_all_messages(self) -> None:
        """Load all GRIB messages into memory with sequential read."""
        grbs = pygrib.open(self.grib_file)

        try:
            for grb in grbs:
                # Get grid info from first message
                if self.distinct_latitudes is None:
                    self.distinct_latitudes = grb.distinctLatitudes
                    self.distinct_longitudes = grb.distinctLongitudes

                # Store message values in memory
                # Key: (name, typeOfLevel, level)
                key = (grb.name, grb.typeOfLevel, grb.level)
                self.messages[key] = grb.values.copy()  # Copy to keep in memory after file closes
        finally:
            grbs.close()

        logger.debug(f"Loaded {len(self.messages)} GRIB messages into memory")

    @staticmethod
    def find_closest(val: float, vect: np.ndarray, _lat_or_lon: int) -> int:
        """
        Find index of closest value in vector.

        Args:
            val: Target value
            vect: Array to search
            _lat_or_lon: 0 for latitude, 1 for longitude (unused, for compatibility)

        Returns:
            Index of closest value
        """
        return int((np.abs(vect - val)).argmin())

    def get_infos(self) -> Tuple[None, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get metadata from GRIB file.

        Returns:
            Tuple of (valid_date, distinct_latitudes, distinct_longitudes)
            Note: valid_date is None (not stored in memory mode)
        """
        return None, self.distinct_latitudes, self.distinct_longitudes

    def get_grid_structure(self) -> Optional[Tuple[float, float, float, float]]:
        """
        Get grid origin and resolution.

        Returns:
            Tuple of (origin_lat, origin_lon, resolution_lat, resolution_lon)
        """
        if self.distinct_latitudes is None or len(self.distinct_latitudes) < 2:
            return None

        resolution_lat = abs(self.distinct_latitudes[1] - self.distinct_latitudes[0])
        resolution_lon = abs(self.distinct_longitudes[1] - self.distinct_longitudes[0])
        origin_lat = self.distinct_latitudes[-1] - 0.5 * resolution_lat
        origin_lon = self.distinct_longitudes[0] - 0.5 * resolution_lon
        return origin_lat, origin_lon, resolution_lat, resolution_lon

    def get_values(
        self,
        params: list[tuple[str, list[tuple[str, float]]]],
        cells_lat_lon: list[tuple[float, float]]
    ) -> list[float]:
        """
        Extract values for given parameters and cell coordinates (from memory).

        Args:
            params: List of (name, [(typeOfLevel, level), ...]) tuples
            cells_lat_lon: List of (lat, lon) tuples for cell centers

        Returns:
            List of values [param1_cell1, param1_cell2, ..., param2_cell1, ...]
        """
        cells: Optional[list[tuple[int, int]]] = None
        values = []

        for param in params:
            name, levels = param
            for level_info in levels:
                type_of_level, level = level_info
                key = (name, type_of_level, level)

                if key not in self.messages:
                    logger.debug(f"Parameter {key} not found in GRIB file")
                    continue

                values_array = self.messages[key]

                # Map cells to grid indices (first time only)
                if cells is None and cells_lat_lon:
                    if self.distinct_latitudes is None or self.distinct_longitudes is None:
                        logger.warning("Grid info not available")
                        continue
                    cells = [
                        (self.find_closest(lat, self.distinct_latitudes, 0),
                         self.find_closest(lon, self.distinct_longitudes, 1))
                        for lat, lon in cells_lat_lon
                    ]

                # Extract values for each cell
                if cells:
                    for cell in cells:
                        values.append(float(values_array[cell[0], cell[1]]))

        return values


__all__ = ["GribReader", "InMemoryGribReader", "PYGRIB_AVAILABLE"]
