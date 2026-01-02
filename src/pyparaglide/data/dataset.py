"""
Dataset loader for PyParaglide.

Loads training data from PKL files created by build_dataset.py.
Compatible with the original Paraglidable PKL format.
"""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np

from pyparaglide.data.normalization import Normalization

# Simple pickle loader (no BinObj dependency needed)


def load_pkl(file_path: Path | str, data_dir: Path | str) -> np.ndarray:
    """
    Load a pickle file from the data directory.

    Args:
        file_path: Name of the file to load
        data_dir: Directory containing the PKL files

    Returns:
        Loaded numpy array or object
    """
    import pickle

    path = Path(data_dir) / file_path
    with open(path, "rb") as f:
        return pickle.load(f)


@dataclass
class DatasetParams:
    """Dataset parameters (nb_days, nb_cells) without loading all data."""

    nb_days: int
    nb_cells: int

    @staticmethod
    def from_data_dir(data_dir: Path | str) -> "DatasetParams":
        """Load dataset parameters from data directory."""
        meteo_days = load_pkl("meteo_days.pkl", data_dir)
        sorted_cells = load_pkl("sorted_cells.pkl", data_dir)
        return DatasetParams(nb_days=len(meteo_days), nb_cells=len(sorted_cells))


@dataclass
class SpotsInfo:
    """Information about spots in a cell."""

    spots: list[dict]  # List of spot dicts with name, lat, lon
    flights_by_spot: dict[int, np.ndarray]  # spot_id -> array of flown values


class Dataset:
    """
    Main dataset class for training.

    Loads and prepares training data from PKL files.
    """

    # Parameter definitions
    _DEFS_OTHER = [
        ("Vertical velocity", [200, 300, 400, 500, 600, 700, 800, 900, 1000]),
        ("Geopotential Height", [200, 300, 400, 500, 600, 700, 800, 900, 1000]),
        ("Absolute vorticity", [200, 300, 400, 500, 600, 700, 800, 900, 1000]),
        ("Temperature", [200, 300, 400, 500, 600, 700, 800, 900, 1000]),
        ("Relative humidity", [200, 300, 400, 500, 600, 700, 800, 900, 1000]),
    ]

    _DEFS_WIND = [
        ("U component of wind", [600, 700, 800, 900, 1000]),
        ("V component of wind", [600, 700, 800, 900, 1000]),
    ]

    _DEFS_HUMIDITY = [
        ("Precipitable water", [0]),
        ("Cloud water", [0]),
    ]

    def __init__(self, data_dir: Path | str):
        """
        Initialize dataset.

        Args:
            data_dir: Directory containing PKL files
        """
        self.data_dir = Path(data_dir)

        # Load metadata
        self.meteo_days = load_pkl("meteo_days.pkl", self.data_dir)
        self.sorted_cells = load_pkl("sorted_cells.pkl", self.data_dir)
        self.meteo_params = load_pkl("meteo_params.pkl", self.data_dir)

        self.nb_days = len(self.meteo_days)
        self.nb_cells = len(self.sorted_cells)

        # Build parameter lists
        self._build_params_lists()

        # Load data matrices
        self._load_data()

    def _build_params_lists(self) -> None:
        """Build parameter lists for training (organized by hour)."""
        hours = [6, 12, 18]
        self.params_other = [[], [], []]
        self.params_wind = [[], [], []]
        self.params_humidity = [[], [], []]

        for h_idx, hour in enumerate(hours):
            # OTHER
            for name, levels in self._DEFS_OTHER:
                for level in levels:
                    found = self._find_param(hour, name, level)
                    if found:
                        self.params_other[h_idx].append(found)
                    else:
                        print(f"Warning: Missing param {name} {level} at {hour}h")

            # WIND
            for name, levels in self._DEFS_WIND:
                for level in levels:
                    found = self._find_param(hour, name, level)
                    if found:
                        self.params_wind[h_idx].append(found)
                    else:
                         print(f"Warning: Missing param {name} {level} at {hour}h")

            # HUMIDITY
            for name, levels in self._DEFS_HUMIDITY:
                for level in levels:
                    found = self._find_param(hour, name, level)
                    if found:
                        self.params_humidity[h_idx].append(found)
                    else:
                         print(f"Warning: Missing param {name} {level} at {hour}h")

    def _find_param(self, hour: int, name: str, level: int) -> tuple | None:
        """Find a parameter tuple in meteo_params."""
        for p in self.meteo_params:
            if p[0] != hour or p[1] != name:
                continue

            # Check level
            # p[2] structure varies:
            # [[('isobaricInhPa', 1000)]]
            # [('entireAtmosphere', 0), ('unknown', 0)]
            
            p_levels = p[2]
            if not isinstance(p_levels, list):
                continue

            # Try to match level
            for item in p_levels:
                if isinstance(item, list) and len(item) > 0 and isinstance(item[0], tuple):
                    # [[('isobaricInhPa', 1000)]] case
                    if item[0][1] == level:
                         return p
                elif isinstance(item, tuple):
                    # [('entireAtmosphere', 0), ...] case
                    if item[1] == level:
                        return p
        
        return None

    def _load_data(self) -> None:
        """Load all data matrices from PKL files."""
        # Meteo data
        self.meteo_content = load_pkl("meteo_content_by_cell_day.pkl", self.data_dir)

        # Flight data
        self.flights_by_cell_day = load_pkl("flights_by_cell_day.pkl", self.data_dir)
        self.mountainess_by_cell_alt = load_pkl("mountainess_by_cell_alt.pkl", self.data_dir)

        # Spots data (if available)
        try:
            self.spots = load_pkl("spots.pkl", self.data_dir)
            self.spots_by_cell = load_pkl("spots_by_cell.pkl", self.data_dir)
            self.flights_by_spot = load_pkl("flights_by_spot.pkl", self.data_dir)
            self.has_spots = True
        except (FileNotFoundError, EOFError):
            self.has_spots = False

    def get_lines(self, cells: list[int]) -> np.ndarray:
        """
        Get line indices for given cells.

        Returns indices in order: cell0_day0, cell0_day1, ..., cell1_day0, ...
        """
        return np.array([d * self.nb_cells + c for c in cells for d in range(self.nb_days)])

    def get_meteo_matrix(self, cells: list[int], params: list[tuple]) -> np.ndarray:
        """
        Get meteo data for specific cells and parameters.

        Args:
            cells: List of cell indices
            params: List of (hour, param_name, [levels]) tuples

        Returns:
            Array of shape (len(cells)*nb_days, len(params))
        """
        lines = self.get_lines(cells)
        param_indices = [self.meteo_params.index(p) for p in params]
        return self.meteo_content[lines][:, param_indices]

    def get_dow(self) -> np.ndarray:
        """Get day-of-week one-hot encoding."""
        X = np.zeros((self.nb_days, 7), dtype=np.float32)
        X[np.arange(self.nb_days), [d.weekday() for d in self.meteo_days]] = 1.0
        return X

    def get_date(self) -> np.ndarray:
        """Get normalized date values (0 to 1) with shape (nb_days, 1)."""
        return (np.arange(self.nb_days, dtype=np.float32) / (self.nb_days - 1)).reshape(-1, 1)

    def get_flights_by_altitude(
        self, cells: list[int], nb_altitudes: int, super_resolution: int, regression: bool
    ) -> list[np.ndarray]:
        """
        Get flight data grouped by altitude for CELLS model.

        Returns 4 arrays: flyability, crossability, wind_flyability, humidity_flyability
        """
        res = [
            np.zeros((len(cells) * super_resolution * super_resolution * self.nb_days, nb_altitudes), dtype=np.float32)
            for _ in range(4)
        ]

        res_line = 0
        for cell in cells:
            lines = self.get_lines([cell])

            # Group flights by super-resolution cell and day
            flight_list = [[[] for _ in lines] for _ in range(super_resolution * super_resolution)]

            for k_dc, dc in enumerate(lines):
                for flight in self.flights_by_cell_day[dc]:
                    # Calculate super-resolution cell position
                    lat, lon = flight[1][3], flight[1][4]
                    sr_x = int(((lon + 0.5) % 1.0) * super_resolution)
                    sr_y = int(((lat + 0.5) % 1.0) * super_resolution)
                    flight_list[sr_y * super_resolution + sr_x][k_dc].append(flight)

            points_limit = 60.0

            for sr_idx in range(super_resolution * super_resolution):
                for daycell in flight_list[sr_idx]:
                    if len(daycell) > 0:
                        if regression:
                            for k_alt in range(nb_altitudes):
                                flown = sum(1 for f in daycell if self._k_altitude(f[1][5]) == k_alt)
                                crossed = sum(1 for f in daycell if self._k_altitude(f[1][5]) == k_alt and f[1][0] >= points_limit)

                                res[0][res_line, k_alt] = flown
                                res[1][res_line, k_alt] = crossed
                                res[2][res_line, k_alt] = flown
                                res[3][res_line, k_alt] = flown
                        else:
                            flown_alts = {self._k_altitude(f[1][5]) for f in daycell}
                            crossed_alts = {self._k_altitude(f[1][5]) for f in daycell if f[1][0] >= points_limit}

                            for alt in flown_alts:
                                res[0][res_line, alt] = 1.0
                                res[2][res_line, alt] = 1.0
                                res[3][res_line, alt] = 1.0
                            for alt in crossed_alts:
                                res[1][res_line, alt] = 1.0

                    res_line += 1

        return res

    def get_flights_by_spots(self, cells: list[int]) -> dict[int, list[np.ndarray]]:
        """
        Get flight data for spots model.

        Returns dict mapping cell_id -> list of arrays (one per spot)
        Each array is binary: 1.0 if there was at least one flight on that day
        """
        if not self.has_spots:
            return {c: [] for c in cells}

        result = {}
        for cell in cells:
            spots_in_cell = self.spots_by_cell[cell]
            spot_flights = []

            for spot_idx in spots_in_cell:
                flights = self.flights_by_spot[spot_idx]
                # Build binary flown array: 1.0 if any flight on that day
                flown = np.zeros(self.nb_days, dtype=np.float32)
                for flight_record in flights:
                    # flight_record is (datetime_str, (score, None, takeoff_alt, lat, lon))
                    datetime_str = flight_record[0]
                    try:
                        flight_date = datetime.strptime(datetime_str, '%Y-%m-%d %H:%M:%S').date()
                        if flight_date in self.meteo_days:
                            day_idx = self.meteo_days.index(flight_date)
                            flown[day_idx] = 1.0
                    except (ValueError, KeyError):
                        pass
                spot_flights.append(flown)

            result[cell] = spot_flights

        return result

    def get_spots(self) -> dict[int, list]:
        """
        Get spots by cell.

        Returns:
            Dict mapping cell_id -> list of spot indices
            Returns empty dict if spots data not available
        """
        if not self.has_spots:
            return {}
        # Convert list to dict: [[spots_cell_0], [spots_cell_1], ...] -> {0: [...], 1: [...]}
        return {i: spots for i, spots in enumerate(self.spots_by_cell) if spots}

    @staticmethod
    def _barometric_leveling(altitude: float) -> float:
        """Convert altitude to pressure level (hPa)."""
        return 1013.25 * pow((1.0 - 0.0065 * altitude / 288.15), 5.255)

    @staticmethod
    def _k_altitude(altitude: float) -> int:
        """Convert altitude to altitude index (0-4 for 5 levels)."""
        pressure = Dataset._barometric_leveling(altitude)
        return max(0, min(4, int((1050 - pressure) // 100)))

    def get_mountainess(self, cells: list[int], nb_altitudes: int) -> np.ndarray:
        """
        Get mountainess values for cells.

        Returns array of shape (nb_cells, nb_altitudes)
        """
        mountainess = []
        for c in cells:
            cell_mountainess = [self.mountainess_by_cell_alt[c][alt] for alt in range(nb_altitudes)]
            mountainess.extend(cell_mountainess)
        return np.array(mountainess, dtype=np.float32).reshape(len(cells), nb_altitudes)


def convert_wind_matrix(wind_matrix: np.ndarray, wind_dim: int) -> np.ndarray:
    """
    Convert wind matrix to wind direction encoding.

    Matches the original implementation in neural_network/inc/utils.py:75-79.

    Input format: (nb_samples, 2*nb_altitudes) with U,V,U,V,... for each altitude
    Output format: (nb_samples, nb_altitudes*wind_dim) with direction bins

    Args:
        wind_matrix: Raw wind data with U,V components for each altitude
        wind_dim: Number of wind direction bins (typically 8)

    Returns:
        Encoded wind matrix where magnitude is placed in the correct direction bin
    """
    from pyparaglide.data.normalization import wind_uv_to_direction_bins

    nb_samples = wind_matrix.shape[0]
    nb_altitudes = 5  # Fixed: 600,700,800,900,1000 hPa

    result = np.zeros((nb_samples, nb_altitudes * wind_dim), dtype=np.float32)

    for alt in range(nb_altitudes):
        # Extract U,V for this altitude (columns alt*2, alt*2+1)
        uv = wind_matrix[:, alt * 2 : alt * 2 + 2]
        # Convert to direction bins
        encoded = wind_uv_to_direction_bins(uv, wind_dim)
        # Place in result at the correct position
        result[:, alt * wind_dim : (alt + 1) * wind_dim] = encoded

    return result