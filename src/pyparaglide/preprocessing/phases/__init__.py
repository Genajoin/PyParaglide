"""
Phase classes for dataset building pipeline.

Four phases:
1. BuildCellsPhase - Generate 1x1 degree cells and map to GRIB grid
2. BuildMeteoPhase - Scan GFS data and extract weather parameters (with date range accumulation)
3. BuildFlightsPhase - Process xContest flights and create spot PKL files
4. BuildTerrainPhase - Extract mountainess data from elevation tiles
"""

from .cells_phase import BuildCellsPhase
from .meteo_phase import BuildMeteoPhase
from .flights_phase import BuildFlightsPhase
from .terrain_phase import BuildTerrainPhase

__all__ = [
    "BuildCellsPhase",
    "BuildMeteoPhase",
    "BuildFlightsPhase",
    "BuildTerrainPhase",
]
