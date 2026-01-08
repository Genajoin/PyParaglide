"""
Phase classes for dataset building pipeline.

Five phases (executed in order):
1. BuildCellsPhase - Generate 1x1 degree cells and map to GRIB grid
2. BuildMeteoPhase - Scan GFS data and extract weather parameters (with date range accumulation)
3. BuildFlightsPhase - Process xContest flights and create flight PKL files
4. BuildTerrainPhase - Extract mountainess data from elevation tiles
3.5. FilterCellsPhase - Filter data-sparse cells and reindex all PKL files (after Phase 4)
"""

from .cells_phase import BuildCellsPhase
from .meteo_phase import BuildMeteoPhase
from .flights_phase import BuildFlightsPhase
from .filter_phase import FilterCellsPhase
from .terrain_phase import BuildTerrainPhase

__all__ = [
    "BuildCellsPhase",
    "BuildMeteoPhase",
    "BuildFlightsPhase",
    "FilterCellsPhase",
    "BuildTerrainPhase",
]
