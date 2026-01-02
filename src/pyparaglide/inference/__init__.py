"""
Inference module for PyParaglide.

Handles forecast generation using trained models and GFS weather data.
"""

from pyparaglide.inference.forecast import Forecaster
from pyparaglide.inference.grib_reader import GribReader

__all__ = ["Forecaster", "GribReader"]
