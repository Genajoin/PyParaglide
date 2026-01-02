"""
Analysis module for PyParaglide.

Provides tools to analyze flight data and weather data
to help users determine optimal training configuration.
"""

from pyparaglide.analysis.flights_analyzer import FlightAnalyzer
from pyparaglide.analysis.meteo_analyzer import MeteoAnalyzer

__all__ = ["FlightAnalyzer", "MeteoAnalyzer"]
