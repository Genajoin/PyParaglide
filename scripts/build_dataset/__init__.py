"""
Unified dataset builder for Paraglidable neural network training.

Replaces:
- scripts/build_pkl_dataset.py
- scripts/build_pkl_from_xcontest.py

Usage:
    python -m build_dataset --dates 2021-06-01:2021-08-31 --bbox 45,47,13,15
"""

from .phases import (
    BuildCellsPhase,
    BuildMeteoPhase,
    BuildFlightsPhase,
    BuildTerrainPhase,
)

__all__ = [
    "BuildCellsPhase",
    "BuildMeteoPhase",
    "BuildFlightsPhase",
    "BuildTerrainPhase",
]
