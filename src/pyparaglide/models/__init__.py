"""
PyParaglide Neural Network Models.

This module contains the TensorFlow 2.x neural network architecture
for paragliding flyability forecasting.

Based on the original Paraglidable TensorFlow 1.15 implementation,
migrated to TensorFlow 2.15+ with modern Python patterns.
"""

from pyparaglide.models.enums import ModelType, ModelSettings, ProblemFormulation
from pyparaglide.models.layers import (
    CrossabilityBlock,
    FlyabilityBlock,
    HumidityFlyabilityBlock,
    PopulationBlock,
    WindBlockCells,
    WindBlockSpots,
    WindFlyabilityBlock,
)
from pyparaglide.models.model_cells import ModelCells
from pyparaglide.models.model_spots import ModelSpots

__all__ = [
    # Enums
    "ModelType",
    "ProblemFormulation",
    "ModelSettings",
    # Layers
    "WindFlyabilityBlock",
    "HumidityFlyabilityBlock",
    "FlyabilityBlock",
    "CrossabilityBlock",
    "WindBlockSpots",
    "WindBlockCells",
    "PopulationBlock",
    # Models
    "ModelCells",
    "ModelSpots",
]
