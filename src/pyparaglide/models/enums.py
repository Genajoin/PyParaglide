"""
Enums and settings for the neural network models.
"""

from enum import Enum

import numpy as np


class ModelType(Enum):
    """Model type: CELLS for grid-based, SPOTS for location-specific."""

    CELLS = 0
    SPOTS = 1


class ProblemFormulation(Enum):
    """Problem formulation: CLASSIFICATION or REGRESSION."""

    CLASSIFICATION = 0
    REGRESSION = 1


class ModelSettings:
    """Global model settings and hyperparameters."""

    optimize_dow = False

    # Day-of-week weights (normalized)
    dow_init = (
        np.array([[111383.0, 107993.0, 117721.0, 131987.0, 154665.0, 266616.0, 238255.0]])
        / np.mean([[111383.0, 107993.0, 117721.0, 131987.0, 154665.0, 266616.0, 238255.0]])
    )
