"""
Weight management utilities for PyParaglide.

This module provides functions to check if CELLS model weights exist,
and to train CELLS as a prerequisite for SPOTS training.
"""

from pathlib import Path
from typing import Literal
import logging

from pyparaglide.models.enums import ModelType, ProblemFormulation

logger = logging.getLogger(__name__)


def get_cells_weight_path(
    models_dir: Path,
    problem_formulation: ProblemFormulation = ProblemFormulation.CLASSIFICATION,
) -> Path:
    """
    Get the expected path for CELLS model weights.

    Args:
        models_dir: Directory containing model weights
        problem_formulation: CLASSIFICATION or REGRESSION

    Returns:
        Path to CELLS weights file
    """
    # Simple format: models_dir/cells_.weights.h5
    # (doesn't include version subdirectory for simplicity)
    return Path(models_dir) / "cells_.weights.h5"


def cells_weights_exist(
    models_dir: Path,
    problem_formulation: ProblemFormulation = ProblemFormulation.CLASSIFICATION,
) -> bool:
    """
    Check if CELLS model weights exist.

    Args:
        models_dir: Directory containing model weights
        problem_formulation: CLASSIFICATION or REGRESSION

    Returns:
        True if CELLS weights file exists
    """
    weight_path = get_cells_weight_path(models_dir, problem_formulation)
    exists = weight_path.exists()

    if exists:
        logger.info(f"Found CELLS weights at: {weight_path}")
    else:
        logger.warning(f"CELLS weights not found at: {weight_path}")

    return exists


def train_cells_prerequisite(
    pkl_dir: Path,
    models_dir: Path,
    cells: int,
    epochs: int,
    batch_size: int,
    problem_formulation: ProblemFormulation = ProblemFormulation.CLASSIFICATION,
    lr_init: float = 0.008,
    lr_end: float = 7e-4,
    validation: bool = True,
) -> Path:
    """
    Train CELLS model as prerequisite for SPOTS training.

    Args:
        pkl_dir: Directory containing PKL files
        models_dir: Directory to save model weights
        cells: Number of cells to train
        epochs: Number of training epochs
        batch_size: Training batch size
        problem_formulation: CLASSIFICATION or REGRESSION
        lr_init: Initial learning rate
        lr_end: Final learning rate
        validation: Whether to use validation set

    Returns:
        Path to saved CELLS weights
    """
    from pyparaglide.training.trainer import Trainer

    logger.info("Training CELLS model as prerequisite for SPOTS...")

    trainer = Trainer(
        data_dir=str(pkl_dir),
        model_type=ModelType.CELLS,
        problem_formulation=problem_formulation,
        models_dir=str(models_dir),
    )

    # Prepare data for all cells
    cell_indices = list(range(cells))
    X, Y = trainer.prepare_data(cells=cell_indices)

    # Create model
    trainer.create_model(cells=cell_indices)

    # Train
    trainer.train(
        X=X,
        Y=Y,
        lr_init=lr_init,
        lr_end=lr_end,
        nb_epochs=epochs,
        batch_size=batch_size,
        use_validation_set=validation,
    )

    # Save weights
    weight_path = trainer.save_weights()

    logger.info(f"CELLS model trained and saved to: {weight_path}")
    return weight_path
