"""
Dataset validation and auto-build utilities for PyParaglide.

This module provides functions to validate that required PKL files exist,
and optionally auto-build the dataset if missing.
"""

from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)

# Required PKL files for training
REQUIRED_CELLS_FILES = [
    "meteo_days.pkl",
    "mountainess_by_cell_alt.pkl",
    "sorted_cells.pkl",
    "meteo_params.pkl",
    "meteo_content_by_cell_day.pkl",
    "flights_by_cell_day.pkl",
]


def validate_dataset(pkl_dir: Path, model_type: str = "CELLS") -> tuple[bool, List[str]]:
    """
    Check if all required PKL files exist.

    Args:
        pkl_dir: Directory containing PKL files
        model_type: Only "CELLS" is supported

    Returns:
        Tuple of (all_exist: bool, missing_files: List[str])
    """
    pkl_dir = Path(pkl_dir)
    required_files = REQUIRED_CELLS_FILES
    missing = []

    for filename in required_files:
        if not (pkl_dir / filename).exists():
            missing.append(filename)

    all_exist = len(missing) == 0

    if all_exist:
        logger.info(f"Dataset validation passed - all PKL files present")
    else:
        logger.warning(f"Missing PKL files: {', '.join(missing)}")

    return all_exist, missing


def ensure_dataset_exists(
    pkl_dir: Path,
    model_type: str = "CELLS",
    auto_build: bool = False,
    flights_dir: Path | None = None,
) -> bool:
    """
    Ensure dataset exists.

    Args:
        pkl_dir: Directory containing PKL files
        model_type: Only "CELLS" is supported
        auto_build: If True, show error message about missing dataset
        flights_dir: Not used (kept for backward compatibility)

    Returns:
        True if dataset exists
    """
    all_exist, missing = validate_dataset(pkl_dir, model_type)

    if all_exist:
        return True

    logger.warning(f"Missing PKL files: {', '.join(missing)}")

    logger.error(
        "Dataset missing. Please run: pyparaglide build-dataset\n"
        "  Example: pyparaglide build-dataset --dates 2024-06-01:2024-08-31"
    )

    return False
