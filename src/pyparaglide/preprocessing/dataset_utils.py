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

REQUIRED_SPOTS_FILES = REQUIRED_CELLS_FILES + [
    "spots_by_cell.pkl",  # Created by scripts/build_dataset.py, not by simplified DatasetBuilder
    "flights_by_spot.pkl",  # Created by scripts/build_dataset.py
]


def validate_dataset(pkl_dir: Path, model_type: str = "CELLS") -> tuple[bool, List[str]]:
    """
    Check if all required PKL files exist.

    Args:
        pkl_dir: Directory containing PKL files
        model_type: "CELLS" or "SPOTS" - which files to check for

    Returns:
        Tuple of (all_exist: bool, missing_files: List[str])
    """
    pkl_dir = Path(pkl_dir)
    required_files = REQUIRED_SPOTS_FILES if model_type == "SPOTS" else REQUIRED_CELLS_FILES
    missing = []

    for filename in required_files:
        if not (pkl_dir / filename).exists():
            missing.append(filename)

    all_exist = len(missing) == 0

    if all_exist:
        logger.info(f"Dataset validation passed ({model_type}) - all PKL files present")
    else:
        logger.warning(f"Missing PKL files for {model_type}: {', '.join(missing)}")

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
        model_type: "CELLS" or "SPOTS" - which files to check for
        auto_build: If True, attempt to build SPOTS files when missing
        flights_dir: Directory containing flight JSON files (for SPOTS auto-build)

    Returns:
        True if dataset exists
    """
    all_exist, missing = validate_dataset(pkl_dir, model_type)

    if all_exist:
        return True

    logger.warning(f"Missing PKL files: {', '.join(missing)}")

    if auto_build:
        if model_type == "SPOTS":
            # Try to build SPOTS files automatically
            if flights_dir is None:
                logger.error(
                    "SPOTS dataset incomplete. Please specify flights_dir or run:\n"
                    "  python scripts/build_dataset.py --dates 2024-06-01:2024-08-31"
                )
                return False

            logger.info("Attempting to build SPOTS PKL files...")
            from pyparaglide.preprocessing.spots_builder import build_spots_dataset

            if build_spots_dataset(Path(flights_dir), pkl_dir):
                # Re-validate after build
                all_exist, missing = validate_dataset(pkl_dir, model_type)
                if all_exist:
                    logger.info("SPOTS dataset build complete!")
                    return True

            logger.error("SPOTS dataset build failed")
            return False
        else:
            logger.error(
                "Dataset missing. Please run: pyparaglide build-dataset\n"
                "  Example: pyparaglide build-dataset --dates 2024-06-01:2024-08-31"
            )
    else:
        logger.error("Dataset missing and auto_build is disabled")

    return False
