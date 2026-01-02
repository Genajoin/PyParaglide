"""
Dataset metadata management.

Handles JSON metadata files for tracking dataset configuration and enabling incremental builds.
"""

import json
from pathlib import Path
from typing import Any, Dict


def load_metadata(out_dir: str | Path) -> Dict[str, Any]:
    """
    Load dataset metadata from JSON file.

    Args:
        out_dir: Output directory containing dataset_config.json

    Returns:
        Dictionary with metadata, or empty dict if file doesn't exist
    """
    metadata_path = Path(out_dir) / "dataset_config.json"
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_metadata(out_dir: str | Path, metadata: Dict[str, Any]) -> None:
    """
    Save dataset metadata to JSON file.

    Args:
        out_dir: Output directory for dataset_config.json
        metadata: Dictionary with configuration data
    """
    metadata_path = Path(out_dir) / "dataset_config.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)


def check_config_match(current: Dict[str, Any], saved: Dict[str, Any]) -> bool:
    """
    Check if current configuration matches saved metadata.

    Args:
        current: Current configuration dictionary
        saved: Saved metadata dictionary

    Returns:
        True if configs match, False otherwise
    """
    if not current:
        return True  # No config to check
    if not saved:
        return False  # Have config but no saved data

    for key, value in current.items():
        if key not in saved:
            return False
        if isinstance(value, list):
            if saved[key] != value:
                return False
        elif isinstance(value, (int, float, str, bool)):
            if saved[key] != value:
                return False
        elif value is None:
            # None matches anything or missing
            continue
        else:
            # Unknown type, try direct comparison
            if saved.get(key) != value:
                return False
    return True
