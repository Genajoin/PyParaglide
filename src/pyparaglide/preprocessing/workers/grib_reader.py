"""
GRIB file reader wrapper.

This module provides optional wrappers for the neural_network GRIB reader.
If pygrib is not available, these will be None.
"""

from typing import Optional

# Try to import from neural_network
GribReader = None
InMemoryGribReader = None

try:
    import sys
    from pathlib import Path

    # Add neural_network to path
    neural_network_path = Path(__file__).parent.parent.parent.parent.parent / "neural_network"
    if neural_network_path.exists():
        sys.path.insert(0, str(neural_network_path))

    from inc.grib_reader import GribReader, InMemoryGribReader

except ImportError:
    # pygrib or neural_network/inc/grib_reader not available
    GribReader = None
    InMemoryGribReader = None

__all__ = ["GribReader", "InMemoryGribReader"]
