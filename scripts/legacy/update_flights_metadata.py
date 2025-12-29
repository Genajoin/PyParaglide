#!/usr/bin/env python3
"""Legacy wrapper for update_flights_metadata.py.

For new code, use: python scripts/cli.py metadata update
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from update_flights_metadata import main as original_main

if __name__ == "__main__":
    original_main()
