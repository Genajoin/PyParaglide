#!/usr/bin/env python3
"""Legacy wrapper for igc_bridge_server.py.

For new code, use: python scripts/cli.py bridge serve
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from igc_bridge_server import main as original_main

if __name__ == "__main__":
    original_main()
