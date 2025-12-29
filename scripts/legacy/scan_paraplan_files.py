#!/usr/bin/env python3
"""Legacy wrapper for scan_paraplan_files.py.

For new code, use: python scripts/cli.py scan paraplan
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scan_paraplan_files import main as original_main

if __name__ == "__main__":
    original_main()
