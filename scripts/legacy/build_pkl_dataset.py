#!/usr/bin/env python3
"""Legacy wrapper for build_pkl_dataset.py.

For new code, use: python scripts/cli.py dataset build
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from build_pkl_dataset import main as original_main

if __name__ == "__main__":
    original_main()
