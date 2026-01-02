#!/usr/bin/env python3
"""Legacy wrapper for igc_ingest_skygr.py.

This script maintains backward compatibility with the original command-line interface.
It maps old arguments to the new CLI structure.

For new code, use: python scripts/cli.py ingest <command>
"""

import sys
import os
from pathlib import Path

# Add scripts directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the original script's main function
from igc_ingest_skygr import main as original_main

if __name__ == "__main__":
    original_main()
