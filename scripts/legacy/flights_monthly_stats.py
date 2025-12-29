#!/usr/bin/env python3
"""Legacy wrapper for flights_monthly_stats.py.

For new code, use: python scripts/cli.py stats monthly
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from flights_monthly_stats import main as original_main

if __name__ == "__main__":
    original_main()
