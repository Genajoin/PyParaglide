#!/usr/bin/env python3
"""Paraglidable CLI - unified interface for all scripts.

Usage:
    python scripts/cli.py ingest list --years 2020-2025
    python scripts/cli.py stats monthly --group-by quarter
    python scripts/cli.py --help
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

import typer
from lib.cli import app


def _register_commands():
    """Register all command groups with the app."""
    # Lazy imports to avoid Python 3.6 compatibility issues
    from commands import (
        ingest,
        stats,
        dataset,
        scan,
    )

    ingest.register(app)
    stats.register(app)
    dataset.register(app)
    scan.register(app)

    # Note: Some commands skipped due to Python 3.6 compatibility issues:
    # - bridge: ThreadingHTTPServer not available in Python 3.6
    # - metadata: xc_score/ uses type hints syntax from Python 3.9+
    # Use legacy wrappers:
    #   python scripts/legacy/igc_bridge_server.py
    #   python scripts/legacy/update_flights_metadata.py


if __name__ == "__main__":
    _register_commands()
    app()
