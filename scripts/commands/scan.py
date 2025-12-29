"""Scan commands for local file matching."""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from lib.cli import app, get_db, db_url_option, source_option

# Import from scan_paraplan_files.py
from scan_paraplan_files import main as scan_main

scan_app = typer.Typer(help="Scan local IGC files")


@scan_app.command("paraplan")
def scan_paraplan(
    db_url: str = db_url_option(),
    source: str = source_option("paraplan"),
    igc_dir: str = typer.Option("/home/gena/par", "--igc-dir", help="Directory containing IGC files"),
):
    """Scan local paraplan IGC downloads and match to database."""
    # scan_main expects args namespace, construct it
    class Args:
        def __init__(self):
            self.db_url = db_url
            self.source = source
            self.igc_dir = igc_dir
            self.dry_run = False

    args = Args()
    scan_main(args)


def register(parent_app: typer.Typer) -> None:
    """Register scan subcommands with parent app."""
    parent_app.add_typer(scan_app, name="scan")
