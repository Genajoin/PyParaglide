"""Metadata update commands for flight records."""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from lib.cli import app, get_db, db_url_option, source_option

# Import from update_flights_metadata.py
from update_flights_metadata import main as update_metadata_main

metadata_app = typer.Typer(help="Update flight metadata")


@metadata_app.command("update")
def metadata_update(
    db_url: str = db_url_option(),
    source: str = source_option("skygr"),
    max_files: int = typer.Option(0, "--max-files", help="Maximum files to process (0=unlimited)"),
    workers: int = typer.Option(4, "--workers", help="Number of worker processes"),
):
    """Reparse IGC files and update database with enhanced metadata."""
    # Construct args namespace for main function
    class Args:
        def __init__(self):
            self.db_url = db_url
            self.source = source
            self.max_files = max_files
            self.workers = workers

    args = Args()
    exit_code = update_metadata_main(args)
    raise SystemExit(exit_code)


def register(parent_app: typer.Typer) -> None:
    """Register metadata subcommands with parent app."""
    parent_app.add_typer(metadata_app, name="metadata")
