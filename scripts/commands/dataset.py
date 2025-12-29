"""Dataset building commands for ML training."""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from lib.cli import app, get_db, db_url_option, source_option, years_option

# Import from build_pkl_dataset.py
from build_pkl_dataset import main as build_dataset_main

dataset_app = typer.Typer(help="Build PKL training datasets")


@dataset_app.command("build")
def dataset_build(
    db_url: str = db_url_option(),
    source: str = source_option("skygr"),
    years: str = years_option(""),
    gfs_dir: str = typer.Option("data/gfs", "--gfs-dir", help="GFS data directory"),
    output_dir: str = typer.Option("neural_network/bin/data", "--output-dir", help="Output directory"),
):
    """Build PKL training dataset from flights + GFS weather data."""
    class Args:
        def __init__(self):
            self.db_url = db_url
            self.source = source
            self.years = years
            self.gfs_dir = gfs_dir
            self.output_dir = output_dir

    args = Args()
    exit_code = build_dataset_main(args)
    raise SystemExit(exit_code)


def register(parent_app: typer.Typer) -> None:
    """Register dataset subcommands with parent app."""
    parent_app.add_typer(dataset_app, name="dataset")
