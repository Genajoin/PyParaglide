"""
PyParaglide CLI - Command-line interface for paragliding flyability forecasting.

This module provides the main CLI entry point using Typer.
"""

from importlib import metadata

import typer
from rich.console import Console
from rich.table import Table

from pyparaglide import __version__
from pyparaglide.config import get_settings

# Create Typer app
app = typer.Typer(
    name="pyparaglide",
    help="AI-based paragliding flyability forecasting (TensorFlow 2.x)",
    add_completion=False,
    no_args_is_help=True,
)

# Rich console for pretty output
console = Console()


@app.command()
def version() -> None:
    """Show PyParaglide version information."""
    console.print(f"[bold cyan]PyParaglide[/bold cyan] version [green]{__version__}[/green]")

    # Show dependency versions
    table = Table(title="Dependencies", show_header=True, header_style="bold magenta")
    table.add_column("Package", style="cyan", width=30)
    table.add_column("Version", style="green")

    try:
        import tensorflow as tf

        table.add_row("TensorFlow", tf.__version__)
    except ImportError:
        table.add_row("TensorFlow", "[red]Not installed[/red]")

    try:
        import numpy as np

        table.add_row("NumPy", np.__version__)
    except ImportError:
        table.add_row("NumPy", "[red]Not installed[/red]")

    console.print(table)


@app.command()
def config(
    show_all: bool = typer.Option(False, "--all", "-a", help="Show all configuration including defaults"),
) -> None:
    """Show current configuration (from environment variables)."""
    settings = get_settings()

    table = Table(title="PyParaglide Configuration", show_header=True, header_style="bold magenta")
    table.add_column("Setting", style="cyan", width=30)
    table.add_column("Value", style="green")

    # Show key settings
    table.add_row("BBox", settings.bbox)
    table.add_row("GFS Directory", settings.gfs_dir)
    table.add_row("Models Directory", settings.models_dir)
    table.add_row("Output Directory", settings.output_dir)
    table.add_row("Flights Directory", settings.flights_dir)
    table.add_row("Elevation Directory", settings.elevation_dir)
    table.add_row("Forecast Days", str(settings.forecast_days))
    table.add_row("Training Dates", settings.training_dates)
    table.add_row("Min Flights per Spot", str(settings.min_flights_per_spot))
    table.add_row("Spot Cluster Distance (km)", str(settings.spot_cluster_distance_km))
    table.add_row("Debug", str(settings.debug))
    table.add_row("Workers", str(settings.workers))

    console.print(table)

    if show_all:
        console.print("\n[dim]All environment variables with PYPARAGLIDE_ prefix are respected.[/dim]")


@app.command()
def info() -> None:
    """Show system and environment information."""
    console.print("[bold cyan]PyParaglide[/bold cyan] - [green]System Information[/green]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Property", style="cyan", width=30)
    table.add_column("Value", style="green")

    table.add_row("Version", __version__)
    table.add_row("Author", "Evgeny Istomin")
    table.add_row("License", "GPL-3.0-or-later")
    table.add_row("Original Project", "Paraglidable by Antoine Bourgois")

    console.print(table)


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress non-error output"),
) -> None:
    """
    PyParaglide - AI-based paragliding flyability forecasting.

    This is a modernized fork of Paraglidable (https://paraglidable.com)
    migrated to TensorFlow 2.15+ and Python 3.12+.
    """
    # Global options can be stored here if needed
    pass


if __name__ == "__main__":
    app()
