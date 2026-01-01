"""
PyParaglide CLI - Command-line interface for paragliding flyability forecasting.

This module provides the main CLI entry point using Typer.
"""

from importlib import metadata
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

from pyparaglide import __version__
from pyparaglide.config import get_settings
from pyparaglide.downloads import GFSDownloader
from pyparaglide.inference import Forecaster
from pyparaglide.models.enums import ModelType, ProblemFormulation
from pyparaglide.preprocessing import DatasetBuilder
from pyparaglide.training import Trainer

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


@app.command()
def train(
    model_type: str = typer.Option("cells", "--model", "-m", help="Model type: 'cells' or 'spots'"),
    data_dir: str = typer.Option(None, "--data-dir", "-d", help="Directory containing PKL files"),
    models_dir: str = typer.Option(None, "--models-dir", "-o", help="Directory to save model weights"),
    cells: str = typer.Option(None, "--cells", "-c", help="Comma-separated list of cell indices (default: all)"),
    lr_init: float = typer.Option(0.008, "--lr-init", help="Initial learning rate"),
    lr_end: float = typer.Option(7e-4, "--lr-end", help="Final learning rate"),
    epochs: int = typer.Option(55, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    validation: bool = typer.Option(True, "--validation/--no-validation", help="Use validation set"),
    super_resolution: int = typer.Option(1, "--super-res", "-s", help="Super-resolution factor"),
    load_weights: bool = typer.Option(False, "--load-weights", help="Load existing weights"),
) -> None:
    """
    Train a PyParaglide model.

    Example:
        pyparaglide train --model cells --data-dir neural_network/bin/data --epochs 55
    """
    settings = get_settings()

    # Use defaults from settings if not specified
    if data_dir is None:
        data_dir = str(Path(settings.gfs_dir).parent.parent / "neural_network" / "bin" / "data")
    if models_dir is None:
        models_dir = settings.models_dir

    # Parse model type
    try:
        model_type_enum = ModelType[model_type.upper()]
    except KeyError:
        console.print(f"[red]Invalid model type: {model_type}[/red]")
        console.print("Valid options: [cyan]cells[/cyan], [cyan]spots[/cyan]")
        raise typer.Exit(1)

    # Parse cells
    if cells:
        cells_list = [int(c.strip()) for c in cells.split(",")]
    else:
        cells_list = None  # Will use all cells

    console.print(f"[bold cyan]Training {model_type_enum.name} model[/bold cyan]\n")

    # Create trainer
    trainer = Trainer(
        data_dir=data_dir,
        model_type=model_type_enum,
        problem_formulation=ProblemFormulation.CLASSIFICATION,
        models_dir=models_dir,
    )

    console.print(f"[dim]Data directory: {data_dir}[/dim]")
    console.print(f"[dim]Models directory: {models_dir}[/dim]")
    console.print(f"[dim]Cells: {'all' if cells_list is None else len(cells_list)}[/dim]")
    console.print(f"[dim]Epochs: {epochs}, Batch size: {batch_size}[/dim]")
    console.print(f"[dim]Learning rate: {lr_init} → {lr_end}[/dim]\n")

    # Prepare data
    console.print("[yellow]Preparing data...[/yellow]")
    X, Y = trainer.prepare_data(cells=cells_list, super_resolution=super_resolution)

    # Create model
    console.print("[yellow]Creating model...[/yellow]")
    actual_cells = cells_list if cells_list else list(range(trainer.nb_cells))
    trainer.create_model(cells=actual_cells, super_resolution=super_resolution, load_weights=load_weights)

    # Train
    console.print("[yellow]Training...[/yellow]\n")
    history = trainer.train(
        X=X,
        Y=Y,
        lr_init=lr_init,
        lr_end=lr_end,
        nb_epochs=epochs,
        batch_size=batch_size,
        use_validation_set=validation,
    )

    # Save
    console.print("\n[yellow]Saving model...[/yellow]")
    trainer.save_weights()

    # Results
    final_loss = history["loss"][-1]
    if "val_loss" in history and history["val_loss"]:
        final_val_loss = history["val_loss"][-1]
        console.print(f"\n[green]Training complete![/green]")
        console.print(f"Final loss: [cyan]{final_loss:.4f}[/cyan], Val loss: [cyan]{final_val_loss:.4f}[/cyan]")
    else:
        console.print(f"\n[green]Training complete![/green]")
        console.print(f"Final loss: [cyan]{final_loss:.4f}[/cyan]")


@app.command()
def forecast(
    date: str = typer.Option(None, "--date", "-d", help="Target date (YYYY-MM-DD, default: today)"),
    model_type: str = typer.Option("cells", "--model", "-m", help="Model type: 'cells' or 'spots'"),
    models_dir: str = typer.Option(None, "--models-dir", help="Directory with model weights"),
    grib_dir: str = typer.Option(None, "--grib-dir", "-g", help="Directory containing GRIB files"),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Output directory for forecasts"),
    bbox: str = typer.Option(None, "--bbox", "-b", help="Bounding box: lat_min,lat_max,lon_min,lon_max"),
) -> None:
    """
    Generate paragliding flyability forecast.

    Requires trained model weights and GFS GRIB files for the target date.

    Example:
        pyparaglide forecast --date 2025-06-15 --grib-dir data/gfs/2025-06-15
    """
    import datetime as dt

    settings = get_settings()

    # Use defaults from settings if not specified
    if models_dir is None:
        models_dir = settings.models_dir
    if grib_dir is None:
        grib_dir = str(Path(settings.gfs_dir) / "forecasts")
    if output_dir is None:
        output_dir = settings.output_dir
    if bbox is None:
        bbox = settings.bbox

    # Parse date
    if date is None:
        target_date = dt.date.today()
    else:
        try:
            target_date = dt.datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            console.print(f"[red]Invalid date format: {date}[/red]")
            console.print("Use format: [cyan]YYYY-MM-DD[/cyan]")
            raise typer.Exit(1)

    # Parse model type
    try:
        model_type_enum = ModelType[model_type.upper()]
    except KeyError:
        console.print(f"[red]Invalid model type: {model_type}[/red]")
        console.print("Valid options: [cyan]cells[/cyan], [cyan]spots[/cyan]")
        raise typer.Exit(1)

    console.print(f"[bold cyan]Generating {model_type_enum.name} forecast[/bold cyan]")
    console.print(f"[dim]Date: {target_date.isoformat()}[/dim]")
    console.print(f"[dim]Models: {models_dir}[/dim]")
    console.print(f"[dim]GRIB: {grib_dir}[/dim]\n")

    # Find GRIB files for the target date
    grib_path = Path(grib_dir)
    grib_files = []
    for hour in [6, 12, 18]:
        grib_file = grib_path / f"{target_date.strftime('%Y%m%d')}-{hour:02d}.grib2"
        if grib_file.exists():
            grib_files.append(grib_file)
        else:
            console.print(f"[yellow]Warning: GRIB file not found: {grib_file}[/yellow]")

    if len(grib_files) < 3:
        console.print(f"[red]Error: Found only {len(grib_files)}/3 GRIB files[/red]")
        console.print("Expected files for 06h, 12h, 18h forecasts")
        raise typer.Exit(1)

    # Create forecaster and generate prediction
    console.print("[yellow]Loading model...[/yellow]")
    forecaster = Forecaster(models_dir, model_type_enum)

    console.print("[yellow]Running prediction...[/yellow]")
    results = forecaster.predict_day(grib_files, target_date, tuple(float(x) for x in bbox.split(",")))

    # Save results
    output_path = Path(output_dir) / f"{target_date.isoformat()}_{model_type_enum.name.lower()}_forecast.json"
    console.print("[yellow]Saving results...[/yellow]")
    forecaster.save_forecast(results, output_path)

    console.print(f"\n[green]Forecast complete![/green]")
    console.print(f"Output: [cyan]{output_path}[/cyan]")


@app.command()
def download(
    start_date: str = typer.Option(None, "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(None, "--end", "-e", help="End date (YYYY-MM-DD)"),
    data_dir: str = typer.Option(None, "--data-dir", "-d", help="Output directory for GRIB files"),
    hours: str = typer.Option("0,6,12,18", "--hours", "-H", help="UTC hours to download (comma-separated)"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of parallel download workers"),
    filter: bool = typer.Option(False, "--filter", help="Filter GRIB files to reduce size by ~50%"),
) -> None:
    """
    Download GFS Analysis data from NOAA.

    Downloads historical GFS Analysis GRIB files for specified date range.
    Files are organized by month in the data directory.

    Example:
        pyparaglide download --start 2021-06-01 --end 2021-08-31 --workers 4 --filter
    """
    settings = get_settings()

    if data_dir is None:
        data_dir = settings.gfs_dir

    if start_date is None or end_date is None:
        console.print("[red]Error: --start and --end dates are required[/red]")
        raise typer.Exit(1)

    # Parse hours
    hour_list = [int(h.strip()) for h in hours.split(",")]

    console.print(f"[bold cyan]Downloading GFS Analysis data[/bold cyan]\n")

    # Create downloader
    downloader = GFSDownloader(
        data_dir=data_dir,
        hours=hour_list,
        workers=workers,
        filter_grib=filter,
    )

    # Download
    stats = downloader.download_range(start_date, end_date)

    # Return exit code based on failures
    if stats["failed"] > 0:
        raise typer.Exit(1)


@app.command()
def build_dataset(
    start_date: str = typer.Option(None, "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(None, "--end", "-e", help="End date (YYYY-MM-DD)"),
    gfs_dir: str = typer.Option(None, "--gfs-dir", help="Directory containing GFS GRIB files"),
    flights_dir: str = typer.Option(None, "--flights-dir", help="Directory with xContest JSON files"),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Output directory for PKL files"),
    bbox: str = typer.Option(None, "--bbox", "-b", help="Bounding box: lat_min,lat_max,lon_min,lon_max"),
    min_flights: int = typer.Option(200, "--min-flights", help="Minimum flights per spot"),
    no_flights: bool = typer.Option(False, "--no-flights", help="Skip flight data processing"),
) -> None:
    """
    Build PKL dataset from GFS GRIB files and flight data.

    Creates the PKL files needed for neural network training.
    For full functionality, use scripts/build_dataset.py directly.

    Example:
        pyparaglide build-dataset --start 2021-06-01 --end 2021-08-31
    """
    settings = get_settings()

    if gfs_dir is None:
        gfs_dir = settings.gfs_dir
    if flights_dir is None:
        flights_dir = settings.flights_dir
    if output_dir is None:
        output_dir = str(Path(gfs_dir).parent.parent / "neural_network" / "bin" / "data")
    if bbox is None:
        bbox = settings.bbox

    if start_date is None or end_date is None:
        console.print("[red]Error: --start and --end dates are required[/red]")
        raise typer.Exit(1)

    console.print(f"[bold cyan]Building PKL dataset[/bold cyan]\n")

    # Create builder
    builder = DatasetBuilder(
        gfs_dir=gfs_dir,
        flights_dir=flights_dir,
        output_dir=output_dir,
        bbox=tuple(float(x) for x in bbox.split(",")),
    )

    # Build dataset
    stats = builder.build(
        start_date=start_date,
        end_date=end_date,
        min_flights_per_spot=min_flights,
        include_flights=not no_flights,
    )

    console.print(f"\n[green]Dataset build complete![/green]")


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
