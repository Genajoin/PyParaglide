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
from pyparaglide.analysis import FlightAnalyzer, MeteoAnalyzer
from pyparaglide.config import get_settings, parse_date_ranges
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


# Analyze subcommand
analyze_app = typer.Typer(help="Analyze flights and weather data", add_completion=False)
app.add_typer(analyze_app, name="analyze")


@analyze_app.command()
def flights(
    flights_dir: str = typer.Option(None, "--flights-dir", "-d", help="Flights directory"),
    bbox: str = typer.Option(None, "--bbox", "-b", help="Bounding box filter: lat_min,lat_max,lon_min,lon_max"),
    min_flights: int = typer.Option(0, "--min-flights", "-m", help="Min flights per spot"),
) -> None:
    """
    Analyze flight distribution by cells and spots.

    Shows:
    - Total flights and bbox coverage
    - Distribution by 1°×1° cells
    - Distribution by month and country
    - Top spots with coordinates
    - Auto-detected cell clusters
    - [bold yellow]Recommended bboxes[/bold yellow] for training

    Example:
        pyparaglide analyze flights
        pyparaglide analyze flights --bbox 45,47,13,16 --min-flights 50
    """
    import datetime as dt

    settings = get_settings()

    if flights_dir is None:
        flights_dir = settings.flights_dir

    # Parse bbox
    parsed_bbox = None
    if bbox:
        parts = [float(x.strip()) for x in bbox.split(",")]
        if len(parts) == 4:
            parsed_bbox = tuple(parts)
        else:
            console.print(f"[red]Invalid bbox format: {bbox}[/red]")
            console.print("Expected: lat_min,lat_max,lon_min,lon_max")
            raise typer.Exit(1)

    # Create analyzer
    console.print("[bold cyan]Analyzing flight data...[/bold cyan]\n")

    try:
        analyzer = FlightAnalyzer(
            flights_dir=Path(flights_dir),
            bbox=parsed_bbox,
        )
        result = analyzer.analyze(min_flights_threshold=min_flights)
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    # Display summary
    from rich.panel import Panel

    summary = f"""Total flights: {result.total_flights:,}
Cells: {len(result.by_cell)}
Countries: {len(result.by_country)}
Spots with >={min_flights} flights: {len(result.by_spot)}"""

    if result.bbox_coverage:
        summary += f"\nInside bbox: {result.bbox_coverage.inside:,}"
        summary += f"\nOutside bbox: {result.bbox_coverage.outside:,}"
        summary += f"\nNo coordinates: {result.bbox_coverage.no_coords:,}"

    console.print(Panel(summary, title="[bold]Flight Summary[/bold]", border_style="cyan"))

    # By cell (top 20)
    if result.by_cell:
        console.print(f"\n[bold]BY CELL (1°×1°) - Top 20[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Cell", style="cyan")
        table.add_column("Flights", style="green")

        sorted_cells = sorted(result.by_cell.items(), key=lambda x: -x[1])
        for cell, count in sorted_cells[:20]:
            lat, lon = map(int, cell.split(","))
            table.add_row(f"({lat:3d}, {lon:3d})", f"{count:,}")

        if len(result.by_cell) > 20:
            table.add_row("...", f"and {len(result.by_cell) - 20} more")

        console.print(table)

    # By month
    if result.by_month:
        console.print(f"\n[bold]BY MONTH[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Month", style="cyan")
        table.add_column("Flights", style="green")

        month_names = {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }

        for month in sorted(result.by_month.keys()):
            table.add_row(month_names[month], f"{result.by_month[month]:,}")

        console.print(table)

    # By country
    if result.by_country:
        console.print(f"\n[bold]BY COUNTRY (top 10)[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Country", style="cyan")
        table.add_column("Flights", style="green")

        sorted_countries = sorted(result.by_country.items(), key=lambda x: -x[1])
        for country, count in sorted_countries[:10]:
            table.add_row(country, f"{count:,}")

        console.print(table)

    # By spot
    if result.by_spot:
        console.print(f"\n[bold]BY SPOT (top 20)[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Name", style="cyan")
        table.add_column("Flights", style="green")
        table.add_column("Coords", style="yellow")

        sorted_spots = sorted(result.by_spot.items(), key=lambda x: -x[1].count)
        for spot_id, data in sorted_spots[:20]:
            table.add_row(data.name[:30], f"{data.count:,}", f"({data.lat:.2f}, {data.lon:.2f})")

        console.print(table)

    # Recommendations
    console.print(f"\n[bold yellow]RECOMMENDATIONS[/bold yellow]")
    console.print("[dim](Analysis-based suggestions - user may have intentionally different settings)[/dim]\n")

    if result.clusters:
        console.print(f"[bold]Detected Cell Clusters:[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("#", style="cyan")
        table.add_column("BBox", style="green")
        table.add_column("Cells", style="yellow")
        table.add_column("Flights", style="yellow")
        table.add_column("Top Spot", style="blue")

        for i, cluster in enumerate(result.clusters[:5], 1):
            bbox_str = f"{cluster.lat_min},{cluster.lat_max},{cluster.lon_min},{cluster.lon_max}"
            table.add_row(str(i), bbox_str, str(cluster.count), f"{cluster.flights:,}", cluster.top_spot[:20])

        console.print(table)

        # Best cluster recommendation
        best = result.clusters[0]
        console.print(f"\n[bold yellow]Recommended bbox for best cluster:[/bold yellow]")
        console.print(f"  [cyan]--bbox {best.lat_min},{best.lat_max},{best.lon_min},{best.lon_max}[/cyan]")
        console.print(f"  [dim]({best.flights:,} flights in {best.count} cells)[/dim]")


@analyze_app.command()
def meteo(
    gfs_dir: str = typer.Option(None, "--gfs-dir", "-d", help="GFS directory"),
    dates: str = typer.Option(None, "--dates", "-D", help="Date ranges to check"),
) -> None:
    """
    Analyze downloaded GFS weather data.

    Shows:
    - Available days with complete data (06h, 12h, 18h)
    - Completeness percentage
    - Missing days by month
    - [bold yellow]Recommendations[/bold yellow] for downloading

    Example:
        pyparaglide analyze meteo
        pyparaglide analyze meteo --dates 2024-06-01:2024-08-31
    """
    import datetime as dt

    settings = get_settings()

    if gfs_dir is None:
        gfs_dir = settings.gfs_dir

    # Parse date ranges
    date_ranges = None
    if dates:
        try:
            date_ranges = parse_date_ranges(dates)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
    elif settings.training_dates:
        str_ranges = settings.parse_training_dates()
        if str_ranges:
            date_ranges = [
                (
                    dt.datetime.strptime(start, "%Y-%m-%d").date(),
                    dt.datetime.strptime(end, "%Y-%m-%d").date(),
                )
                for start, end in str_ranges
            ]

    # Create analyzer
    console.print("[bold cyan]Analyzing GFS data...[/bold cyan]\n")

    try:
        analyzer = MeteoAnalyzer(
            gfs_dir=Path(gfs_dir),
            date_ranges=date_ranges,
        )
        result = analyzer.check_completeness()
    except ValueError as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)

    # Display summary
    from rich.panel import Panel

    completeness_color = "green" if result.complete_percentage >= 90 else "yellow"
    if result.complete_percentage < 70:
        completeness_color = "red"

    summary = f"""Completeness: [{completeness_color}]{result.complete_percentage:.1f}%[/{completeness_color}]
Available days: {len(result.available_days)}
Missing days: {len(result.missing_days)}"""

    console.print(Panel(summary, title="[bold]GFS Data Summary[/bold]", border_style="cyan"))

    # By month
    if result.by_month:
        console.print(f"\n[bold]Completeness by Month:[/bold]")
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Month", style="cyan")
        table.add_column("Expected", style="yellow")
        table.add_column("Available", style="green")
        table.add_column("Missing", style="red")
        table.add_column("Completeness", style="yellow")

        for month, stats in sorted(result.by_month.items()):
            color = "green" if stats.completeness_percentage >= 90 else "yellow"
            if stats.completeness_percentage < 70:
                color = "red"
            table.add_row(
                month,
                str(stats.expected_days),
                str(stats.available_days),
                str(len(stats.missing_days)),
                f"[{color}]{stats.completeness_percentage:.1f}%[/{color}]",
            )

        console.print(table)

    # Recommendations
    console.print(f"\n[bold yellow]RECOMMENDATIONS[/bold yellow]")
    console.print("[dim](Analysis-based suggestions - user may have intentionally different settings)[/dim]\n")

    if result.missing_days:
        console.print(f"[yellow]Missing {len(result.missing_days)} days - consider downloading:[/yellow]")

        # Group by date ranges
        ranges = []
        current_start = None
        prev_date = None

        for d in sorted(result.missing_days):
            if current_start is None:
                current_start = d
                prev_date = d
            elif (d - prev_date).days <= 2:  # Within 2 days - same range
                prev_date = d
            else:
                ranges.append((current_start, prev_date))
                current_start = d
                prev_date = d

        if current_start:
            ranges.append((current_start, prev_date))

        # Format ranges for command line
        range_strs = []
        for start, end in ranges:
            range_strs.append(f"{start.isoformat()}:{end.isoformat()}")

        console.print(f"  [cyan]pyparaglide download --dates \"{' ,'.join(range_strs)}\"[/cyan]")
    else:
        console.print(f"[green]All expected days are available![/green]")


@app.command()
def train(
    model_type: str = typer.Option("spots", "--model", "-m", help="Model type: 'cells' or 'spots' (default: spots)"),
    data_dir: str = typer.Option(None, "--data-dir", "-d", help="Directory containing PKL files"),
    models_dir: str = typer.Option(None, "--models-dir", "-o", help="Directory to save model weights"),
    cells: str = typer.Option(None, "--cells", "-c", help="Number of cells or comma-separated list (default: all)"),
    lr_init: float = typer.Option(0.008, "--lr-init", help="Initial learning rate"),
    lr_end: float = typer.Option(7e-4, "--lr-end", help="Final learning rate"),
    epochs: int = typer.Option(55, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    validation: bool = typer.Option(True, "--validation/--no-validation", help="Use validation set"),
    super_resolution: int = typer.Option(1, "--super-res", "-s", help="Super-resolution factor"),
    load_weights: bool = typer.Option(False, "--load-weights", help="Load existing weights (CELLS only)"),
) -> None:
    """
    Train a PyParaglide model.

    CELLS Model: Trains all cells at once for grid-based flyability prediction.
    SPOTS Model (default): Trains one model per cell for spot-specific prediction.

    SPOTS training requires CELLS weights - will auto-train if missing.

    Example:
        pyparaglide train --cells 10 --epochs 55
    """
    from pyparaglide.preprocessing.dataset_utils import ensure_dataset_exists
    from pyparaglide.training.weight_utils import cells_weights_exist, train_cells_prerequisite

    settings = get_settings()

    # Use defaults from settings if not specified
    if data_dir is None:
        data_dir = settings.pkl_dir
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
        # Support: "--cells 3" (cell index 3) and "--cells 0,1,2" (specific indices)
        cells_list = [int(c.strip()) for c in cells.split(",")]
    else:
        cells_list = None  # Will use all cells

    # Determine number of cells
    nb_cells_to_train = len(cells_list) if cells_list else 10  # Default to 10

    console.print(f"[bold cyan]Training {model_type_enum.name} model[/bold cyan]\n")

    # Step 1: Validate dataset
    console.print("[bold blue]Step 1/3: Validating dataset...[/bold blue]")
    if not ensure_dataset_exists(
        pkl_dir=Path(data_dir),
        model_type=model_type_enum.name,
        auto_build=True,
        flights_dir=Path(settings.flights_dir) if model_type_enum == ModelType.SPOTS else None,
    ):
        if model_type_enum == ModelType.SPOTS:
            console.print("[red]SPOTS dataset incomplete. Options:[/red]")
            console.print("  1. [cyan]Ensure flight JSON files exist in:[/cyan]")
            console.print(f"     {settings.flights_dir}")
            console.print("  2. [cyan]Or use the original build script:[/cyan]")
            console.print("     python scripts/build_dataset.py --dates 2024-06-01:2024-08-31")
            console.print("  3. [cyan]Or train CELLS model instead:[/cyan]")
            console.print("     pyparaglide train --model cells --cells 10 --epochs 55")
        else:
            console.print("[red]Dataset validation failed. Please run:[/red]")
            console.print("  [cyan]pyparaglide build-dataset --dates 2024-06-01:2024-08-31[/cyan]")
        raise typer.Exit(1)
    console.print("[green]Dataset validation passed ✓[/green]\n")

    # Step 2: For SPOTS, ensure CELLS weights exist
    if model_type_enum == ModelType.SPOTS:
        console.print("[bold blue]Step 2/3: Checking CELLS weights prerequisite...[/bold blue]")

        if not cells_weights_exist(Path(models_dir)):
            console.print("[yellow]CELLS weights not found. Training CELLS first...[/yellow]")
            train_cells_prerequisite(
                pkl_dir=Path(data_dir),
                models_dir=Path(models_dir),
                cells=nb_cells_to_train,
                epochs=epochs,
                batch_size=batch_size,
                lr_init=lr_init,
                lr_end=lr_end,
                validation=validation,
            )
        else:
            console.print("[green]CELLS weights found ✓[/green]")
        console.print()

    # Step 3: Train model
    console.print("[bold blue]Step 3/3: Training model...[/bold blue]\n")

    if model_type_enum == ModelType.CELLS:
        # CELLS: Train all cells at once
        _train_cells(
            data_dir=data_dir,
            models_dir=models_dir,
            cells_list=cells_list,
            nb_cells_to_train=nb_cells_to_train,
            epochs=epochs,
            batch_size=batch_size,
            lr_init=lr_init,
            lr_end=lr_end,
            validation=validation,
            super_resolution=super_resolution,
            load_weights=load_weights,
        )
    else:
        # SPOTS: Train one model per cell
        actual_cells = cells_list if cells_list is not None else list(range(10))
        _train_spots(
            data_dir=data_dir,
            models_dir=models_dir,
            cells_to_train=actual_cells,
            epochs=epochs,
            batch_size=batch_size,
            lr_init=lr_init,
            lr_end=lr_end,
            validation=validation,
            super_resolution=super_resolution,
        )


def _train_cells(
    data_dir: str,
    models_dir: str,
    cells_list: list[int] | None,
    nb_cells_to_train: int,
    epochs: int,
    batch_size: int,
    lr_init: float,
    lr_end: float,
    validation: bool,
    super_resolution: int,
    load_weights: bool,
) -> None:
    """Train CELLS model (all cells at once)."""
    console.print(f"[dim]Data directory: {data_dir}[/dim]")
    console.print(f"[dim]Models directory: {models_dir}[/dim]")
    console.print(f"[dim]Cells: {nb_cells_to_train}[/dim]")
    console.print(f"[dim]Epochs: {epochs}, Batch size: {batch_size}[/dim]")
    console.print(f"[dim]Learning rate: {lr_init} → {lr_end}[/dim]\n")

    # Create trainer
    trainer = Trainer(
        data_dir=data_dir,
        model_type=ModelType.CELLS,
        problem_formulation=ProblemFormulation.CLASSIFICATION,
        models_dir=models_dir,
    )

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


def _train_spots(
    data_dir: str,
    models_dir: str,
    cells_to_train: list[int],
    epochs: int,
    batch_size: int,
    lr_init: float,
    lr_end: float,
    validation: bool,
    super_resolution: int,
) -> None:
    """Train SPOTS models (one per cell)."""
    cells_weight_path = Path(models_dir) / "cells.weights.h5"
    trained_count = 0
    skipped_count = 0

    console.print(f"[dim]Data directory: {data_dir}[/dim]")
    console.print(f"[dim]Models directory: {models_dir}[/dim]")
    console.print(f"[dim]Training {len(cells_to_train)} SPOTS models (one per cell)[/dim]")
    console.print(f"[dim]Epochs: {epochs}, Batch size: {batch_size}[/dim]")
    console.print(f"[dim]Learning rate: {lr_init} → {lr_end}[/dim]\n")

    for cell_idx in cells_to_train:
        console.print(f"[bold cyan]Training cell {cell_idx}...[/bold cyan]")

        try:
            # Create trainer for this cell
            trainer = Trainer(
                data_dir=data_dir,
                model_type=ModelType.SPOTS,
                problem_formulation=ProblemFormulation.CLASSIFICATION,
                models_dir=models_dir,
            )

            # Check if cell has spots
            try:
                spot_count = trainer.get_spot_count_for_cell(cell_idx)
                console.print(f"[dim]Cell {cell_idx} has {spot_count} spots[/dim]")
            except ValueError as e:
                console.print(f"[yellow]Skipping cell {cell_idx}: {e}[/yellow]\n")
                skipped_count += 1
                continue

            # Prepare data for this cell only
            X, Y = trainer.prepare_data_for_cell(cell_idx)

            # Create model for this cell
            trainer.create_model(cells=[cell_idx], super_resolution=super_resolution)

            # Load weights from CELLS
            trainer.load_weights_from_cells(cells_weight_path, freeze_transferred=False)

            # Train
            console.print("[yellow]Training...[/yellow]")
            history = trainer.train(
                X=X,
                Y=Y,
                lr_init=lr_init,
                lr_end=lr_end,
                nb_epochs=epochs,
                batch_size=batch_size,
                use_validation_set=validation,
            )

            # Save with cell suffix
            weight_path = trainer.save_weights(suffix=f"_cell_{cell_idx}")

            final_loss = history["loss"][-1]
            console.print(f"[green]Cell {cell_idx} complete: {weight_path.name}[/green]")
            console.print(f"[dim]Final loss: {final_loss:.4f}[/dim]\n")
            trained_count += 1

        except Exception as e:
            console.print(f"[red]Error training cell {cell_idx}: {e}[/red]\n")
            skipped_count += 1

    # Summary
    console.print("[bold green]SPOTS training complete![/bold green]")
    console.print(f"Trained: [cyan]{trained_count}[/cyan] cells")
    if skipped_count > 0:
        console.print(f"Skipped: [yellow]{skipped_count}[/yellow] cells")


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
    dates: str = typer.Option(None, "--dates", "-D", help="Date ranges (format: YYYY-MM-DD:YYYY-MM-DD,...). Overrides .env TRAINING_DATES"),
    start_date: str = typer.Option(None, "--start", "-s", help="Start date (YYYY-MM-DD). Shorthand for single range. Use --dates for multiple ranges"),
    end_date: str = typer.Option(None, "--end", "-e", help="End date (YYYY-MM-DD). Must be used with --start"),
    data_dir: str = typer.Option(None, "--data-dir", "-d", help="Output directory for GRIB files"),
    hours: str = typer.Option("0,6,12,18", "--hours", "-H", help="UTC hours to download (comma-separated)"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of parallel download workers"),
    filter: bool = typer.Option(False, "--filter", help="Filter GRIB files to reduce size by ~50%"),
) -> None:
    """
    Download GFS Analysis data from NOAA.

    Downloads historical GFS Analysis GRIB files for specified date ranges.

    Date sources (in order of priority):
    1. --dates "2024-06-01:2024-08-31,2025-06-01:2025-08-31" (multiple ranges)
    2. --start/--end (single range)
    3. TRAINING_DATES from .env (default)

    Examples:
        # Use dates from .env (TRAINING_DATES)
        pyparaglide download

        # Single range with --dates
        pyparaglide download --dates 2024-06-01:2024-08-31

        # Multiple ranges
        pyparaglide download --dates "2024-06-01:2024-08-31,2025-06-01:2025-08-31"

        # Legacy format (single range)
        pyparaglide download --start 2024-06-01 --end 2024-08-31

        # With options
        pyparaglide download --workers 4 --filter
    """
    import datetime as dt

    settings = get_settings()

    if data_dir is None:
        data_dir = settings.gfs_dir

    # Parse hours
    hour_list = [int(h.strip()) for h in hours.split(",")]

    console.print(f"[bold cyan]Downloading GFS Analysis data[/bold cyan]\n")

    # Determine date ranges (priority: --dates > --start/--end > .env TRAINING_DATES)
    if dates:
        # Parse --dates argument (same format as .env)
        try:
            date_ranges = parse_date_ranges(dates)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
    elif start_date and end_date:
        # Legacy --start/--end format (single range)
        try:
            date_ranges = parse_date_ranges(f"{start_date}:{end_date}")
        except ValueError:
            console.print(f"[red]Invalid date format[/red]")
            console.print("Use format: [cyan]YYYY-MM-DD[/cyan]")
            raise typer.Exit(1)
    elif start_date or end_date:
        # Only one of --start/--end specified
        console.print("[red]Error: --start and --end must be used together, or use --dates instead[/red]")
        raise typer.Exit(1)
    else:
        # Use ranges from .env TRAINING_DATES
        # parse_training_dates returns strings, so we need to convert back to dates
        str_ranges = settings.parse_training_dates()
        if not str_ranges:
            console.print("[red]Error: No date ranges found. Specify --dates, --start/--end, or set TRAINING_DATES in .env[/red]")
            raise typer.Exit(1)
        date_ranges = [
            (dt.datetime.strptime(start, "%Y-%m-%d").date(),
             dt.datetime.strptime(end, "%Y-%m-%d").date())
            for start, end in str_ranges
        ]

    # Create downloader
    downloader = GFSDownloader(
        data_dir=data_dir,
        hours=hour_list,
        workers=workers,
        filter_grib=filter,
    )

    # Download all ranges
    total_stats = {"downloaded": 0, "skipped": 0, "failed": 0, "total_mb": 0.0}

    for start, end in date_ranges:
        console.print(f"\n[yellow]Processing range: {start} to {end}[/yellow]")
        stats = downloader.download_range(start, end)

        for key in total_stats:
            if key != "total_mb":
                total_stats[key] += stats[key]
        total_stats["total_mb"] += stats["total_mb"]

    # Print summary
    console.print(f"\n[bold]Total Summary:[/bold]")
    console.print(f"  Downloaded: {total_stats['downloaded']} files ({total_stats['total_mb']:.1f} MB)")
    console.print(f"  Skipped: {total_stats['skipped']} files")
    console.print(f"  Failed: {total_stats['failed']} files")

    # Return exit code based on failures
    if total_stats["failed"] > 0:
        raise typer.Exit(1)


@app.command()
def build_dataset(
    dates: str = typer.Option(None, "--dates", "-D", help="Date ranges (format: YYYY-MM-DD:YYYY-MM-DD,...). Overrides .env TRAINING_DATES"),
    start_date: str = typer.Option(None, "--start", "-s", help="Start date (YYYY-MM-DD). Shorthand for single range. Use --dates for multiple ranges"),
    end_date: str = typer.Option(None, "--end", "-e", help="End date (YYYY-MM-DD). Must be used with --start"),
    gfs_dir: str = typer.Option(None, "--gfs-dir", help="Directory containing GFS GRIB files"),
    flights_dir: str = typer.Option(None, "--flights-dir", help="Directory with xContest JSON files"),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Output directory for PKL files"),
    bbox: str = typer.Option(None, "--bbox", "-b", help="Bounding box: lat_min,lat_max,lon_min,lon_max"),
    min_flights: int = typer.Option(200, "--min-flights", help="Minimum flights per spot"),
    no_flights: bool = typer.Option(False, "--no-flights", help="Skip flight data processing"),
    force: bool = typer.Option(False, "--force", help="Force rebuild even if PKL files exist"),
    analyze: bool = typer.Option(False, "--analyze", "-a", help="Run analysis and show recommendations after build"),
) -> None:
    """
    Build PKL dataset from GFS GRIB files and flight data.

    Creates the PKL files needed for neural network training.

    Date sources (in order of priority):
    1. --dates "2024-06-01:2024-08-31,2025-06-01:2025-08-31" (multiple ranges)
    2. --start/--end (single range)
    3. TRAINING_DATES from .env (default)

    Examples:
        # Use dates from .env (TRAINING_DATES)
        pyparaglide build-dataset

        # Single range with --dates
        pyparaglide build-dataset --dates 2024-06-01:2024-08-31

        # Multiple ranges
        pyparaglide build-dataset --dates "2024-06-01:2024-08-31,2025-06-01:2025-08-31"

        # Legacy format (single range)
        pyparaglide build-dataset --start 2024-06-01 --end 2024-08-31
    """
    import datetime as dt

    settings = get_settings()

    if gfs_dir is None:
        gfs_dir = settings.gfs_dir
    if flights_dir is None:
        flights_dir = settings.flights_dir
    if output_dir is None:
        output_dir = settings.pkl_dir
    if bbox is None:
        bbox = settings.bbox

    # Determine date ranges (priority: --dates > --start/--end > .env TRAINING_DATES)
    if dates:
        # Parse --dates argument (same format as .env)
        try:
            date_ranges = parse_date_ranges(dates)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
    elif start_date and end_date:
        # Legacy --start/--end format (single range)
        try:
            date_ranges = parse_date_ranges(f"{start_date}:{end_date}")
        except ValueError:
            console.print(f"[red]Invalid date format[/red]")
            console.print("Use format: [cyan]YYYY-MM-DD[/cyan]")
            raise typer.Exit(1)
    elif start_date or end_date:
        # Only one of --start/--end specified
        console.print("[red]Error: --start and --end must be used together, or use --dates instead[/red]")
        raise typer.Exit(1)
    else:
        # Use ranges from .env TRAINING_DATES
        # parse_training_dates returns strings, so we need to convert back to dates
        str_ranges = settings.parse_training_dates()
        if not str_ranges:
            console.print("[red]Error: No date ranges found. Specify --dates, --start/--end, or set TRAINING_DATES in .env[/red]")
            raise typer.Exit(1)
        date_ranges = [
            (dt.datetime.strptime(start, "%Y-%m-%d").date(),
             dt.datetime.strptime(end, "%Y-%m-%d").date())
            for start, end in str_ranges
        ]

    console.print(f"[bold cyan]Building PKL dataset[/bold cyan]")
    console.print(f"[dim]Date ranges: {len(date_ranges)}[/dim]\n")

    # Create builder with elevation_dir from settings
    from pyparaglide.config import settings

    builder = DatasetBuilder(
        gfs_dir=gfs_dir,
        flights_dir=flights_dir,
        output_dir=output_dir,
        bbox=tuple(float(x) for x in bbox.split(",")),
        elevation_dir=settings.elevation_dir,
    )

    # CRITICAL FIX: Build dataset for ALL date ranges at once
    # This fixes the sequential overwrite bug where each range was overwriting previous data
    stats = builder.build_all(
        date_ranges=date_ranges,
        min_flights_per_spot=min_flights,
        include_flights=not no_flights,
        cluster_distance_km=settings.spot_cluster_distance_km,
        num_workers=settings.workers,
        force=force,
    )

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Cells: {stats['cells']}")
    console.print(f"  Spots: {stats['spots']}")
    console.print(f"  Days: {stats['days']}")

    console.print(f"\n[green]Dataset build complete![/green]")

    # Run analysis if requested
    if analyze:
        console.print(f"\n[bold cyan]Running post-build analysis...[/bold cyan]\n")

        # Flight analysis
        if not no_flights:
            console.print(f"[bold]Analyzing flight data...[/bold]")
            try:
                flight_analyzer = FlightAnalyzer(
                    flights_dir=Path(flights_dir),
                    bbox=tuple(float(x) for x in bbox.split(",")),
                )
                flight_result = flight_analyzer.analyze(min_flights_threshold=min_flights)

                # Show flight summary
                from rich.panel import Panel

                flight_summary = f"""Total flights: {flight_result.total_flights:,}
Cells: {len(flight_result.by_cell)}
Spots with >={min_flights} flights: {len(flight_result.by_spot)}"""

                console.print(Panel(flight_summary, title="[bold]Flight Summary[/bold]", border_style="cyan"))

                # Show top clusters
                if flight_result.clusters:
                    console.print(f"\n[bold]Top Clusters:[/bold]")
                    table = Table(show_header=True, header_style="bold magenta")
                    table.add_column("BBox", style="green")
                    table.add_column("Cells", style="yellow")
                    table.add_column("Flights", style="yellow")
                    table.add_column("Top Spot", style="blue")

                    for cluster in flight_result.clusters[:3]:
                        bbox_str = f"{cluster.lat_min},{cluster.lat_max},{cluster.lon_min},{cluster.lon_max}"
                        table.add_row(bbox_str, str(cluster.count), f"{cluster.flights:,}", cluster.top_spot[:20])

                    console.print(table)

                    # Recommendation
                    best = flight_result.clusters[0]
                    console.print(f"\n[bold yellow]Recommended bbox:[/bold yellow] [cyan]{best.lat_min},{best.lat_max},{best.lon_min},{best.lon_max}[/cyan]")
                    console.print(f"[dim]({best.flights:,} flights in {best.count} cells)[/dim]")
            except ValueError as e:
                console.print(f"[yellow]Flight analysis skipped: {e}[/yellow]")

        # Meteo analysis
        console.print(f"\n[bold]Analyzing GFS data...[/bold]")
        try:
            meteo_analyzer = MeteoAnalyzer(
                gfs_dir=Path(gfs_dir),
                date_ranges=date_ranges,
            )
            meteo_result = meteo_analyzer.check_completeness()

            # Show meteo summary
            completeness_color = "green" if meteo_result.complete_percentage >= 90 else "yellow"
            if meteo_result.complete_percentage < 70:
                completeness_color = "red"

            meteo_summary = f"""Completeness: [{completeness_color}]{meteo_result.complete_percentage:.1f}%[/{completeness_color}]
Available days: {len(meteo_result.available_days)}
Missing days: {len(meteo_result.missing_days)}"""

            console.print(Panel(meteo_summary, title="[bold]GFS Data Summary[/bold]", border_style="cyan"))

            if meteo_result.missing_days:
                console.print(f"\n[yellow]Missing {len(meteo_result.missing_days)} days[/yellow]")
            else:
                console.print(f"\n[green]All expected days are available![/green]")
        except ValueError as e:
            console.print(f"[yellow]Meteo analysis skipped: {e}[/yellow]")

        # Show disclaimer
        console.print(f"\n[dim]Analysis results are recommendations based on your data.[/dim]")
        console.print(f"[dim]You may have intentionally chosen different settings.[/dim]")


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
