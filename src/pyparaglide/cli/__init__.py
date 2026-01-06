"""
PyParaglide CLI - Command-line interface for paragliding flyability forecasting.

This module provides the main CLI entry point using Typer.
"""

import warnings
from importlib import metadata
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table
from tqdm import tqdm

# Suppress warnings from dependencies
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*is ill-defined.*")
warnings.filterwarnings("ignore", message=".*ROC AUC score.*")

from pyparaglide import __version__
from pyparaglide.analysis import FlightAnalyzer, MeteoAnalyzer
from pyparaglide.config import get_settings, parse_date_ranges
from pyparaglide.downloads import GFSDownloader, GFSForecastDownloader
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

# Download subcommands (short alias 'dl' to avoid conflict with flat 'download' command)
download_app = typer.Typer(help="Download GFS weather and elevation data (subcommands)", add_completion=False)
app.add_typer(download_app, name="dl")


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

        console.print(f"  [cyan]pyparaglide download --dates \"{','.join(range_strs)}\"[/cyan]")
    else:
        console.print(f"[green]All expected days are available![/green]")


# =============================================================================
# Download Subcommands
# =============================================================================


@download_app.command("analysis")
def download_analysis(
    dates: str = typer.Option(None, "--dates", "-D", help="Date ranges (format: YYYY-MM-DD:YYYY-MM-DD,...)"),
    start_date: str = typer.Option(None, "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(None, "--end", "-e", help="End date (YYYY-MM-DD)"),
    data_dir: str = typer.Option(None, "--data-dir", "-d", help="Output directory for GRIB files"),
    hours: str = typer.Option("0,6,12,18", "--hours", "-H", help="UTC hours to download (comma-separated)"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of parallel download workers"),
    filter: bool = typer.Option(False, "--filter", help="Filter GRIB files to reduce size by ~50%"),
) -> None:
    """
    Download GFS Analysis data (historical weather for training).

    Downloads historical GFS Analysis GRIB files from AWS S3 for neural network training.

    Examples:
        pyparaglide download analysis --dates 2024-06-01:2024-08-31
        pyparaglide download analysis --start 2024-06-01 --end 2024-08-31
        pyparaglide download analysis --workers 4 --filter
    """
    from pyparaglide.cli._download_helpers import download_analysis_impl

    download_analysis_impl(
        dates=dates,
        start_date=start_date,
        end_date=end_date,
        data_dir=data_dir,
        hours=hours,
        workers=workers,
        filter_grib=filter,
    )


@download_app.command("forecast")
def download_forecast(
    days: int = typer.Option(10, "--days", "-D", help="Number of days ahead to download from latest forecast run"),
    date: str = typer.Option(None, "--date", "-d", help="Specific target date (YYYY-MM-DD)"),
    start_date: str = typer.Option(None, "--start", "-s", help="Start date (YYYY-MM-DD)"),
    end_date: str = typer.Option(None, "--end", "-e", help="End date (YYYY-MM-DD)"),
    data_dir: str = typer.Option(None, "--data-dir", "-o", help="Output directory for GRIB files"),
    hours: str = typer.Option("6,12,18", "--hours", "-H", help="Forecast hours to download (comma-separated)"),
    force: bool = typer.Option(False, "--force", "-f", help="Redownload existing files"),
) -> None:
    """
    Download GFS Forecast data (for generating predictions).

    Downloads GFS Forecast GRIB files from NOMADS for generating flyability predictions.
    Automatically finds the best available forecast run and uses correct forecast offsets.

    MODES:
    1. --days N   : Download N days ahead from latest forecast run (default)
    2. --date D   : Download forecast for specific date
    3. --start S --end E : Download for date range

    Examples:
        pyparaglide dl forecast --days 10
        pyparaglide dl forecast --date 2026-01-05
        pyparaglide dl forecast --start 2026-01-05 --end 2026-01-15
    """
    import datetime as dt

    settings = get_settings()

    if data_dir is None:
        # Use parent of gfs_dir (data/gfs) instead of gfs_dir (data/gfs/anl)
        # Forecast files should go to data/gfs/forecasts/, not data/gfs/anl/forecasts/
        gfs_path = Path(settings.gfs_dir)
        data_dir = str(gfs_path.parent if gfs_path.parent.name == "gfs" else gfs_path)

    # Parse hours
    hour_list = [int(h.strip()) for h in hours.split(",")]

    # Create downloader
    downloader = GFSForecastDownloader(data_dir=data_dir)
    downloader.forecast_hours = hour_list

    # Determine download mode
    if date:
        # Single date mode
        if start_date or end_date:
            console.print("[red]Error: --date cannot be used with --start/--end[/red]")
            raise typer.Exit(1)

        try:
            target_date = dt.datetime.strptime(date, "%Y-%m-%d").date()
        except ValueError:
            console.print(f"[red]Invalid date format: {date}[/red]")
            console.print("Use format: [cyan]YYYY-MM-DD[/cyan]")
            raise typer.Exit(1)

        console.print(f"[bold cyan]Downloading GFS Forecast data[/bold cyan]\n")
        console.print(f"[dim]Mode: Single date[/dim]")
        console.print(f"[dim]Date: {target_date.isoformat()}[/dim]")
        console.print(f"[dim]Hours: {hours}[/dim]")
        if force:
            console.print(f"[dim]Force: Enabled (will re-download existing files)[/dim]")
        console.print()

        stats = downloader.download_day(target_date, force=force)

    elif start_date and end_date:
        # Date range mode
        try:
            start = dt.datetime.strptime(start_date, "%Y-%m-%d").date()
            end = dt.datetime.strptime(end_date, "%Y-%m-%d").date()
        except ValueError:
            console.print("[red]Invalid date format[/red]")
            console.print("Use format: [cyan]YYYY-MM-DD[/cyan]")
            raise typer.Exit(1)

        console.print(f"[bold cyan]Downloading GFS Forecast data[/bold cyan]\n")
        console.print(f"[dim]Mode: Date range[/dim]")
        console.print(f"[dim]Range: {start.isoformat()} to {end.isoformat()}[/dim]")
        console.print(f"[dim]Hours: {hours}[/dim]")
        if force:
            console.print(f"[dim]Force: Enabled (will re-download existing files)[/dim]")
        console.print()

        stats = {"downloaded": 0, "skipped": 0, "failed": 0, "total_mb": 0.0}

        current = start
        while current <= end:
            day_stats = downloader.download_day(current, force=force)
            stats["downloaded"] += day_stats["downloaded"]
            stats["skipped"] += day_stats["skipped"]
            stats["failed"] += day_stats["failed"]
            stats["total_mb"] += day_stats["total_mb"]
            current += dt.timedelta(days=1)

    else:
        # Days ahead mode (default)
        console.print(f"[bold cyan]Downloading GFS Forecast data[/bold cyan]\n")
        console.print(f"[dim]Mode: Days ahead from latest run[/dim]")
        console.print(f"[dim]Days: {days}[/dim]")
        console.print(f"[dim]Hours: {hours}[/dim]")
        if force:
            console.print(f"[dim]Force: Enabled (will re-download existing files)[/dim]")
        console.print()

        stats = downloader.download_days_ahead(days=days, force=force)

    # Print summary
    console.print(f"\n[bold]Download Summary:[/bold]")
    console.print(f"  Downloaded: {stats['downloaded']} files ({stats['total_mb']:.1f} MB)")
    console.print(f"  Skipped: {stats['skipped']} files")
    console.print(f"  Failed: {stats['failed']} files")

    if stats["failed"] > 0:
        raise typer.Exit(1)


@download_app.command("elevation")
def download_elevation(
    data_dir: str = typer.Option(None, "--data-dir", "-o", help="Output directory for elevation files"),
    bbox: str = typer.Option(None, "--bbox", "-b", help="Bounding box: lat_min,lat_max,lon_min,lon_max"),
) -> None:
    """
    Download SRTM elevation data.

    Downloads SRTM elevation tiles for terrain modeling.

    Examples:
        pyparaglide download elevation
        pyparaglide download elevation --bbox 45,47,13,15
    """
    from pyparaglide.cli._download_helpers import download_elevation_impl

    download_elevation_impl(
        data_dir=data_dir,
        bbox=bbox,
    )


@app.command()
def train(
    data_dir: str = typer.Option(None, "--data-dir", "-d", help="Directory containing PKL files"),
    models_dir: str = typer.Option(None, "--models-dir", "-o", help="Directory to save model weights"),
    lr_init: float = typer.Option(0.008, "--lr-init", help="Initial learning rate"),
    lr_end: float = typer.Option(7e-4, "--lr-end", help="Final learning rate"),
    epochs: int = typer.Option(55, "--epochs", "-e", help="Number of training epochs"),
    batch_size: int = typer.Option(32, "--batch-size", "-b", help="Batch size"),
    validation: bool = typer.Option(True, "--validation/--no-validation", help="Use validation set"),
    validation_split: float = typer.Option(0.0, "--validation-split", "-v", help="Validation split: 0=alternating days (default), 0.2=Keras random split (0.0-1.0)"),
    super_resolution: int = typer.Option(1, "--super-res", "-s", help="Super-resolution factor"),
    load_weights: bool = typer.Option(False, "--load-weights", help="Load existing weights"),
    early_stopping_patience: int = typer.Option(0, "--early-stopping-patience", "-p", help="Early stopping patience (0 = disabled, requires --validation)"),
    thermo: bool = typer.Option(False, "--thermo", help="Enable thermodynamic parameters (PBLH, TCDC, CAPE, LI, CIN)"),
    suffix: str = typer.Option("", "--suffix", help="Model suffix for versioning (e.g., '_thermo', '_baseline')"),
) -> None:
    """
    Train PyParaglide CELLS model for grid-based flyability prediction.

    Trains all cells at once using the CELLS architecture.

    Example:
        pyparaglide train --epochs 55
    """
    from pyparaglide.preprocessing.dataset_utils import ensure_dataset_exists

    settings = get_settings()

    # Use defaults from settings if not specified
    if data_dir is None:
        data_dir = settings.pkl_dir
    if models_dir is None:
        models_dir = settings.models_dir

    console.print(f"[bold cyan]Training CELLS model[/bold cyan]\n")

    # Step 1: Validate dataset
    console.print("[bold blue]Step 1/2: Validating dataset...[/bold blue]")
    if not ensure_dataset_exists(
        pkl_dir=Path(data_dir),
        model_type="CELLS",
        auto_build=True,
    ):
        console.print("[red]Dataset validation failed. Please run:[/red]")
        console.print("  [cyan]pyparaglide build-dataset --dates 2024-06-01:2024-08-31[/cyan]")
        raise typer.Exit(1)
    console.print("[green]Dataset validation passed ✓[/green]\n")

    # Step 2: Train model
    console.print("[bold blue]Step 2/2: Training model...[/bold blue]\n")

    _train_cells(
        data_dir=data_dir,
        models_dir=models_dir,
        epochs=epochs,
        batch_size=batch_size,
        lr_init=lr_init,
        lr_end=lr_end,
        validation=validation,
        validation_split=validation_split,
        super_resolution=super_resolution,
        load_weights=load_weights,
        early_stopping_patience=early_stopping_patience,
        thermo=thermo,
        suffix=suffix,
    )


def _train_cells(
    data_dir: str,
    models_dir: str,
    epochs: int,
    batch_size: int,
    lr_init: float,
    lr_end: float,
    validation: bool,
    validation_split: float,
    super_resolution: int,
    load_weights: bool,
    early_stopping_patience: int,
    thermo: bool = False,
    suffix: str = "",
) -> None:
    """Train CELLS model (all cells at once)."""
    # Determine suffix if not provided
    # Baseline: no suffix (cells.weights.h5), Thermo: _thermo suffix
    if not suffix:
        suffix = "_thermo" if thermo else ""

    console.print(f"[dim]Data directory: {data_dir}[/dim]")
    console.print(f"[dim]Models directory: {models_dir}[/dim]")
    console.print(f"[dim]Epochs: {epochs}, Batch size: {batch_size}[/dim]")
    console.print(f"[dim]Learning rate: {lr_init} → {lr_end}[/dim]")
    console.print(f"[dim]Thermo: {thermo}, Suffix: {suffix}[/dim]\n")

    # Create trainer
    trainer = Trainer(
        data_dir=data_dir,
        model_type=ModelType.CELLS,
        problem_formulation=ProblemFormulation.CLASSIFICATION,
        models_dir=models_dir,
        thermo_dim=4 if thermo else 0,  # NEW: 4 thermo params (PBLH not available in GFS)
    )

    # Prepare data (all cells)
    console.print("[yellow]Preparing data...[/yellow]")
    X, Y = trainer.prepare_data(super_resolution=super_resolution)

    # Create model (all cells)
    console.print("[yellow]Creating model...[/yellow]")
    trainer.create_model(super_resolution=super_resolution, load_weights=load_weights)

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
        validation_split=validation_split,
        early_stopping_patience=early_stopping_patience,
    )

    # Save
    console.print("\n[yellow]Saving model...[/yellow]")
    trainer.save_weights(suffix=suffix)  # NEW: use suffix for versioning

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
def evaluate(
    year: int = typer.Option(2025, "--year", "-y", help="Year to use for testing"),
    data_dir: str = typer.Option(None, "--data-dir", "-d", help="Directory containing PKL files"),
    models_dir: str = typer.Option(None, "--models-dir", help="Directory with model weights"),
    threshold: float = typer.Option(0.5, "--threshold", "-t", help="Decision threshold for classification"),
    output: str = typer.Option("flown", "--output", "-o", help="Output to evaluate: 'flown' (default), 'crossed' (XC)"),
) -> None:
    """
    Evaluate trained model performance on a specific test year.

    Calculates:
    - Confusion Matrix (True/False Positives/Negatives)
    - Classification Report (Precision, Recall, F1)
    - ROC-AUC Score

    Available outputs for CELLS:
    - flown: Basic flyability (default)
    - crossed: XC cross-country potential

    Example:
        pyparaglide evaluate --year 2025 --threshold 0.7 --output crossed
    """
    import numpy as np
    from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
    from rich.panel import Panel

    settings = get_settings()

    # Use defaults from settings if not specified
    if data_dir is None:
        data_dir = settings.pkl_dir
    if models_dir is None:
        models_dir = settings.models_dir

    console.print(f"[bold cyan]Evaluating CELLS model on year {year}[/bold cyan]")
    console.print(f"[dim]Data: {data_dir}[/dim]")
    console.print(f"[dim]Models: {models_dir}[/dim]")
    console.print(f"[dim]Threshold: {threshold}[/dim]\n")

    # Initialize Trainer
    trainer = Trainer(
        data_dir=data_dir,
        model_type=ModelType.CELLS,
        problem_formulation=ProblemFormulation.CLASSIFICATION,
        models_dir=models_dir,
    )

    # Find indices for the test year
    # dataset.meteo_days contains all available days
    test_indices = [
        i for i, d in enumerate(trainer.dataset.meteo_days)
        if d.year == year
    ]

    if not test_indices:
        years = sorted(list(set(d.year for d in trainer.dataset.meteo_days)))
        console.print(f"[red]Error: No data found for year {year}[/red]")
        console.print(f"Available years: {years}")
        raise typer.Exit(1)

    console.print(f"[green]Found {len(test_indices)} days for testing[/green]")

    # Map output name to index
    # CELLS: [flown, crossed]
    output_map = {
        "flown": 0,
        "crossed": 1,
    }

    # Validate output parameter
    if output not in output_map:
        console.print(f"[red]Invalid output: {output}[/red]")
        console.print(f"Valid options: {list(output_map.keys())}")
        raise typer.Exit(1)

    output_idx = output_map[output]
    console.print(f"[dim]Evaluating output: {output} (index {output_idx})[/dim]\n")

    # Prepare data
    console.print("[yellow]Preparing data (this might take a moment)...[/yellow]")

    # Use all cells for CELLS model
    cells_to_use = list(range(trainer.nb_cells))

    # Prepare data for selected cells
    X_full, Y_full = trainer.prepare_data(cells=cells_to_use)

    # Filter for test year
    # X is [Date, Dow, (Mountain), Other, Rain, Wind]
    X_test = [x[test_indices] for x in X_full]

    # Y is list of outputs. For CELLS: [Flown, Crossed]
    # We use the selected output
    Y_test_raw = [y[test_indices] for y in Y_full]

    # Check if output index exists
    if output_idx >= len(Y_test_raw):
        console.print(f"[red]Error: Output index {output_idx} not available (model has {len(Y_test_raw)} outputs)[/red]")
        raise typer.Exit(1)

    # Flatten for global evaluation
    # Y_test_raw[output_idx] shape: (nb_days, nb_cells * super_res^2 * nb_alts) OR (nb_days, nb_spots)
    y_true = Y_test_raw[output_idx].flatten()

    # Load Model & Predict
    console.print("[yellow]Loading model and predicting...[/yellow]")
    trainer.create_model(cells=cells_to_use, load_weights=True)

    preds = trainer.model.predict(X_test, verbose=1)

    # Check if predictions have the selected output
    if output_idx >= len(preds):
        console.print(f"[red]Error: Prediction output index {output_idx} not available[/red]")
        raise typer.Exit(1)

    # Flatten predictions
    y_pred_prob = preds[output_idx].flatten()

    # Calculate Metrics
    y_pred_bool = (y_pred_prob >= threshold).astype(int)
    y_true_int = y_true.astype(int)

    # Confusion Matrix - handle case where all labels are the same
    unique_labels = sorted(set(y_true_int) | set(y_pred_bool))
    if len(unique_labels) < 2:
        console.print(f"\n[yellow]Warning: Only one class present in data ({unique_labels})[/yellow]")
        console.print("[yellow]Cannot compute confusion matrix and classification metrics[/yellow]")

        # Basic stats
        total_samples = len(y_true_int)
        pos_pred = int(y_pred_bool.sum())
        neg_pred = total_samples - pos_pred
        pos_true = int(y_true_int.sum())
        neg_true = total_samples - pos_true

        console.print(f"\n[bold]Basic Statistics[/bold]")
        console.print(f"Total samples: {total_samples:,}")
        console.print(f"Actual positives: [cyan]{pos_true:,}[/cyan], negatives: [cyan]{neg_true:,}[/cyan]")
        console.print(f"Predicted positives: [cyan]{pos_pred:,}[/cyan], negatives: [cyan]{neg_pred:,}[/cyan]")
        return

    # Both classes present - compute full metrics
    cm = confusion_matrix(y_true_int, y_pred_bool, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    
    # ROC AUC
    try:
        auc = roc_auc_score(y_true_int, y_pred_prob)
        auc_str = f"{auc:.4f}"
    except ValueError:
        auc_str = "N/A"

    # Display Results
    console.print(f"\n[bold]RESULTS (Threshold: {threshold})[/bold]")
    
    # Confusion Matrix Table
    cm_table = Table(title="Confusion Matrix", show_header=True)
    cm_table.add_column("Type", style="cyan")
    cm_table.add_column("Count", style="bold")
    cm_table.add_column("Meaning", style="dim")
    
    cm_table.add_row("True Negatives", f"[green]{tn:,}[/green]", "Correctly predicted NOT flyable")
    cm_table.add_row("False Positives", f"[red]{fp:,}[/red]", "Predicted flyable, but wasn't (Wasted trip)")
    cm_table.add_row("False Negatives", f"[yellow]{fn:,}[/yellow]", "Predicted NOT flyable, but was (Missed day)")
    cm_table.add_row("True Positives", f"[green]{tp:,}[/green]", "Correctly predicted flyable")
    
    console.print(cm_table)
    
    # Metrics Panel
    report = classification_report(
        y_true_int, y_pred_bool, target_names=['Not Flyable', 'Flyable'], zero_division=0
    )
    summary = f"ROC AUC Score: [bold magenta]{auc_str}[/bold magenta]\n\n{report}"
    
    console.print(Panel(summary, title="[bold]Detailed Metrics[/bold]", border_style="cyan"))


@app.command()
def forecast(
    date: str = typer.Option(None, "--date", "-d", help="Target date (YYYY-MM-DD, default: today)"),
    models_dir: str = typer.Option(None, "--models-dir", help="Directory with model weights"),
    grib_dir: str = typer.Option(None, "--grib-dir", "-g", help="Directory containing GRIB files"),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Output directory for forecasts"),
    bbox: str = typer.Option(None, "--bbox", "-b", help="Bounding box: lat_min,lat_max,lon_min,lon_max"),
) -> None:
    """
    Generate paragliding flyability forecast using CELLS model.

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
        # Use parent of gfs_dir (data/gfs) instead of gfs_dir (data/gfs/anl)
        # Forecast files should be in data/gfs/forecasts/, not data/gfs/anl/forecasts/
        gfs_path = Path(settings.gfs_dir)
        grib_dir = str(gfs_path.parent / "forecasts" if gfs_path.parent.name == "gfs" else gfs_path / "forecasts")
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

    console.print(f"[bold cyan]Generating CELLS forecast[/bold cyan]")
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
    forecaster = Forecaster(models_dir, ProblemFormulation.CLASSIFICATION)

    console.print("[yellow]Running prediction...[/yellow]")
    results = forecaster.predict_day(grib_files, target_date, tuple(float(x) for x in bbox.split(",")))

    # Save results
    output_path = Path(output_dir) / f"{target_date.isoformat()}_cells_forecast.json"
    console.print("[yellow]Saving results...[/yellow]")
    forecaster.save_forecast(results, output_path)

    console.print(f"\n[green]Forecast complete![/green]")
    console.print(f"Output: [cyan]{output_path}[/cyan]")


@app.command()
def forecast_ab_test(
    date: str = typer.Option(None, "--date", "-d", help="Target date (YYYY-MM-DD, default: today)"),
    models_dir: str = typer.Option(None, "--models-dir", help="Directory with model weights"),
    grib_dir: str = typer.Option(None, "--grib-dir", "-g", help="Directory containing GRIB files"),
    output_dir: str = typer.Option(None, "--output-dir", "-o", help="Output directory for A/B test results"),
    bbox: str = typer.Option(None, "--bbox", "-b", help="Bounding box: lat_min,lat_max,lon_min,lon_max"),
    baseline_variant: str = typer.Option("", "--baseline", help="Baseline model variant (default: empty for cells.weights.h5)"),
    thermo_variant: str = typer.Option("thermo", "--thermo", help="Thermo model variant (default: thermo for cells_thermo.weights.h5)"),
) -> None:
    """
    Generate A/B test forecast comparing baseline and thermo-enhanced models.

    Example:
        pyparaglide forecast-ab-test --date 2025-06-15
    """
    import datetime as dt
    import json

    import numpy as np  # NEW: needed for np.mean in comparison

    settings = get_settings()

    # Use defaults from settings if not specified
    if models_dir is None:
        models_dir = settings.models_dir
    if grib_dir is None:
        # Use parent of gfs_dir (data/gfs) instead of gfs_dir (data/gfs/anl)
        # Forecast files should be in data/gfs/forecasts/, not data/gfs/anl/forecasts/
        gfs_path = Path(settings.gfs_dir)
        grib_dir = str(gfs_path.parent / "forecasts" if gfs_path.parent.name == "gfs" else gfs_path / "forecasts")
    if output_dir is None:
        output_dir = str(Path(settings.output_dir) / "ab_tests")
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

    console.print(f"[bold cyan]A/B Testing: baseline vs thermo[/bold cyan]")
    console.print(f"[dim]Date: {target_date.isoformat()}[/dim]")
    console.print(f"[dim]Models: {models_dir}[/dim]")
    console.print(f"[dim]Baseline: {baseline_variant}, Thermo: {thermo_variant}[/dim]\n")

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

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Generate baseline forecast
    console.print("[yellow]Loading baseline model...[/yellow]")
    forecaster_baseline = Forecaster(models_dir, ProblemFormulation.CLASSIFICATION, model_variant=baseline_variant)

    console.print("[yellow]Running baseline prediction...[/yellow]")
    results_baseline = forecaster_baseline.predict_day(grib_files, target_date, tuple(float(x) for x in bbox.split(",")))

    # Generate thermo forecast
    console.print("[yellow]Loading thermo model...[/yellow]")
    forecaster_thermo = Forecaster(models_dir, ProblemFormulation.CLASSIFICATION, model_variant=thermo_variant)

    console.print("[yellow]Running thermo prediction...[/yellow]")
    results_thermo = forecaster_thermo.predict_day(grib_files, target_date, tuple(float(x) for x in bbox.split(",")))

    # Compare results
    # Extract flyability values from predictions list
    baseline_flyability = [p["flyability"] for p in results_baseline.get("predictions", [])]
    thermo_flyability = [p["flyability"] for p in results_thermo.get("predictions", [])]

    baseline_mean = np.mean(baseline_flyability) if baseline_flyability else 0.0
    thermo_mean = np.mean(thermo_flyability) if thermo_flyability else 0.0
    difference = thermo_mean - baseline_mean

    comparison = {
        "date": target_date.isoformat(),
        "baseline_variant": baseline_variant,
        "thermo_variant": thermo_variant,
        "baseline_mean_flyability": float(baseline_mean),
        "thermo_mean_flyability": float(thermo_mean),
        "difference_mean_flyability": float(difference),
        "baseline_results": results_baseline,
        "thermo_results": results_thermo,
    }

    # Save results
    output_file = output_path / f"ab_test_{target_date.isoformat()}.json"
    console.print("[yellow]Saving A/B test results...[/yellow]")
    with open(output_file, "w") as f:
        json.dump(comparison, f, indent=2)

    console.print(f"\n[green]A/B test complete![/green]")
    console.print(f"Baseline mean flyability: [cyan]{baseline_mean:.4f}[/cyan]")
    console.print(f"Thermo mean flyability: [cyan]{thermo_mean:.4f}[/cyan]")
    console.print(f"Difference: [cyan]{difference:+.4f}[/cyan]")
    console.print(f"Output: [cyan]{output_file}[/cyan]")


@app.command()
def download(
    dates: str = typer.Option(None, "--dates", "-D", help="Date ranges (format: YYYY-MM-DD:YYYY-MM-DD,...). Overrides .env TRAINING_DATES"),
    start_date: str = typer.Option(None, "--start", "-s", help="Start date (YYYY-MM-DD). Shorthand for single range. Use --dates for multiple ranges"),
    end_date: str = typer.Option(None, "--end", "-e", help="End date (YYYY-MM-DD). Must be used with --start"),
    data_dir: str = typer.Option(None, "--data-dir", "-d", help="Output directory for GRIB files"),
    hours: str = typer.Option("0,6,12,18", "--hours", "-H", help="UTC hours to download (comma-separated)"),
    workers: int = typer.Option(1, "--workers", "-w", help="Number of parallel download workers"),
    filter: bool = typer.Option(False, "--filter", help="Filter GRIB files to reduce size by ~50%"),
    skip_gfs: bool = typer.Option(False, "--skip-gfs", help="Skip GFS download"),
    skip_elevation: bool = typer.Option(False, "--skip-elevation", help="Skip elevation download"),
) -> None:
    """
    Download GFS Analysis and Elevation data.

    [bold yellow]New:[/bold yellow] Use subcommands for more control:
      - [cyan]pyparaglide dl analysis[/cyan] for GFS Analysis
      - [cyan]pyparaglide dl forecast[/cyan] for GFS Forecast
      - [cyan]pyparaglide dl elevation[/cyan] for elevation

    Downloads historical GFS Analysis GRIB files and/or SRTM elevation data.

    Date sources (in order of priority):
    1. --dates "2024-06-01:2024-08-31,2025-06-01:2025-08-31" (multiple ranges)
    2. --start/--end (single range)
    3. TRAINING_DATES from .env (default)

    Elevation source is configured via ELEVATION_SOURCE in .env:
    - SRTM1: 30 arc-second resolution (~900m)
    - SRTM3: 90 arc-second resolution (~2.7km), default

    Examples:
        # Download both GFS and elevation data
        pyparaglide download

        # Download only GFS (skip elevation)
        pyparaglide download --skip-elevation

        # Download only elevation (skip GFS)
        pyparaglide download --skip-gfs

        # Single range with --dates
        pyparaglide download --dates 2024-06-01:2024-08-31

        # Multiple ranges
        pyparaglide download --dates "2024-06-01:2024-08-31,2025-06-01:2025-08-31"

        # Legacy format (single range)
        pyparaglide download --start 2024-06-01 --end 2024-08-31

        # With options
        pyparaglide download --workers 4 --filter
    """
    from pyparaglide.cli._download_helpers import download_analysis_impl, download_elevation_impl

    # Call helpers
    if not skip_gfs:
        download_analysis_impl(
            dates=dates,
            start_date=start_date,
            end_date=end_date,
            data_dir=data_dir,
            hours=hours,
            workers=workers,
            filter_grib=filter,
        )

    if not skip_elevation:
        download_elevation_impl(
            data_dir=data_dir,
            bbox=None,
        )

    if skip_gfs and skip_elevation:
        console.print("[yellow]Warning: Both --skip-gfs and --skip-elevation specified, nothing to download[/yellow]")
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
    rebuild_cache: bool = typer.Option(False, "--rebuild-cache", help="Force rebuild of GRIB cache (re-extract all files)"),
    skip_cache: bool = typer.Option(False, "--skip-cache", help="Disable GRIB caching (always re-extract)"),
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
        use_cache=not skip_cache,
        rebuild_cache=rebuild_cache,
    )

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Cells: {stats['cells']}")
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
