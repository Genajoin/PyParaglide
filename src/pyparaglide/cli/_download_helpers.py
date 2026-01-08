"""
Helper functions for download commands.

Extracted to enable both flat and subcommand interfaces.
"""

import datetime as dt

from rich.console import Console
import typer

from pyparaglide.config import get_settings, parse_date_ranges
from pyparaglide.downloads import GFSDownloader, ElevationDownloader

console = Console()


def download_analysis_impl(
    dates: str | None,
    start_date: str | None,
    end_date: str | None,
    data_dir: str | None,
    hours: str,
    workers: int,
    filter_grib: bool,
) -> None:
    """
    Implementation for GFS Analysis download.

    Shared by both flat command and subcommand.

    Args:
        dates: Date ranges string (format: YYYY-MM-DD:YYYY-MM-DD,...)
        start_date: Start date string (YYYY-MM-DD)
        end_date: End date string (YYYY-MM-DD)
        data_dir: Output directory for GRIB files
        hours: UTC hours to download (comma-separated string)
        workers: Number of parallel download workers
        filter_grib: Filter GRIB files to reduce size
    """
    settings = get_settings()

    if data_dir is None:
        data_dir = settings.gfs_dir

    # Parse hours
    hour_list = [int(h.strip()) for h in hours.split(",")]

    console.print(f"[bold cyan]Downloading GFS Analysis data[/bold cyan]\n")

    # Determine date ranges (priority: --dates > --start/--end > .env TRAINING_DATES)
    if dates:
        try:
            date_ranges = parse_date_ranges(dates)
        except ValueError as e:
            console.print(f"[red]{e}[/red]")
            raise typer.Exit(1)
    elif start_date and end_date:
        try:
            date_ranges = parse_date_ranges(f"{start_date}:{end_date}")
        except ValueError:
            console.print(f"[red]Invalid date format[/red]")
            console.print("Use format: [cyan]YYYY-MM-DD[/cyan]")
            raise typer.Exit(1)
    elif start_date or end_date:
        console.print("[red]Error: --start and --end must be used together, or use --dates instead[/red]")
        raise typer.Exit(1)
    else:
        str_ranges = settings.parse_training_dates()
        if not str_ranges:
            console.print("[red]Error: No date ranges found. Specify --dates, --start/--end, or set TRAINING_DATES in .env[/red]")
            raise typer.Exit(1)
        date_ranges = [
            (
                dt.datetime.strptime(start, "%Y-%m-%d").date(),
                dt.datetime.strptime(end, "%Y-%m-%d").date(),
            )
            for start, end in str_ranges
        ]

    # Create downloader
    downloader = GFSDownloader(
        data_dir=data_dir,
        hours=hour_list,
        workers=workers,
        filter_grib=filter_grib,
    )

    # Download all ranges
    total_stats = {"downloaded": 0, "skipped": 0, "failed": 0, "total_mb": 0.0}

    for start, end in date_ranges:
        console.print(f"\n[yellow]Processing range: {start} to {end}[/yellow]")
        stats = downloader.download_range(start, end)

        total_stats["downloaded"] += stats["downloaded"]
        total_stats["skipped"] += stats["skipped"]
        total_stats["failed"] += stats["failed"]
        total_stats["total_mb"] += stats["total_mb"]

    # Print summary
    console.print(f"\n[bold]GFS Download Summary:[/bold]")
    console.print(f"  Downloaded: {total_stats['downloaded']} files ({total_stats['total_mb']:.1f} MB)")
    console.print(f"  Skipped: {total_stats['skipped']} files")
    console.print(f"  Failed: {total_stats['failed']} files")

    if total_stats["failed"] > 0:
        raise typer.Exit(1)


def download_elevation_impl(data_dir: str | None, bbox: str | None) -> None:
    """
    Implementation for elevation download.

    Shared by both flat command and subcommand.

    Args:
        data_dir: Output directory for elevation files
        bbox: Bounding box string in GeoJSON format (lon_min,lat_min,lon_max,lat_max)
    """
    settings = get_settings()

    console.print(f"\n[bold cyan]Downloading Elevation data[/bold cyan]\n")

    if data_dir is None:
        data_dir = settings.elevation_dir

    if bbox is None:
        bbox_tuple = settings.parse_bbox()
    else:
        # Parse bbox from string
        parts = [float(x.strip()) for x in bbox.split(",")]
        if len(parts) != 4:
            console.print(f"[red]Invalid bbox format: {bbox}[/red]")
            console.print("Expected: lon_min,lat_min,lon_max,lat_max (GeoJSON format)")
            raise typer.Exit(1)
        bbox_tuple = (parts[0], parts[1], parts[2], parts[3])

    downloader = ElevationDownloader(
        data_dir=data_dir,
        bbox=bbox_tuple,
        product=settings.elevation_source,
    )

    stats = downloader.download()

    console.print(f"\n[bold]Elevation Download Summary:[/bold]")
    console.print(f"  Source: {stats['source']}")
    console.print(f"  BBox: {stats['bbox']}")
    console.print(f"  Output: {stats['output_size_mb']:.1f} MB")
