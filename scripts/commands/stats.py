"""Statistics commands for flight data."""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from lib.cli import app, get_db

# Import functions from original flights_monthly_stats.py
from flights_monthly_stats import print_stats, get_monthly_stats, get_quarterly_stats, get_yearly_stats

# Create sub-app for stats commands
stats_app = typer.Typer(help="Flight statistics")


@stats_app.command("monthly")
def stats_monthly(
    group_by: str = typer.Option("month", "--group-by", help="Aggregation level: month, quarter, year"),
    source: str = typer.Option(None, "--source", help="Filter by source (skygr, paraplan, etc.)"),
    top: int = typer.Option(10, "--top", help="Number of top bbox regions to show"),
):
    """Show flight statistics by time period."""
    db = get_db()
    print_stats(db, group_by=group_by, source=source, bbox_limit=top)
    db.close()


@stats_app.command("yearly")
def stats_yearly(
    source: str = typer.Option(None, "--source", help="Filter by source"),
    top: int = typer.Option(10, "--top", help="Number of top bbox regions"),
):
    """Show yearly flight statistics."""
    db = get_db()
    print_stats(db, group_by="year", source=source, bbox_limit=top)
    db.close()


@stats_app.command("quarterly")
def stats_quarterly(
    source: str = typer.Option(None, "--source", help="Filter by source"),
    top: int = typer.Option(10, "--top", help="Number of top bbox regions"),
):
    """Show quarterly flight statistics."""
    db = get_db()
    print_stats(db, group_by="quarter", source=source, bbox_limit=top)
    db.close()


def register(parent_app: typer.Typer) -> None:
    """Register stats subcommands with parent app."""
    parent_app.add_typer(stats_app, name="stats")
