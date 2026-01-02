"""IGC ingestion commands for sky.gr source."""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from lib.cli import app, get_db, get_config, db_url_option, source_option, years_option

# Import functions from original igc_ingest_skygr.py
from igc_ingest_skygr import (
    crawl_list,
    build_year_list_url,
    parse_years,
    process_links,
    process_flights,
    reparse_existing,
    print_stats,
    RateLimiter,
)

# Create sub-app for ingest commands
ingest_app = typer.Typer(help="IGC ingestion from sky.gr")


@ingest_app.command("list")
def ingest_list(
    years: str = years_option(""),
    source: str = source_option("skygr"),
    max_pages: int = typer.Option(3, "--max-pages", help="Maximum pages to crawl"),
    list_only: bool = typer.Option(False, "--list-only", help="Only collect links, don't download"),
):
    """Collect IGC flight links from sky.gr."""
    db = get_db()

    year_list: list[int] = []
    if years:
        year_list = parse_years(years)

    base_url = "https://www.sky.gr/"
    list_url = "https://www.sky.gr/modules.php?lng=english&name=leonardo&op=list_flights&sortOrder=DATE"

    # Create session and limiter (matching original behavior)
    import requests
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; igc-downloader/0.2; +https://example.com)",
        "Accept-Encoding": "gzip, deflate",
    })
    limiter = RateLimiter(1.0, 1.5)

    total = 0
    if year_list:
        for year in year_list:
            list_url_year = build_year_list_url(year, list_url)
            list_key = list_url_year  # Simplified
            count = crawl_list(
                db=db,
                session=session,
                base_url=base_url,
                list_url=list_url_year,
                list_key=list_key,
                max_pages=max_pages,
                limiter=limiter,
                timeout=30,
                retries=3,
                source=source,
                year=year,
            )
            print(f"list {year} crawl: {count} flights discovered")
            total += count
        print(f"list crawl total: {total} flights discovered")
    else:
        count = crawl_list(
            db=db,
            session=session,
            base_url=base_url,
            list_url=list_url,
            list_key=list_url,
            max_pages=max_pages,
            limiter=limiter,
            timeout=30,
            retries=3,
            source=source,
            year=None,
        )
        print(f"list crawl: {count} flights discovered")

    db.close()


@ingest_app.command("links")
def ingest_links(
    max_links: int = typer.Option(0, "--max-links", help="Maximum links to process (0=unlimited)"),
    source: str = source_option("skygr"),
):
    """Extract IGC URLs from flight pages."""
    db = get_db()

    import requests
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; igc-downloader/0.2; +https://example.com)",
        "Accept-Encoding": "gzip, deflate",
    })
    limiter = RateLimiter(1.0, 1.5)

    base_url = "https://www.sky.gr/"

    count = process_links(
        db=db,
        session=session,
        base_url=base_url,
        timeout=30,
        retries=3,
        limiter=limiter,
        source=source,
        max_retries=5,
        max_links=max_links if max_links > 0 else None,
    )
    print(f"links queued: {count} flights")

    db.close()


@ingest_app.command("download")
def ingest_download(
    max_flights: int = typer.Option(200, "--max-flights", help="Maximum flights to download (0=unlimited)"),
    min_delay: float = typer.Option(1.0, "--min-delay", help="Minimum delay between requests"),
    max_delay: float = typer.Option(1.5, "--max-delay", help="Maximum delay between requests"),
    source: str = source_option("skygr"),
    out_dir: str = typer.Option("data/igc/skygr", "--out-dir", help="Output directory"),
):
    """Download IGC files."""
    db = get_db()

    import requests
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (compatible; igc-downloader/0.2; +https://example.com)",
        "Accept-Encoding": "gzip, deflate",
    })
    limiter = RateLimiter(min_delay, max_delay)

    base_url = "https://www.sky.gr/"

    count = process_flights(
        db=db,
        session=session,
        base_url=base_url,
        out_dir=out_dir,
        timeout=30,
        retries=3,
        limiter=limiter,
        source=source,
        max_retries=5,
        max_flights=max_flights if max_flights > 0 else None,
    )
    print(f"downloaded+parsed: {count} flights")

    db.close()


@ingest_app.command("reparse")
def ingest_reparse(
    max_files: int = typer.Option(0, "--max-files", help="Maximum files to reparse (0=unlimited)"),
    source: str = source_option("skygr"),
):
    """Reparse existing IGC files with enhanced metadata."""
    db = get_db()

    count = reparse_existing(
        db=db,
        source=source,
        max_files=max_files if max_files > 0 else None,
    )
    print(f"reparse updated: {count} flights")

    db.close()


@ingest_app.command("stats")
def ingest_stats(
    source: str = source_option("skygr"),
):
    """Show ingestion statistics."""
    db = get_db()
    print_stats(db, source)
    db.close()


def register(parent_app: typer.Typer) -> None:
    """Register ingest subcommands with parent app."""
    parent_app.add_typer(ingest_app, name="ingest")
