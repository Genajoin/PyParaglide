"""Common CLI utilities for Paraglidable scripts."""

import typer
from typing import Optional, Any, Dict
from .db import Db, connect_db
from .config import Config

# Main typer app - used by commands to register subcommands
app = typer.Typer(help="Paraglidable CLI - unified interface for all scripts")

# Global state for config and database connections
_state: Dict[str, Any] = {"config": None, "db": None}


def get_config() -> Config:
    """Get or create global Config instance."""
    if _state["config"] is None:
        _state["config"] = Config()
    return _state["config"]


def get_db(db_url: Optional[str] = None) -> Db:
    """Get or create global Db connection."""
    if _state["db"] is None:
        cfg = get_config()
        url = db_url or cfg.db_url
        _state["db"] = connect_db(url)
    return _state["db"]


def close_db() -> None:
    """Close global Db connection if open."""
    if _state["db"] is not None:
        _state["db"].close()
        _state["db"] = None


# Common typer option factories
def db_url_option(default: Optional[str] = None) -> typer.Option:
    """Create --db-url option with default from config."""
    return typer.Option(
        default,
        "--db-url",
        help="PostgreSQL connection URL (overrides IGC_DB_URL env var)",
    )


def source_option(default: str = "skygr") -> typer.Option:
    """Create --source option."""
    return typer.Option(
        default,
        "--source",
        help="Flight source name in database",
    )


def years_option(default: str = "") -> typer.Option:
    """Create --years option for year ranges."""
    return typer.Option(
        default,
        "--years",
        help='Year range (e.g., "2020-2025" or "2020,2021,2022")',
    )
