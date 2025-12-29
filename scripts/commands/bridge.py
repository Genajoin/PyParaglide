"""Bridge server commands for browser extension integration."""

import sys
import os
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from lib.cli import app, get_config

# Import from igc_bridge_server.py
from igc_bridge_server import main as bridge_main

bridge_app = typer.Typer(help="IGC bridge server for browser extension")


@bridge_app.command("serve")
def bridge_serve(
    host: str = typer.Option("0.0.0.0", "--host", help="Host to bind to"),
    port: int = typer.Option(8080, "--port", help="Port to bind to"),
):
    """Start HTTP server for browser extension to submit flight links."""
    cfg = get_config()
    db_url = cfg.db_url

    class Args:
        def __init__(self):
            self.db_url = db_url
            self.host = host
            self.port = port

    args = Args()
    exit_code = bridge_main(args)
    raise SystemExit(exit_code)


def register(parent_app: typer.Typer) -> None:
    """Register bridge subcommands with parent app."""
    parent_app.add_typer(bridge_app, name="bridge")
