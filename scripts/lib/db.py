"""Database connection and schema management for Paraglidable scripts."""

from typing import Any, Tuple, Optional

try:
    import psycopg
except Exception:
    psycopg = None


PG_SCHEMA = """
CREATE TABLE IF NOT EXISTS flights (
    id BIGSERIAL PRIMARY KEY,
    source TEXT NOT NULL,
    flight_id TEXT NOT NULL,
    show_url TEXT,
    igc_url TEXT,
    igc_path TEXT,
    file_size INTEGER,
    sha256 TEXT,
    duplicate_of TEXT,
    status TEXT NOT NULL,
    error_msg TEXT,
    retry_count INTEGER NOT NULL DEFAULT 0,
    downloaded_at TEXT,
    updated_at TEXT,
    flight_date TEXT,
    pilot TEXT,
    glider TEXT,
    glider_class TEXT,
    takeoff_lat DOUBLE PRECISION,
    takeoff_lon DOUBLE PRECISION,
    takeoff_name TEXT,
    duration_sec INTEGER,
    distance_km DOUBLE PRECISION,
    score DOUBLE PRECISION,
    track_points INTEGER,
    has_baro_alt INTEGER,
    has_gps_alt INTEGER
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_flights_source_id ON flights(source, flight_id);
CREATE INDEX IF NOT EXISTS idx_flights_status ON flights(status);
CREATE INDEX IF NOT EXISTS idx_flights_flight_date ON flights(flight_date);
CREATE INDEX IF NOT EXISTS idx_flights_sha256 ON flights(sha256);

CREATE TABLE IF NOT EXISTS crawl_state (
    source TEXT PRIMARY KEY,
    list_url TEXT,
    next_list_url TEXT,
    last_seen_flight_id TEXT,
    updated_at TEXT
);
CREATE TABLE IF NOT EXISTS crawl_state_list (
    source TEXT NOT NULL,
    list_key TEXT NOT NULL,
    list_url TEXT,
    next_list_url TEXT,
    last_seen_flight_id TEXT,
    year INTEGER,
    updated_at TEXT,
    PRIMARY KEY (source, list_key)
);
CREATE INDEX IF NOT EXISTS idx_crawl_state_list_year ON crawl_state_list(year);
"""


class Db:
    """PostgreSQL database wrapper with ? placeholder support."""

    def __init__(self, conn: Any) -> None:
        self.conn = conn

    def execute(self, sql: str, params: Tuple[Any, ...] = ()) -> Any:
        """Execute SQL with ? placeholders (converted to %s for PostgreSQL)."""
        sql = sql.replace("?", "%s")
        return self.conn.execute(sql, params)

    def executescript(self, script: str) -> None:
        """Execute multi-statement SQL script."""
        for stmt in script.split(";"):
            stmt = stmt.strip()
            if stmt:
                self.conn.execute(stmt)

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


def connect_db(db_url: str) -> Db:
    """Connect to PostgreSQL database.

    Args:
        db_url: PostgreSQL connection URL (e.g., "postgresql://user:pass@host:5432/db")

    Returns:
        Db wrapper instance

    Raises:
        RuntimeError: If psycopg is not installed
    """
    if psycopg is None:
        raise RuntimeError("psycopg is required. Install with: pip install psycopg[binary]")
    conn = psycopg.connect(db_url)
    return Db(conn)


def ensure_db(db: Db) -> None:
    """Ensure database schema exists."""
    db.executescript(PG_SCHEMA)
    db.commit()
    _ensure_columns(db)


def _ensure_columns(db: Db) -> None:
    """Add legacy columns if missing (for backwards compatibility)."""
    db.execute("ALTER TABLE flights ADD COLUMN IF NOT EXISTS duplicate_of TEXT")
    db.execute("ALTER TABLE flights ADD COLUMN IF NOT EXISTS updated_at TEXT")
    db.commit()
