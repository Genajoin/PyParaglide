#!/usr/bin/env python3
import argparse
import json
import os
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

try:
    import psycopg  # type: ignore
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


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def log(msg: str) -> None:
    print(f"{utcnow()} {msg}", flush=True)


class Db:
    """PostgreSQL database wrapper."""

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


def connect_db(db_url: str, db_path: str = None) -> Db:
    """Connect to PostgreSQL database.

    Args:
        db_url: PostgreSQL connection URL
        db_path: Ignored (kept for backwards compatibility)

    Returns:
        Db wrapper instance
    """
    if psycopg is None:
        raise RuntimeError("psycopg is required. Install with: pip install psycopg[binary]")
    conn = psycopg.connect(db_url)
    return Db(conn)


def ensure_db(db: Db) -> None:
    """Ensure database schema exists."""
    db.executescript(PG_SCHEMA)
    db.commit()
    ensure_columns(db)


def ensure_columns(db: Db) -> None:
    """Add legacy columns if missing (for backwards compatibility)."""
    db.execute("ALTER TABLE flights ADD COLUMN IF NOT EXISTS duplicate_of TEXT")
    db.execute("ALTER TABLE flights ADD COLUMN IF NOT EXISTS updated_at TEXT")
    db.commit()


def db_upsert_flight(db: Db, source: str, flight_id: str, show_url: Optional[str]) -> None:
    now = utcnow()
    db.execute(
        """
        INSERT INTO flights (source, flight_id, show_url, status, updated_at)
        VALUES (?, ?, ?, 'new', ?)
        ON CONFLICT(source, flight_id) DO UPDATE SET
            show_url=COALESCE(excluded.show_url, flights.show_url),
            updated_at=excluded.updated_at
        """,
        (source, flight_id, show_url, now),
    )


def db_update_igc_url(db: Db, source: str, flight_id: str, igc_url: str) -> None:
    now = utcnow()
    db.execute(
        """
        UPDATE flights
        SET igc_url=?,
            status=CASE
                WHEN status IN ('downloaded','parsed','duplicate') THEN status
                ELSE 'queued'
            END,
            error_msg=NULL,
            updated_at=?
        WHERE source=? AND flight_id=?
        """,
        (igc_url, now, source, flight_id),
    )


def db_mark_failed(db: Db, source: str, flight_id: str, error_msg: str) -> None:
    now = utcnow()
    db.execute(
        """
        UPDATE flights
        SET status='failed', error_msg=?, retry_count=retry_count+1, updated_at=?
        WHERE source=? AND flight_id=?
        """,
        (error_msg[:500], now, source, flight_id),
    )


def db_update_download(
    db: Db,
    source: str,
    flight_id: str,
    igc_url: Optional[str],
    igc_path: Optional[str],
    file_size: Optional[int],
    downloaded_at: Optional[str],
) -> None:
    db.execute(
        """
        UPDATE flights
        SET igc_url=COALESCE(?, igc_url),
            igc_path=COALESCE(?, igc_path),
            file_size=COALESCE(?, file_size),
            downloaded_at=COALESCE(?, downloaded_at),
            updated_at=COALESCE(?, updated_at),
            status=CASE
                WHEN status IN ('parsed','duplicate') THEN status
                ELSE 'downloaded'
            END,
            error_msg=NULL
        WHERE source=? AND flight_id=?
        """,
        (igc_url, igc_path, file_size, downloaded_at, downloaded_at, source, flight_id),
    )


def extract_flight_id(url: Optional[str]) -> Optional[str]:
    if not url:
        return None
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    for key in ("flightID", "flightId", "flight_id"):
        if key in qs and qs[key]:
            return qs[key][0]
    match = re.search(r"/flights?/(\d+)", parsed.path)
    if match:
        return match.group(1)
    return None


def resolve_flight_id(show_url: Optional[str], igc_url: Optional[str]) -> Optional[str]:
    flight_id = extract_flight_id(show_url)
    if flight_id:
        return flight_id
    return extract_flight_id(igc_url)


def parse_json_body(handler: BaseHTTPRequestHandler) -> Any:
    length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(length) if length > 0 else b""
    if not raw:
        return None
    return json.loads(raw.decode("utf-8"))


class BridgeHandler(BaseHTTPRequestHandler):
    server_version = "IGCBridge/0.1"

    def _send_json(self, status_code: int, payload: Dict[str, Any]) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "content-type")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.end_headers()

    def do_GET(self) -> None:
        if not self.path.startswith("/stats"):
            if self.path.startswith("/resolve/next"):
                self._handle_resolve_next()
                return
            if self.path.startswith("/downloads/next"):
                self._handle_downloads_next()
                return
            self._send_json(404, {"ok": False, "error": "not found"})
            return
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        source = qs.get("source", [self.server.default_source])[0]
        log(f"stats request: source={source}")
        db = connect_db(self.server.db_url, None)
        try:
            ensure_db(db)
            cur = db.execute("SELECT COUNT(*) FROM flights WHERE source=?", (source,))
            total = int(cur.fetchone()[0])
            cur = db.execute(
                "SELECT COUNT(*) FROM flights WHERE source=? AND igc_url IS NOT NULL",
                (source,),
            )
            with_igc = int(cur.fetchone()[0])
            cur = db.execute(
                """
                SELECT status, COUNT(*)
                FROM flights
                WHERE source=?
                GROUP BY status
                """,
                (source,),
            )
            status_counts = {row[0]: int(row[1]) for row in cur.fetchall()}
        finally:
            db.close()
        log(f"stats response: source={source} total={total} with_igc={with_igc}")
        self._send_json(
            200,
            {
                "ok": True,
                "source": source,
                "total": total,
                "with_igc_url": with_igc,
                "status_counts": status_counts,
            },
        )

    def do_POST(self) -> None:
        if self.path.startswith("/links"):
            self._handle_links_post()
            return
        if self.path.startswith("/resolve"):
            self._handle_resolve_post()
            return
        if self.path.startswith("/downloads"):
            self._handle_downloads_post()
            return
        self._send_json(404, {"ok": False, "error": "not found"})

    def _handle_resolve_next(self) -> None:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        source = qs.get("source", [self.server.default_source])[0]
        limit = int(qs.get("limit", ["50"])[0])
        include_failed = qs.get("include_failed", ["0"])[0] in ("1", "true", "yes")
        statuses = ("new", "failed") if include_failed else ("new",)
        log(f"resolve next: source={source} limit={limit} include_failed={include_failed}")

        db = connect_db(self.server.db_url, None)
        items = []
        try:
            ensure_db(db)
            placeholders = ", ".join(["?"] * len(statuses))
            cur = db.execute(
                f"""
                SELECT flight_id, show_url
                FROM flights
                WHERE source=?
                  AND show_url IS NOT NULL
                  AND igc_url IS NULL
                  AND status IN ({placeholders})
                ORDER BY id ASC
                LIMIT ?
                """,
                (source, *statuses, limit),
            )
            for row in cur.fetchall():
                items.append({"flight_id": row[0], "show_url": row[1]})
        finally:
            db.close()
        self._send_json(200, {"ok": True, "items": items})

    def _handle_resolve_post(self) -> None:
        try:
            items, default_source = self._parse_payload()
        except Exception:
            return

        processed = 0
        queued = 0
        failed = 0
        missing_flight_id = 0
        errors = 0
        log(f"resolve request: items={len(items)} default_source={default_source}")

        db = connect_db(self.server.db_url, None)
        try:
            ensure_db(db)
            for raw in items:
                if not isinstance(raw, dict):
                    errors += 1
                    continue
                source = raw.get("source") or default_source
                show_url = raw.get("show_url") or raw.get("showUrl")
                igc_url = raw.get("igc_url") or raw.get("igcUrl")
                flight_id = raw.get("flight_id") or raw.get("flightId")
                if not flight_id:
                    flight_id = resolve_flight_id(show_url, igc_url)
                if not flight_id:
                    missing_flight_id += 1
                    continue

                try:
                    db_upsert_flight(db, source, flight_id, show_url)
                    if igc_url:
                        db_update_igc_url(db, source, flight_id, igc_url)
                        queued += 1
                    else:
                        error_msg = raw.get("error_msg") or raw.get("errorMsg") or "no igc url found"
                        db_mark_failed(db, source, flight_id, error_msg)
                        failed += 1
                    processed += 1
                except Exception:
                    errors += 1
            db.commit()
        finally:
            db.close()

        log(
            "resolve response: "
            f"processed={processed} queued={queued} failed={failed} "
            f"missing_flight_id={missing_flight_id} errors={errors}"
        )
        self._send_json(
            200,
            {
                "ok": True,
                "processed": processed,
                "queued": queued,
                "failed": failed,
                "missing_flight_id": missing_flight_id,
                "errors": errors,
            },
        )

    def _handle_downloads_next(self) -> None:
        parsed = urlparse(self.path)
        qs = parse_qs(parsed.query)
        source = qs.get("source", [self.server.default_source])[0]
        limit = int(qs.get("limit", ["50"])[0])
        include_failed = qs.get("include_failed", ["0"])[0] in ("1", "true", "yes")
        statuses = ("queued", "failed") if include_failed else ("queued",)
        log(f"downloads next: source={source} limit={limit} include_failed={include_failed}")

        db = connect_db(self.server.db_url, None)
        items = []
        try:
            ensure_db(db)
            placeholders = ", ".join(["?"] * len(statuses))
            cur = db.execute(
                f"""
                SELECT flight_id, show_url, igc_url
                FROM flights
                WHERE source=?
                  AND igc_url IS NOT NULL
                  AND status IN ({placeholders})
                ORDER BY id ASC
                LIMIT ?
                """,
                (source, *statuses, limit),
            )
            for row in cur.fetchall():
                items.append({"flight_id": row[0], "show_url": row[1], "igc_url": row[2]})
            for item in items:
                db.execute(
                    """
                    UPDATE flights
                    SET status='downloading'
                    WHERE source=? AND flight_id=? AND status IN ('queued','failed')
                    """,
                    (source, item["flight_id"]),
                )
            db.commit()
        finally:
            db.close()
        self._send_json(200, {"ok": True, "items": items})

    def _parse_payload(self) -> Tuple[list, str]:
        try:
            payload = parse_json_body(self)
        except Exception as exc:
            self._send_json(400, {"ok": False, "error": f"invalid json: {exc}"})
            raise
        if payload is None:
            self._send_json(400, {"ok": False, "error": "empty body"})
            raise RuntimeError("empty body")
        if isinstance(payload, dict):
            items = payload.get("items") or payload.get("links") or []
            default_source = payload.get("source") or self.server.default_source
        else:
            items = payload
            default_source = self.server.default_source
        if not isinstance(items, list):
            self._send_json(400, {"ok": False, "error": "items must be a list"})
            raise RuntimeError("items not list")
        return items, default_source

    def _handle_links_post(self) -> None:
        try:
            items, default_source = self._parse_payload()
        except Exception:
            return

        processed = 0
        queued = 0
        missing_flight_id = 0
        errors = 0
        log(f"links request: items={len(items)} default_source={default_source}")

        db = connect_db(self.server.db_url, None)
        try:
            ensure_db(db)
            for raw in items:
                if not isinstance(raw, dict):
                    errors += 1
                    continue
                source = raw.get("source") or default_source
                show_url = raw.get("show_url") or raw.get("showUrl")
                igc_url = raw.get("igc_url") or raw.get("igcUrl")
                flight_id = raw.get("flight_id") or raw.get("flightId")
                if not flight_id:
                    flight_id = resolve_flight_id(show_url, igc_url)
                if not flight_id:
                    missing_flight_id += 1
                    continue

                try:
                    db_upsert_flight(db, source, flight_id, show_url)
                    if igc_url:
                        db_update_igc_url(db, source, flight_id, igc_url)
                        queued += 1
                    processed += 1
                except Exception:
                    errors += 1
            db.commit()
        finally:
            db.close()

        log(
            "links response: "
            f"processed={processed} queued={queued} missing_flight_id={missing_flight_id} errors={errors}"
        )
        self._send_json(
            200,
            {
                "ok": True,
                "processed": processed,
                "queued": queued,
                "missing_flight_id": missing_flight_id,
                "errors": errors,
            },
        )

    def _handle_downloads_post(self) -> None:
        try:
            items, default_source = self._parse_payload()
        except Exception:
            return

        processed = 0
        missing_flight_id = 0
        errors = 0
        log(f"downloads request: items={len(items)} default_source={default_source}")

        db = connect_db(self.server.db_url, None)
        try:
            ensure_db(db)
            for raw in items:
                if not isinstance(raw, dict):
                    errors += 1
                    continue
                source = raw.get("source") or default_source
                show_url = raw.get("show_url") or raw.get("showUrl")
                igc_url = raw.get("igc_url") or raw.get("igcUrl")
                igc_path = raw.get("igc_path") or raw.get("igcPath")
                file_size = raw.get("file_size") or raw.get("fileSize")
                downloaded_at = raw.get("downloaded_at") or raw.get("downloadedAt")
                flight_id = raw.get("flight_id") or raw.get("flightId")
                if not flight_id:
                    flight_id = resolve_flight_id(show_url, igc_url)
                if not flight_id:
                    missing_flight_id += 1
                    continue

                try:
                    db_upsert_flight(db, source, flight_id, show_url)
                    db_update_download(
                        db,
                        source,
                        flight_id,
                        igc_url,
                        igc_path,
                        int(file_size) if file_size is not None else None,
                        downloaded_at,
                    )
                    processed += 1
                except Exception:
                    errors += 1
            db.commit()
        finally:
            db.close()

        log(
            "downloads response: "
            f"processed={processed} missing_flight_id={missing_flight_id} errors={errors}"
        )
        self._send_json(
            200,
            {
                "ok": True,
                "processed": processed,
                "missing_flight_id": missing_flight_id,
                "errors": errors,
            },
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Local bridge server to save Leonardo links into a database.")
    parser.add_argument("--db-url",
                       default=os.environ.get("IGC_DB_URL", "postgresql://paraglidable:paraglidable@localhost:5432/paraglidable"),
                       help="PostgreSQL connection URL (default: from IGC_DB_URL env or localhost)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--source", default="paraplan")
    args = parser.parse_args()

    db = connect_db(args.db_url, None)
    try:
        ensure_db(db)
    finally:
        db.close()

    server = ThreadingHTTPServer((args.host, args.port), BridgeHandler)
    server.db_url = args.db_url
    server.default_source = args.source
    log(f"bridge server listening on http://{args.host}:{args.port}")
    log(f"bridge server db: {args.db_url}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log("shutting down")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
