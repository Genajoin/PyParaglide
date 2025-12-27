#!/usr/bin/env python3
import argparse
import hashlib
import math
import os
import random
import re
import signal
import socket
import sqlite3
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Any, Iterable, List, Optional, Tuple
from urllib.parse import parse_qs, urlencode, urljoin, urlparse, urlunparse

try:
    import requests  # type: ignore
except Exception as exc:
    print("ERROR: requests is required to run this script", file=sys.stderr)
    raise

try:
    import psycopg  # type: ignore
except Exception:
    psycopg = None
try:
    import urllib3.util.connection as urllib3_connection  # type: ignore
except Exception:
    urllib3_connection = None


BASE_URL_DEFAULT = "https://www.sky.gr/"
LIST_URL_DEFAULT = (
    "https://www.sky.gr/modules.php?lng=english&name=leonardo&op=list_flights&sortOrder=DATE"
)
USER_AGENT_DEFAULT = "Mozilla/5.0 (compatible; igc-downloader/0.2; +https://example.com)"

STOP_REQUESTED = False


@dataclass
class FlightStub:
    flight_id: str
    show_url: str


class SimpleLinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: List[Tuple[str, Optional[str]]] = []
        self._current_href: Optional[str] = None

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag.lower() != "a":
            return
        href = None
        for k, v in attrs:
            if k.lower() == "href":
                href = v
                break
        self._current_href = href

    def handle_data(self, data: str) -> None:
        if self._current_href is None:
            return
        text = data.strip()
        if not text:
            return
        self.links.append((self._current_href, text))

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a":
            self._current_href = None


class RateLimiter:
    def __init__(self, min_delay: float, max_delay: float) -> None:
        self.min_delay = min_delay
        self.max_delay = max_delay
        self._last_time = 0.0

    def sleep(self) -> None:
        now = time.time()
        elapsed = now - self._last_time
        delay = self.min_delay + random.random() * max(0.0, self.max_delay - self.min_delay)
        if elapsed < delay:
            time.sleep(delay - elapsed)
        self._last_time = time.time()


SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS flights (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
    takeoff_lat REAL,
    takeoff_lon REAL,
    takeoff_name TEXT,
    duration_sec INTEGER,
    distance_km REAL,
    score REAL,
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
    def __init__(self, conn: Any, kind: str) -> None:
        self.conn = conn
        self.kind = kind

    def execute(self, sql: str, params: Tuple[Any, ...] = ()) -> Any:
        if self.kind == "postgres":
            sql = sql.replace("?", "%s")
        return self.conn.execute(sql, params)

    def executescript(self, script: str) -> None:
        if self.kind == "sqlite":
            self.conn.executescript(script)
            return
        for stmt in script.split(";"):
            stmt = stmt.strip()
            if stmt:
                self.conn.execute(stmt)

    def commit(self) -> None:
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()


def connect_db(db_url: str, db_path: str) -> Db:
    if db_url:
        if psycopg is None:
            raise RuntimeError("psycopg is required for postgres. Install with: pip install psycopg[binary]")
        conn = psycopg.connect(db_url)
        return Db(conn, "postgres")
    conn = sqlite3.connect(db_path)
    return Db(conn, "sqlite")


def ensure_db(db: Db) -> None:
    if db.kind == "sqlite":
        db.executescript(SQLITE_SCHEMA)
    else:
        db.executescript(PG_SCHEMA)
    db.commit()
    ensure_columns(db)


def ensure_columns(db: Db) -> None:
    if db.kind == "sqlite":
        cur = db.execute("PRAGMA table_info(flights)")
        columns = {row[1] for row in cur.fetchall()}
        if "duplicate_of" not in columns:
            db.execute("ALTER TABLE flights ADD COLUMN duplicate_of TEXT")
            db.commit()
        if "updated_at" not in columns:
            db.execute("ALTER TABLE flights ADD COLUMN updated_at TEXT")
            db.commit()
        return
    db.execute("ALTER TABLE flights ADD COLUMN IF NOT EXISTS duplicate_of TEXT")
    db.execute("ALTER TABLE flights ADD COLUMN IF NOT EXISTS updated_at TEXT")
    db.commit()


def db_upsert_flight_stub(db: Db, source: str, stub: FlightStub) -> None:
    now = utcnow()
    db.execute(
        """
        INSERT INTO flights (source, flight_id, show_url, status, updated_at)
        VALUES (?, ?, ?, 'new', ?)
        ON CONFLICT(source, flight_id) DO UPDATE SET
            show_url=COALESCE(excluded.show_url, flights.show_url),
            updated_at=excluded.updated_at
        """,
        (source, stub.flight_id, stub.show_url, now),
    )


def db_update_crawl_state(
    db: Db,
    source: str,
    list_url: str,
    next_list_url: Optional[str],
    last_seen_flight_id: Optional[str],
) -> None:
    db.execute(
        """
        INSERT INTO crawl_state (source, list_url, next_list_url, last_seen_flight_id, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(source) DO UPDATE SET
            list_url=excluded.list_url,
            next_list_url=excluded.next_list_url,
            last_seen_flight_id=excluded.last_seen_flight_id,
            updated_at=excluded.updated_at
        """,
        (source, list_url, next_list_url, last_seen_flight_id, utcnow()),
    )


def db_get_crawl_state(db: Db, source: str) -> Tuple[Optional[str], Optional[str]]:
    cur = db.execute(
        "SELECT next_list_url, list_url FROM crawl_state WHERE source=?",
        (source,),
    )
    row = cur.fetchone()
    if not row:
        return None, None
    return row[0], row[1]


def db_update_crawl_state_list(
    db: Db,
    source: str,
    list_key: str,
    list_url: str,
    next_list_url: Optional[str],
    last_seen_flight_id: Optional[str],
    year: Optional[int],
) -> None:
    db.execute(
        """
        INSERT INTO crawl_state_list (
            source,
            list_key,
            list_url,
            next_list_url,
            last_seen_flight_id,
            year,
            updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source, list_key) DO UPDATE SET
            list_url=excluded.list_url,
            next_list_url=excluded.next_list_url,
            last_seen_flight_id=excluded.last_seen_flight_id,
            year=excluded.year,
            updated_at=excluded.updated_at
        """,
        (source, list_key, list_url, next_list_url, last_seen_flight_id, year, utcnow()),
    )


def db_get_crawl_state_list(
    db: Db,
    source: str,
    list_key: str,
) -> Tuple[Optional[str], Optional[str]]:
    cur = db.execute(
        "SELECT next_list_url, list_url FROM crawl_state_list WHERE source=? AND list_key=?",
        (source, list_key),
    )
    row = cur.fetchone()
    if not row:
        return None, None
    return row[0], row[1]


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


def db_update_after_download(
    db: Db,
    source: str,
    flight_id: str,
    igc_url: str,
    igc_path: str,
    file_size: int,
    sha256: str,
    downloaded_at: str,
) -> None:
    db.execute(
        """
        UPDATE flights
        SET igc_url=?, igc_path=?, file_size=?, sha256=?, downloaded_at=?, updated_at=?,
            status='downloaded', error_msg=NULL
        WHERE source=? AND flight_id=?
        """,
        (igc_url, igc_path, file_size, sha256, downloaded_at, downloaded_at, source, flight_id),
    )


def db_update_file_hash(
    db: Db,
    source: str,
    flight_id: str,
    sha256: str,
    file_size: int,
) -> None:
    now = utcnow()
    db.execute(
        """
        UPDATE flights
        SET sha256=?, file_size=?, updated_at=?
        WHERE source=? AND flight_id=?
        """,
        (sha256, file_size, now, source, flight_id),
    )


def db_mark_duplicate(
    db: Db,
    source: str,
    flight_id: str,
    sha256: str,
    file_size: int,
    duplicate_of: str,
    igc_path: Optional[str],
) -> None:
    now = utcnow()
    db.execute(
        """
        UPDATE flights
        SET status='duplicate',
            duplicate_of=?,
            sha256=?,
            file_size=?,
            igc_path=COALESCE(?, igc_path),
            updated_at=?
        WHERE source=? AND flight_id=?
        """,
        (duplicate_of, sha256, file_size, igc_path, now, source, flight_id),
    )


def db_update_igc_url(
    db: Db,
    source: str,
    flight_id: str,
    igc_url: str,
) -> None:
    now = utcnow()
    db.execute(
        """
        UPDATE flights
        SET igc_url=?, status='queued', error_msg=NULL, updated_at=?
        WHERE source=? AND flight_id=?
        """,
        (igc_url, now, source, flight_id),
    )


def db_update_after_parse(
    db: Db,
    source: str,
    flight_id: str,
    meta: dict,
) -> None:
    now = utcnow()
    db.execute(
        """
        UPDATE flights
        SET status='parsed',
            flight_date=?, pilot=?, glider=?, glider_class=?,
            takeoff_lat=?, takeoff_lon=?, takeoff_name=?,
            duration_sec=?, track_points=?, has_baro_alt=?, has_gps_alt=?,
            updated_at=?
        WHERE source=? AND flight_id=?
        """,
        (
            meta.get("flight_date"),
            meta.get("pilot"),
            meta.get("glider"),
            meta.get("glider_class"),
            meta.get("takeoff_lat"),
            meta.get("takeoff_lon"),
            meta.get("takeoff_name"),
            meta.get("duration_sec"),
            meta.get("track_points"),
            meta.get("has_baro_alt"),
            meta.get("has_gps_alt"),
            now,
            source,
            flight_id,
        ),
    )


def utcnow() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def request_stop(_: int, __: object) -> None:
    global STOP_REQUESTED
    if STOP_REQUESTED:
        return
    STOP_REQUESTED = True
    print("stop requested: finishing current task and exiting", flush=True)


def force_ipv4_only() -> None:
    if urllib3_connection is None:
        return

    def allowed_gai_family() -> int:
        return socket.AF_INET

    urllib3_connection.allowed_gai_family = allowed_gai_family


def extract_antibot_cookie(html: str) -> Optional[str]:
    if "document.cookie" not in html:
        return None
    base_match = re.search(r'document.cookie="([^"]+)"', html)
    if not base_match:
        return None
    raw = base_match.group(1)
    try:
        decoded = raw.encode("utf-8").decode("unicode_escape")
    except Exception:
        decoded = raw
    sqrt_match = re.search(r'document.cookie="[^"]+"\\+Math.sqrt\\((\\d+)\\)', html)
    if sqrt_match:
        decoded += str(int(math.sqrt(int(sqrt_match.group(1)))))
    if not decoded.startswith("antibot="):
        return None
    return decoded.split(";", 1)[0]


def update_cookie_header(session: requests.Session, cookie_kv: str) -> None:
    if "=" not in cookie_kv:
        return
    name, value = cookie_kv.split("=", 1)
    current = session.headers.get("Cookie", "")
    parts = [p.strip() for p in current.split(";") if p.strip()]
    cookies = {}
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            cookies[k] = v
    cookies[name] = value
    session.headers["Cookie"] = "; ".join(f"{k}={v}" for k, v in cookies.items())


def normalize_url(url: str) -> str:
    return url.replace("&amp;", "&")


def sanitize_list_key(url: str) -> str:
    url = normalize_url(url)
    parsed = urlparse(url)
    path = re.sub(r"&page_num=\d+", "", parsed.path)
    q = parse_qs(parsed.query, keep_blank_values=True)
    for key in ("page_num", "sid", "start"):
        q.pop(key, None)
    query = urlencode(q, doseq=True)
    return urlunparse((parsed.scheme, parsed.netloc, path, parsed.params, query, parsed.fragment))


def parse_years(years_str: str) -> List[int]:
    years = set()
    for part in years_str.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start = int(start_s)
            end = int(end_s)
            if end < start:
                start, end = end, start
            for year in range(start, end + 1):
                years.add(year)
        else:
            years.add(int(part))
    return sorted(years)


def build_year_list_url(year: int, list_url: str) -> str:
    if "/tracks/world/" in list_url:
        return re.sub(r"/tracks/world/(\d{4}|alltimes)/", f"/tracks/world/{year}/", list_url)
    return (
        "https://www.sky.gr/leonardo/tracks/world/"
        f"{year}/brand:all,cat:0,class:all,xctype:all,club:all,pilot:0_0,takeoff:all&sortOrder=DATE"
    )


def fetch(session: requests.Session, url: str, timeout: int, retries: int, limiter: RateLimiter) -> str:
    last_exc: Optional[Exception] = None
    antibot_applied = False
    for attempt in range(retries + 1):
        try:
            limiter.sleep()
            resp = session.get(url, timeout=timeout)
            text = resp.text
            if resp.status_code >= 400:
                antibot = extract_antibot_cookie(text)
                if antibot and not antibot_applied:
                    update_cookie_header(session, antibot)
                    antibot_applied = True
                    continue
                resp.raise_for_status()
            return text
        except Exception as exc:
            last_exc = exc
            backoff = 1.0 + (2 ** attempt) * 0.5
            time.sleep(backoff + random.random())
    raise RuntimeError(f"fetch failed for {url}: {last_exc}")


def download_file(
    session: requests.Session,
    url: str,
    out_path: str,
    timeout: int,
    retries: int,
    limiter: RateLimiter,
) -> Tuple[int, str]:
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            limiter.sleep()
            with session.get(url, timeout=timeout, stream=True) as resp:
                resp.raise_for_status()
                hasher = hashlib.sha256()
                size = 0
                with open(out_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 64):
                        if not chunk:
                            continue
                        f.write(chunk)
                        hasher.update(chunk)
                        size += len(chunk)
            return size, hasher.hexdigest()
        except Exception as exc:
            last_exc = exc
            backoff = 1.0 + (2 ** attempt) * 0.5
            time.sleep(backoff + random.random())
    raise RuntimeError(f"download failed for {url}: {last_exc}")


def compute_sha256(file_path: str) -> str:
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 64), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def find_duplicate(
    db: Db,
    sha256: str,
    source: str,
    flight_id: str,
) -> Optional[Tuple[str, str, Optional[str]]]:
    cur = db.execute(
        """
        SELECT source, flight_id, igc_path
        FROM flights
        WHERE sha256=? AND NOT (source=? AND flight_id=?) AND igc_path IS NOT NULL
        ORDER BY id ASC
        LIMIT 1
        """,
        (sha256, source, flight_id),
    )
    row = cur.fetchone()
    if not row:
        return None
    return row[0], row[1], row[2]


def parse_show_flight_urls(html: str, base_url: str) -> List[FlightStub]:
    urls: List[str] = []
    for href in re.findall(r"href=[\"']([^\"']+)[\"']", html, flags=re.IGNORECASE):
        if "op=show_flight" in href and "flightID=" in href:
            urls.append(urljoin(base_url, href))
        if "/leonardo/flight/" in href:
            urls.append(urljoin(base_url, href))
    for fid in re.findall(r"id=['\"]row_(\d+)['\"]", html, flags=re.IGNORECASE):
        urls.append(urljoin(base_url, f"/leonardo/flight/{fid}"))
    stubs = []
    seen = set()
    for u in urls:
        fid = extract_flight_id(u)
        if fid and fid not in seen:
            stubs.append(FlightStub(flight_id=fid, show_url=u))
            seen.add(fid)
    return stubs


def extract_flight_id(show_url: str) -> Optional[str]:
    parsed = urlparse(show_url)
    q = parse_qs(parsed.query)
    if "flightID" in q and q["flightID"]:
        return q["flightID"][0]
    match = re.search(r"/flight/(\d+)", parsed.path)
    if match:
        return match.group(1)
    match = re.search(r"flightID=(\d+)", show_url)
    if match:
        return match.group(1)
    return None


def extract_igc_url(html: str, base_url: str) -> Optional[str]:
    for href in re.findall(r"href=[\"']([^\"']+\.igc[^\"']*)[\"']", html, flags=re.IGNORECASE):
        return urljoin(base_url, href)

    parser = SimpleLinkParser()
    parser.feed(html)
    for href, text in parser.links:
        if text.strip().upper() == "IGC" and href:
            return urljoin(base_url, href)
    return None


def find_next_list_url(html: str, base_url: str, current_url: str) -> Optional[str]:
    candidates = []
    for href in re.findall(r"href=[\"']([^\"']+)[\"']", html, flags=re.IGNORECASE):
        if "op=list_flights" in href and ("start=" in href or "page=" in href):
            candidates.append(normalize_url(urljoin(base_url, href)))
        if "page_num=" in href and "/leonardo/tracks/" in href:
            candidates.append(normalize_url(urljoin(base_url, href)))

    if not candidates:
        return None

    current_page = extract_page_num(current_url)
    if current_page is None:
        current_page = -1
    current_start = extract_query_int(current_url, "start")
    if current_start is None:
        current_start = -1

    def page_num_from_url(url: str) -> Optional[int]:
        match = re.search(r"page_num=(\d+)", url)
        if match:
            return int(match.group(1))
        return extract_query_int(url, "page_num")

    best_url = None
    best_value = None
    for url in candidates:
        page = page_num_from_url(url)
        if page is not None:
            if page > current_page and (best_value is None or page < best_value):
                best_value = page
                best_url = url
            continue
        start = extract_query_int(url, "start")
        if start is None:
            continue
        if start > current_start and (best_value is None or start < best_value):
            best_value = start
            best_url = url

    return best_url or candidates[0]


def extract_query_int(url: str, key: str) -> Optional[int]:
    parsed = urlparse(url)
    q = parse_qs(parsed.query)
    if key not in q or not q[key]:
        return None
    try:
        return int(q[key][0])
    except ValueError:
        return None


def extract_page_num(url: str) -> Optional[int]:
    match = re.search(r"page_num=(\d+)", url)
    if match:
        return int(match.group(1))
    return None


def replace_page_num(url: str, new_page: int) -> str:
    if "page_num=" in url:
        return re.sub(r"page_num=\d+", f"page_num={new_page}", url)
    if "?" in url or "&" in url:
        return f"{url}&page_num={new_page}"
    return f"{url}?page_num={new_page}"


def extract_page_nums(html: str) -> List[int]:
    return [int(x) for x in re.findall(r"page_num=(\d+)", html)]


def parse_igc(file_path: str) -> dict:
    meta = {
        "flight_date": None,
        "pilot": None,
        "glider": None,
        "glider_class": None,
        "takeoff_lat": None,
        "takeoff_lon": None,
        "takeoff_name": None,
        "duration_sec": None,
        "track_points": 0,
        "has_baro_alt": 0,
        "has_gps_alt": 0,
    }

    first_time = None
    last_time = None
    fallback_lat = None
    fallback_lon = None

    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.upper().startswith("HFDTEDATE"):
                match = re.search(r"(\d{6})", line)
                if match:
                    meta["flight_date"] = parse_igc_date(match.group(1))
            elif line.startswith("HFDTE"):
                date_str = line[5:11]
                meta["flight_date"] = parse_igc_date(date_str)
            elif line.upper().startswith("HFPLT"):
                meta["pilot"] = line.split(":", 1)[-1].strip() or meta["pilot"]
            elif line.upper().startswith("HFGTY"):
                meta["glider"] = line.split(":", 1)[-1].strip() or meta["glider"]
            elif line.upper().startswith("HFGCL"):
                meta["glider_class"] = line.split(":", 1)[-1].strip() or meta["glider_class"]
            elif line.upper().startswith("HFCCL"):
                meta["glider_class"] = line.split(":", 1)[-1].strip() or meta["glider_class"]
            elif line.startswith("B") and len(line) >= 35:
                meta["track_points"] += 1
                time_str = line[1:7]
                if time_str.isdigit():
                    t = parse_hhmmss(time_str)
                    if first_time is None:
                        first_time = t
                    last_time = t
                lat_str = line[7:15]
                lon_str = line[15:24]
                lat = parse_lat(lat_str)
                lon = parse_lon(lon_str)
                if fallback_lat is None and fallback_lon is None:
                    fallback_lat = lat
                    fallback_lon = lon
                fix_validity = line[24:25]
                baro_alt = line[25:30]
                gps_alt = line[30:35]
                if fix_validity and fix_validity in ("A", "V"):
                    if baro_alt.isdigit():
                        meta["has_baro_alt"] = 1
                    if gps_alt.isdigit():
                        meta["has_gps_alt"] = 1
                if (
                    fix_validity == "A"
                    and meta["takeoff_lat"] is None
                    and meta["takeoff_lon"] is None
                    and lat is not None
                    and lon is not None
                ):
                    meta["takeoff_lat"] = lat
                    meta["takeoff_lon"] = lon

    if first_time is not None and last_time is not None and last_time >= first_time:
        meta["duration_sec"] = last_time - first_time

    if meta["takeoff_lat"] is None and meta["takeoff_lon"] is None:
        meta["takeoff_lat"] = fallback_lat
        meta["takeoff_lon"] = fallback_lon

    return meta


def parse_igc_date(date_str: str) -> Optional[str]:
    if len(date_str) != 6 or not date_str.isdigit():
        return None
    dd = int(date_str[0:2])
    mm = int(date_str[2:4])
    yy = int(date_str[4:6])
    year = 2000 + yy if yy < 80 else 1900 + yy
    try:
        return datetime(year, mm, dd).date().isoformat()
    except ValueError:
        return None


def parse_hhmmss(s: str) -> int:
    hh = int(s[0:2])
    mm = int(s[2:4])
    ss = int(s[4:6])
    return hh * 3600 + mm * 60 + ss


def parse_lat(lat_str: str) -> Optional[float]:
    if len(lat_str) != 8:
        return None
    try:
        deg = int(lat_str[0:2])
        minutes = int(lat_str[2:4])
        thousandths = int(lat_str[4:7])
        hemi = lat_str[7].upper()
        value = deg + (minutes + thousandths / 1000.0) / 60.0
        if hemi == "S":
            value = -value
        return value
    except ValueError:
        return None


def parse_lon(lon_str: str) -> Optional[float]:
    if len(lon_str) != 9:
        return None
    try:
        deg = int(lon_str[0:3])
        minutes = int(lon_str[3:5])
        thousandths = int(lon_str[5:8])
        hemi = lon_str[8].upper()
        value = deg + (minutes + thousandths / 1000.0) / 60.0
        if hemi == "W":
            value = -value
        return value
    except ValueError:
        return None


def ensure_dirs(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def build_dest_path(out_dir: str, flight_id: str, filename: str, flight_date: Optional[str]) -> str:
    if flight_date:
        try:
            dt = datetime.strptime(flight_date, "%Y-%m-%d")
            year = f"{dt.year:04d}"
            month = f"{dt.month:02d}"
        except ValueError:
            year = "unknown"
            month = "unknown"
    else:
        year = "unknown"
        month = "unknown"
    dest_dir = os.path.join(out_dir, year, month, flight_id)
    ensure_dirs(dest_dir)
    return os.path.join(dest_dir, filename)


def iter_pending_flights(db: Db, source: str, max_retries: int) -> Iterable[Tuple[str, str]]:
    cur = db.execute(
        """
        SELECT flight_id, show_url
        FROM flights
        WHERE source=? AND status IN ('new','failed','queued','downloading') AND retry_count < ? AND igc_url IS NOT NULL
        ORDER BY id ASC
        """,
        (source, max_retries),
    )
    for row in cur:
        yield row[0], row[1]


def count_pending_flights(db: Db, source: str, max_retries: int) -> int:
    cur = db.execute(
        """
        SELECT COUNT(*)
        FROM flights
        WHERE source=? AND status IN ('new','failed','queued','downloading') AND retry_count < ? AND igc_url IS NOT NULL
        """,
        (source, max_retries),
    )
    row = cur.fetchone()
    return int(row[0]) if row else 0


def iter_link_flights(db: Db, source: str, max_retries: int) -> Iterable[Tuple[str, str]]:
    cur = db.execute(
        """
        SELECT flight_id, show_url
        FROM flights
        WHERE source=? AND status IN ('new','failed') AND retry_count < ?
        ORDER BY id ASC
        """,
        (source, max_retries),
    )
    for row in cur:
        yield row[0], row[1]


def crawl_list(
    db: Db,
    session: requests.Session,
    base_url: str,
    list_url: str,
    list_key: str,
    max_pages: int,
    limiter: RateLimiter,
    timeout: int,
    retries: int,
    source: str,
    year: Optional[int],
) -> int:
    url = list_url
    total = 0
    max_page: Optional[int] = None
    page = 0
    while True:
        if STOP_REQUESTED:
            break
        if max_pages > 0 and page >= max_pages:
            break
        html = fetch(session, url, timeout=timeout, retries=retries, limiter=limiter)
        stubs = parse_show_flight_urls(html, base_url)
        page_nums = extract_page_nums(html)
        if page_nums:
            page_max = max(page_nums)
            if max_page is None or page_max > max_page:
                max_page = page_max
        last_flight_id = None
        for stub in stubs:
            db_upsert_flight_stub(db, source, stub)
            last_flight_id = stub.flight_id
        db.commit()
        total += len(stubs)
        if year is None:
            print(f"list page {page + 1}: +{len(stubs)} flights", flush=True)
        else:
            print(f"list {year} page {page + 1}: +{len(stubs)} flights", flush=True)
        next_url = find_next_list_url(html, base_url, url)
        db_update_crawl_state(db, source, url, next_url, last_flight_id)
        db_update_crawl_state_list(db, source, list_key, url, next_url, last_flight_id, year)
        db.commit()
        if not next_url or next_url == url:
            current_page = extract_page_num(url)
            if current_page is not None and stubs:
                if max_page is not None and current_page >= max_page:
                    break
                next_url = replace_page_num(url, current_page + 1)
            else:
                break
        url = normalize_url(next_url)
        page += 1
    return total


def process_flights(
    db: Db,
    session: requests.Session,
    base_url: str,
    out_dir: str,
    timeout: int,
    retries: int,
    limiter: RateLimiter,
    source: str,
    max_retries: int,
    max_flights: Optional[int],
) -> int:
    ensure_dirs(out_dir)
    incoming_dir = os.path.join(out_dir, "_incoming")
    ensure_dirs(incoming_dir)

    total_count: Optional[int]
    if max_flights is None:
        total_count = count_pending_flights(db, source, max_retries)
    else:
        total_count = max_flights

    count = 0
    for flight_id, show_url in iter_pending_flights(db, source, max_retries):
        if STOP_REQUESTED:
            break
        if max_flights is not None and count >= max_flights:
            break
        if total_count is None:
            progress = f"{count + 1}"
        else:
            progress = f"{count + 1}/{total_count}"
        prefix = f"flight {progress}: {flight_id}"
        try:
            html = fetch(session, show_url, timeout=timeout, retries=retries, limiter=limiter)
            igc_url = extract_igc_url(html, base_url)
            if not igc_url:
                db_mark_failed(db, source, flight_id, "no igc url found")
                db.commit()
                print(f"{prefix}: no igc url", flush=True)
                continue

            filename = igc_url.split("?")[0].split("/")[-1]
            if not filename.lower().endswith(".igc"):
                filename += ".igc"

            tmp_path = os.path.join(incoming_dir, f"{flight_id}_{filename}")
            size, sha256 = download_file(
                session,
                igc_url,
                tmp_path,
                timeout=timeout,
                retries=retries,
                limiter=limiter,
            )

            duplicate = find_duplicate(db, sha256, source, flight_id)
            if duplicate:
                dup_source, dup_flight_id, dup_path = duplicate
                dup_tag = f"{dup_source}:{dup_flight_id}"
                db_mark_duplicate(
                    db,
                    source,
                    flight_id,
                    sha256,
                    size,
                    dup_tag,
                    dup_path,
                )
                db.commit()
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                print(f"{prefix}: duplicate of {dup_tag}", flush=True)
                continue

            meta = parse_igc(tmp_path)
            dest_path = build_dest_path(out_dir, flight_id, filename, meta.get("flight_date"))
            os.replace(tmp_path, dest_path)

            db_update_after_download(
                db,
                source,
                flight_id,
                igc_url,
                dest_path,
                size,
                sha256,
                utcnow(),
            )
            db_update_after_parse(db, source, flight_id, meta)
            db.commit()
            count += 1
            print(f"{prefix}: downloaded", flush=True)
        except Exception as exc:
            db_mark_failed(db, source, flight_id, str(exc))
            db.commit()
            print(f"{prefix}: failed ({exc})", flush=True)
    return count


def process_links(
    db: Db,
    session: requests.Session,
    base_url: str,
    timeout: int,
    retries: int,
    limiter: RateLimiter,
    source: str,
    max_retries: int,
    max_links: Optional[int],
) -> int:
    count = 0
    for flight_id, show_url in iter_link_flights(db, source, max_retries):
        if STOP_REQUESTED:
            break
        if max_links is not None and count >= max_links:
            break
        if max_links is None:
            progress = f"{count + 1}"
        else:
            progress = f"{count + 1}/{max_links}"
        print(f"link {progress}: {flight_id}", flush=True)
        try:
            html = fetch(session, show_url, timeout=timeout, retries=retries, limiter=limiter)
            igc_url = extract_igc_url(html, base_url)
            if not igc_url:
                db_mark_failed(db, source, flight_id, "no igc url found")
                db.commit()
                print(f"link {flight_id}: no igc url", flush=True)
                continue
            db_update_igc_url(db, source, flight_id, igc_url)
            db.commit()
            count += 1
            print(f"link {flight_id}: queued", flush=True)
        except Exception as exc:
            db_mark_failed(db, source, flight_id, str(exc))
            db.commit()
            print(f"link {flight_id}: failed ({exc})", flush=True)
    return count


def reparse_existing(
    db: Db,
    source: str,
    max_files: Optional[int],
) -> int:
    cur = db.execute(
        """
        SELECT flight_id, igc_path
        FROM flights
        WHERE source=? AND igc_path IS NOT NULL
        ORDER BY id ASC
        """,
        (source,),
    )
    count = 0
    for flight_id, igc_path in cur:
        if STOP_REQUESTED:
            break
        if max_files is not None and count >= max_files:
            break
        if not igc_path or not os.path.exists(igc_path):
            print(f"reparse {flight_id}: missing file", flush=True)
            continue
        try:
            sha256 = compute_sha256(igc_path)
            file_size = os.path.getsize(igc_path)
            duplicate = find_duplicate(db, sha256, source, flight_id)
            if duplicate:
                dup_source, dup_flight_id, _dup_path = duplicate
                dup_tag = f"{dup_source}:{dup_flight_id}"
                db_mark_duplicate(
                    db,
                    source,
                    flight_id,
                    sha256,
                    file_size,
                    dup_tag,
                    None,
                )
                db.commit()
                print(f"reparse {flight_id}: duplicate of {dup_tag}", flush=True)
                continue
            db_update_file_hash(db, source, flight_id, sha256, file_size)
            meta = parse_igc(igc_path)
            db_update_after_parse(db, source, flight_id, meta)
            db.commit()
            count += 1
            print(f"reparse {flight_id}: updated", flush=True)
        except Exception as exc:
            db_mark_failed(db, source, flight_id, str(exc))
            db.commit()
            print(f"reparse {flight_id}: failed ({exc})", flush=True)
    return count


def collect_stats(db: Db, source: str) -> dict:
    stats = {
        "total": 0,
        "with_igc_url": 0,
        "without_igc_url": 0,
        "downloaded": 0,
        "parsed": 0,
        "duplicate": 0,
    }
    cur = db.execute(
        "SELECT COUNT(*) FROM flights WHERE source=?",
        (source,),
    )
    stats["total"] = int(cur.fetchone()[0])
    cur = db.execute(
        "SELECT COUNT(*) FROM flights WHERE source=? AND igc_url IS NOT NULL",
        (source,),
    )
    stats["with_igc_url"] = int(cur.fetchone()[0])
    stats["without_igc_url"] = stats["total"] - stats["with_igc_url"]
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
    stats["status_counts"] = status_counts
    stats["downloaded"] = status_counts.get("downloaded", 0)
    stats["parsed"] = status_counts.get("parsed", 0)
    stats["duplicate"] = status_counts.get("duplicate", 0)
    return stats


def print_stats(db: Db, source: str) -> None:
    stats = collect_stats(db, source)
    status_counts = stats.get("status_counts", {})
    parts = [
        f"total={stats['total']}",
        f"with_igc_url={stats['with_igc_url']}",
        f"without_igc_url={stats['without_igc_url']}",
        f"downloaded={stats['downloaded']}",
        f"parsed={stats['parsed']}",
        f"duplicate={stats['duplicate']}",
    ]
    if status_counts:
        extras = ",".join(f"{k}={v}" for k, v in sorted(status_counts.items()))
        parts.append(f"status[{extras}]")
    print("stats: " + " ".join(parts), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download IGC flights from sky.gr Leonardo.")
    parser.add_argument("--base-url", default=BASE_URL_DEFAULT)
    parser.add_argument("--list-url", default=LIST_URL_DEFAULT)
    parser.add_argument("--years", default="")
    parser.add_argument("--source", default="skygr")
    parser.add_argument("--cookie", default="")
    parser.add_argument("--db-path", default=os.path.join("data", "igc", "index.sqlite"))
    parser.add_argument("--db-url", default=os.environ.get("IGC_DB_URL", ""))
    parser.add_argument("--out-dir", default=os.path.join("data", "igc", "skygr"))
    parser.add_argument("--max-pages", type=int, default=3)
    parser.add_argument("--max-flights", type=int, default=200)
    parser.add_argument("--max-links", type=int, default=0)
    parser.add_argument("--max-reparse", type=int, default=0)
    parser.add_argument("--min-delay", type=float, default=1.0)
    parser.add_argument("--max-delay", type=float, default=1.5)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--links-only", action="store_true")
    parser.add_argument("--download-only", action="store_true")
    parser.add_argument("--continue", dest="continue_mode", action="store_true")
    parser.add_argument("--reparse", action="store_true")
    parser.add_argument("--reparse-only", action="store_true")
    parser.add_argument("--force-ipv4", action="store_true")
    parser.add_argument("--proxy", default="")
    parser.add_argument("--header", action="append", default=[])
    parser.add_argument("--user-agent", default=USER_AGENT_DEFAULT)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, request_stop)

    db = connect_db(args.db_url, args.db_path)
    if db.kind == "sqlite":
        ensure_dirs(os.path.dirname(args.db_path))
    ensure_db(db)

    if args.force_ipv4:
        force_ipv4_only()

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": args.user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
    )
    for header in args.header:
        if ":" not in header:
            continue
        key, value = header.split(":", 1)
        session.headers[key.strip()] = value.strip()
    if args.proxy:
        session.proxies.update({"http": args.proxy, "https": args.proxy})
    if args.cookie:
        session.headers.update({"Cookie": args.cookie})

    limiter = RateLimiter(args.min_delay, args.max_delay)

    if args.reparse_only:
        args.reparse = True
        args.list_only = True
        args.links_only = True
        args.download_only = True

    try:
        if not args.download_only:
            year_list: List[int] = []
            if args.years:
                year_list = parse_years(args.years)
            if year_list:
                total_all = 0
                for year in year_list:
                    if STOP_REQUESTED:
                        break
                    list_key = sanitize_list_key(build_year_list_url(year, args.list_url))
                    list_url = list_key
                    if args.continue_mode:
                        next_url, last_url = db_get_crawl_state_list(db, args.source, list_key)
                        list_url = next_url or last_url or list_url
                        if list_url != list_key:
                            print(f"continue {year} list from: {list_url}", flush=True)
                    total = crawl_list(
                        db=db,
                        session=session,
                        base_url=args.base_url,
                        list_url=list_url,
                        list_key=list_key,
                        max_pages=args.max_pages,
                        limiter=limiter,
                        timeout=args.timeout,
                        retries=args.retries,
                        source=args.source,
                        year=year,
                    )
                    print(f"list {year} crawl: {total} flights discovered")
                    total_all += total
                print(f"list crawl total: {total_all} flights discovered")
            else:
                list_key = sanitize_list_key(args.list_url)
                list_url = list_key
                if args.continue_mode:
                    next_url, last_url = db_get_crawl_state_list(db, args.source, list_key)
                    list_url = next_url or last_url or list_url
                    if list_url != list_key:
                        print(f"continue list from: {list_url}", flush=True)
                total = crawl_list(
                    db=db,
                    session=session,
                    base_url=args.base_url,
                    list_url=list_url,
                    list_key=list_key,
                    max_pages=args.max_pages,
                    limiter=limiter,
                    timeout=args.timeout,
                    retries=args.retries,
                    source=args.source,
                    year=None,
                )
                print(f"list crawl: {total} flights discovered")

        max_links: Optional[int]
        if args.max_links <= 0:
            max_links = None
        else:
            max_links = args.max_links

        if args.links_only and not args.list_only:
            linked = process_links(
                db=db,
                session=session,
                base_url=args.base_url,
                timeout=args.timeout,
                retries=args.retries,
                limiter=limiter,
                source=args.source,
                max_retries=args.max_retries,
                max_links=max_links,
            )
            print(f"links queued: {linked} flights")

        max_flights: Optional[int]
        if args.max_flights <= 0:
            max_flights = None
        else:
            max_flights = args.max_flights

        if not args.list_only and not args.links_only:
            processed = process_flights(
                db=db,
                session=session,
                base_url=args.base_url,
                out_dir=args.out_dir,
                timeout=args.timeout,
                retries=args.retries,
                limiter=limiter,
                source=args.source,
                max_retries=args.max_retries,
                max_flights=max_flights,
            )
            print(f"downloaded+parsed: {processed} flights")

        max_reparse: Optional[int]
        if args.max_reparse <= 0:
            max_reparse = None
        else:
            max_reparse = args.max_reparse

        if args.reparse:
            updated = reparse_existing(db=db, source=args.source, max_files=max_reparse)
            print(f"reparse updated: {updated} flights")
    finally:
        try:
            print_stats(db, args.source)
        finally:
            db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
