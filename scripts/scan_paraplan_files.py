#!/usr/bin/env python3
"""
Scan downloaded paraplan IGC files and update database with igc_path.

Matches files by filename found in igc_url field.
"""
import argparse
import os
import re
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import unquote

try:
    import psycopg
except ImportError:
    raise RuntimeError("psycopg required. Install: pip install psycopg[binary]")

# IGC date format: DDMMYY (e.g., 290419 for 2019-04-29)
IGC_DATE_RE = re.compile(r"HFDTEDATE:(\d{6})")


def extract_igc_date(filepath: str) -> Optional[str]:
    """Extract date from IGC file header.

    Returns ISO format date string (YYYY-MM-DD) or None.
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                if line.startswith('HFDTE'):
                    match = IGC_DATE_RE.search(line)
                    if match:
                        date_str = match.group(1)  # DDMMYY
                        day = int(date_str[0:2])
                        month = int(date_str[2:4])
                        year = int(date_str[4:6])
                        # Convert 2-digit year to 4-digit (pivot at 50)
                        year += 2000 if year < 50 else 1900
                        try:
                            return f"{year:04d}-{month:02d}-{day:02d}"
                        except ValueError:
                            return None
    except Exception:
        pass
    return None


def compute_sha256(filepath: str) -> str:
    """Compute SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            sha256.update(chunk)
    return sha256.hexdigest()


def extract_filename_from_url(url: str) -> Optional[str]:
    """Extract filename from URL, handling URL encoding."""
    if not url:
        return None
    # Get the last part of URL path
    parts = url.rstrip('/').split('/')
    if parts:
        filename = parts[-1]
        # Decode URL encoding (e.g., %20 -> space)
        return unquote(filename)
    return None


def scan_directory(base_dir: str) -> list:
    """Scan directory for IGC files, return list of (path, filename, date, size, sha256)."""
    files_info = []
    base_path = Path(base_dir)

    if not base_path.exists():
        return files_info

    for igc_file in base_path.rglob("*.igc"):
        filepath = str(igc_file)
        filename = igc_file.name
        file_size = igc_file.stat().st_size

        # Extract date from IGC content
        flight_date = extract_igc_date(filepath)

        # Compute SHA256
        sha256_hash = compute_sha256(filepath)

        files_info.append((filepath, filename, flight_date, file_size, sha256_hash))

    return files_info


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan paraplan IGC files and update database")
    parser.add_argument("--db-url",
                       default=os.environ.get("IGC_DB_URL", "postgresql://paraglidable:paraglidable@localhost:5432/paraglidable"),
                       help="PostgreSQL connection URL")
    parser.add_argument("--source", default="paraplan", help="Source name in database")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done without changes")
    args = parser.parse_args()

    # Directories to scan
    download_dir = os.path.expanduser("~/Downloads/paraplan.ru")
    home_dir = os.path.expanduser("~/paraplan.ru")

    print(f"Scanning {download_dir}...")
    files_download = scan_directory(download_dir)
    print(f"Found {len(files_download)} IGC files in Downloads")

    print(f"Scanning {home_dir}...")
    files_home = scan_directory(home_dir)
    print(f"Found {len(files_home)} IGC files in home")

    all_files = files_download + files_home
    print(f"Total: {len(all_files)} files")

    # Build filename -> files mapping (handle duplicates)
    filename_to_files = {}
    for filepath, filename, flight_date, file_size, sha256_hash in all_files:
        if filename not in filename_to_files:
            filename_to_files[filename] = []
        filename_to_files[filename].append((filepath, flight_date, file_size, sha256_hash))

    print(f"Unique filenames: {len(filename_to_files)}")

    # Connect to database
    conn = psycopg.connect(args.db_url)

    try:
        # Fetch all flights with igc_url but without igc_path
        cur = conn.execute(
            """
            SELECT flight_id, igc_url, status
            FROM flights
            WHERE source=%s AND igc_url IS NOT NULL AND igc_path IS NULL
            """,
            (args.source,)
        )

        updated = 0
        skipped = 0
        not_found = 0

        for flight_id, igc_url, status in cur.fetchall():
            filename = extract_filename_from_url(igc_url)
            if not filename:
                not_found += 1
                continue

            if filename not in filename_to_files:
                not_found += 1
                continue

            # Get file(s) with this filename
            matching_files = filename_to_files[filename]

            if len(matching_files) == 1:
                filepath, flight_date, file_size, sha256_hash = matching_files[0]
                if args.dry_run:
                    print(f"Would update: {flight_id} -> {filepath}")
                else:
                    now = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
                    conn.execute(
                        """
                        UPDATE flights
                        SET igc_path=%s, file_size=%s, sha256=%s, flight_date=%s,
                            downloaded_at=%s, updated_at=%s, status='downloaded'
                        WHERE source=%s AND flight_id=%s
                        """,
                        (filepath, file_size, sha256_hash, flight_date, now, now, args.source, flight_id)
                    )
                updated += 1
                # Remove from dict so we don't match it again
                del filename_to_files[filename]
            else:
                # Multiple files with same name - skip or handle differently
                not_found += 1

        if not args.dry_run:
            conn.commit()

        print(f"\nResults:")
        print(f"  Updated: {updated}")
        print(f"  Not found: {not_found}")
        print(f"  Remaining files in DB without match: {len(filename_to_files)}")

    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
