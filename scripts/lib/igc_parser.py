"""IGC file parsing utilities."""

import re
from datetime import datetime
from typing import Optional, Dict, Any


def parse_igc(file_path: str) -> Dict[str, Any]:
    """Parse IGC file header and B-records for basic metadata.

    Args:
        file_path: Path to IGC file

    Returns:
        Dictionary with parsed metadata:
        - flight_date: ISO date string (YYYY-MM-DD)
        - pilot: Pilot name from HFPLT
        - glider: Glider model from HFGTY
        - glider_class: Glider class from HFGCL/HFCCL
        - takeoff_lat: Takeoff latitude (decimal degrees)
        - takeoff_lon: Takeoff longitude (decimal degrees)
        - takeoff_name: Takeoff site name
        - duration_sec: Flight duration in seconds
        - track_points: Number of B-records
        - has_baro_alt: 1 if barometric altitude present
        - has_gps_alt: 1 if GPS altitude present
    """
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

            # Date records
            if line.upper().startswith("HFDTEDATE"):
                match = re.search(r"(\d{6})", line)
                if match:
                    meta["flight_date"] = parse_igc_date(match.group(1))
            elif line.startswith("HFDTE"):
                date_str = line[5:11]
                meta["flight_date"] = parse_igc_date(date_str)

            # Header records
            elif line.upper().startswith("HFPLT"):
                meta["pilot"] = line.split(":", 1)[-1].strip() or meta["pilot"]
            elif line.upper().startswith("HFGTY"):
                meta["glider"] = line.split(":", 1)[-1].strip() or meta["glider"]
            elif line.upper().startswith("HFGCL"):
                meta["glider_class"] = line.split(":", 1)[-1].strip() or meta["glider_class"]
            elif line.upper().startswith("HFCCL"):
                meta["glider_class"] = line.split(":", 1)[-1].strip() or meta["glider_class"]

            # Track points (B-records)
            elif line.startswith("B") and len(line) >= 35:
                meta["track_points"] += 1

                # Time
                time_str = line[1:7]
                if time_str.isdigit():
                    t = _parse_hhmmss(time_str)
                    if first_time is None:
                        first_time = t
                    last_time = t

                # Position
                lat_str = line[7:15]
                lon_str = line[15:24]
                lat = _parse_lat(lat_str)
                lon = _parse_lon(lon_str)

                if fallback_lat is None and fallback_lon is None:
                    fallback_lat = lat
                    fallback_lon = lon

                # Altitude
                fix_validity = line[24:25]
                baro_alt = line[25:30]
                gps_alt = line[30:35]

                if fix_validity and fix_validity in ("A", "V"):
                    if baro_alt.isdigit():
                        meta["has_baro_alt"] = 1
                    if gps_alt.isdigit():
                        meta["has_gps_alt"] = 1

                # First valid fix as takeoff
                if (
                    fix_validity == "A"
                    and meta["takeoff_lat"] is None
                    and meta["takeoff_lon"] is None
                    and lat is not None
                    and lon is not None
                ):
                    meta["takeoff_lat"] = lat
                    meta["takeoff_lon"] = lon

    # Calculate duration
    if first_time is not None and last_time is not None and last_time >= first_time:
        meta["duration_sec"] = last_time - first_time

    # Fallback to first track point if no valid takeoff found
    if meta["takeoff_lat"] is None and meta["takeoff_lon"] is None:
        meta["takeoff_lat"] = fallback_lat
        meta["takeoff_lon"] = fallback_lon

    return meta


def parse_igc_date(date_str: str) -> Optional[str]:
    """Parse IGC date format (DDMMYY) to ISO format (YYYY-MM-DD).

    Args:
        date_str: Date string in DDMMYY format

    Returns:
        ISO date string (YYYY-MM-DD) or None if invalid
    """
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


def _parse_hhmmss(s: str) -> int:
    """Parse HHMMSS to seconds since midnight."""
    hh = int(s[0:2])
    mm = int(s[2:4])
    ss = int(s[4:6])
    return hh * 3600 + mm * 60 + ss


def _parse_lat(lat_str: str) -> Optional[float]:
    """Parse IGC latitude format (DDMMmmmN/S) to decimal degrees."""
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


def _parse_lon(lon_str: str) -> Optional[float]:
    """Parse IGC longitude format (DDDMMmmmE/W) to decimal degrees."""
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
