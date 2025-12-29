#!/usr/bin/env python3
"""
IGC parser using libigc for metadata and xc_score for XC scoring.

This module provides enhanced IGC parsing with:
- libigc for flight analysis (thermals, glides, statistics)
- xc_score for accurate XC score calculation (FAI triangles, free distance)
- Fallback to parse_igc_extended if libraries not available
"""

import os
import sys
from typing import Optional, Dict, Any

# Try to import libigc
try:
    from libigc import Flight as LibIGCFlight
    HAS_LIBIGC = True
except ImportError:
    HAS_LIBIGC = False
    print("Warning: libigc not available, using deprecated parse_igc_extended.py fallback", file=sys.stderr)
    print("  Install libigc for better thermal/glide analysis: pip install libigc", file=sys.stderr)

# Try to import xc_score (from local copy)
try:
    # Add xc_score directory to path
    xc_score_dir = os.path.join(os.path.dirname(__file__), "xc_score")
    if xc_score_dir not in sys.path:
        sys.path.insert(0, xc_score_dir)

    from xc_scorer import process_igc_file, XCScorer, IGCParser
    HAS_XC_SCORE = True
except ImportError as e:
    HAS_XC_SCORE = False
    print(f"Warning: xc_score not available: {e}", file=sys.stderr)

# Import existing parser for H-records and fallback
try:
    from igc_ingest_skygr import parse_igc, parse_igc_date
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from igc_ingest_skygr import parse_igc, parse_igc_date

# Fallback to our custom parser if needed
try:
    from parse_igc_extended import (
        calculate_distance,
        extract_altitude_stats,
        build_takeoff_datetime,
        parse_b_record as parse_b_record_extended
    )
    HAS_EXTENDED_PARSER = True
except ImportError:
    # If parse_igc_extended doesn't exist yet, we'll handle it
    calculate_distance = None
    extract_altitude_stats = None
    build_takeoff_datetime = None
    parse_b_record_extended = None
    HAS_EXTENDED_PARSER = False


# Default XC scoring rules (XContest-style)
DEFAULT_SCORING_RULES = {
    "flat": {
        "multiplier": 1.2
    },
    "FAI": {
        "multiplier": 1.4
    },
    "closedFAI": {
        "multiplier": 1.6
    },
    "closedFlat": {
        "multiplier": 1.4
    },
    "freeFlight": {
        "multiplier": 1.0
    }
}


def parse_igc_with_libs(file_path: str) -> Dict[str, Any]:
    """
    Полный парсинг IGC используя libigc + xc_score.

    Args:
        file_path: путь к IGC файлу

    Returns:
        dict со всеми метриками
    """
    result = {}

    # 1. Parse H-records for pilot/glider (always use igc_ingest_skygr for this)
    try:
        basic_meta = parse_igc(file_path)
        result.update({
            "flight_date": basic_meta.get("flight_date"),
            "pilot": basic_meta.get("pilot"),
            "glider": basic_meta.get("glider"),
            "glider_class": basic_meta.get("glider_class"),
        })
    except Exception as e:
        print(f"Warning: Basic parsing failed: {e}", file=sys.stderr)
        result.update({
            "flight_date": None,
            "pilot": None,
            "glider": None,
            "glider_class": None,
        })

    # 2. Parse with libigc if available
    if HAS_LIBIGC:
        try:
            flight_data = parse_with_libigc(file_path)
            result.update(flight_data)
        except Exception as e:
            print(f"Warning: libigc parsing failed: {e}", file=sys.stderr)
            # Fallback to basic parsing
            flight_data = parse_with_fallback(file_path)
            result.update(flight_data)
    else:
        # Fallback to parse_igc_extended
        flight_data = parse_with_fallback(file_path)
        result.update(flight_data)

    # 3. Calculate XC score if xc_score available
    if HAS_XC_SCORE:
        try:
            xc_result = calculate_xc_score_safe(file_path)
            result.update({
                "xc_score": xc_result["score"],
                "xc_distance_km": xc_result["distance_km"],
                "xc_type": xc_result["type"],
            })
        except Exception as e:
            print(f"Warning: XC score calculation failed: {e}", file=sys.stderr)
            # Use distance_km as fallback score
            result.update({
                "xc_score": result.get("distance_km", 0.0),
                "xc_distance_km": result.get("distance_km", 0.0),
                "xc_type": "fallback_distance",
            })
    else:
        # Use distance_km as score
        result.update({
            "xc_score": result.get("distance_km", 0.0),
            "xc_distance_km": result.get("distance_km", 0.0),
            "xc_type": "fallback_distance",
        })

    return result


def parse_with_libigc(file_path: str) -> Dict[str, Any]:
    """
    Parse IGC using libigc library.

    Args:
        file_path: path to IGC file

    Returns:
        dict with flight metrics
    """
    flight = LibIGCFlight.create_from_file(file_path)

    # Check if flight is valid and has takeoff/landing fixes
    if not flight.valid or not hasattr(flight, 'takeoff_fix') or not hasattr(flight, 'landing_fix'):
        # Flight parsing failed, raise exception to trigger fallback
        raise ValueError(f"Flight validation failed: {', '.join(flight.notes) if hasattr(flight, 'notes') else 'unknown error'}")

    # Extract basic metadata
    takeoff = flight.takeoff_fix
    landing = flight.landing_fix

    # Calculate altitude metrics
    altitudes = [fix.alt for fix in flight.fixes if hasattr(fix, 'alt')]
    takeoff_alt = takeoff.alt if takeoff and hasattr(takeoff, 'alt') else 0.0
    max_alt = max(altitudes) if altitudes else 0.0
    min_alt = min(altitudes) if altitudes else 0.0

    # Thermal metrics
    thermal_count = len(flight.thermals) if hasattr(flight, 'thermals') else 0
    glide_count = len(flight.glides) if hasattr(flight, 'glides') else 0

    # Calculate climb rates
    avg_climb_rate = 0.0
    max_climb_rate = 0.0
    if hasattr(flight, 'thermals') and flight.thermals:
        try:
            climb_rates = []
            for t in flight.thermals:
                if hasattr(t, 'vertical_velocity'):
                    vv = t.vertical_velocity()
                    if vv is not None:
                        climb_rates.append(vv)

            if climb_rates:
                avg_climb_rate = sum(climb_rates) / len(climb_rates)
                max_climb_rate = max(climb_rates)
        except Exception as e:
            print(f"Warning: Could not calculate climb rates: {e}", file=sys.stderr)

    # Calculate sink rates
    avg_sink_rate = 0.0
    if hasattr(flight, 'glides') and flight.glides:
        try:
            sink_rates = []
            for g in flight.glides:
                if hasattr(g, 'vertical_velocity'):
                    vv = g.vertical_velocity()
                    if vv is not None:
                        sink_rates.append(abs(vv))

            if sink_rates:
                avg_sink_rate = sum(sink_rates) / len(sink_rates)
        except Exception as e:
            print(f"Warning: Could not calculate sink rates: {e}", file=sys.stderr)

    # Build datetime strings
    takeoff_datetime = None
    landing_datetime = None
    if takeoff and hasattr(takeoff, 'timestamp') and takeoff.timestamp:
        if hasattr(takeoff.timestamp, 'isoformat'):
            takeoff_datetime = takeoff.timestamp.isoformat()
    if landing and hasattr(landing, 'timestamp') and landing.timestamp:
        if hasattr(landing.timestamp, 'isoformat'):
            landing_datetime = landing.timestamp.isoformat()

    # Calculate duration
    duration_sec = 0
    if takeoff and landing and hasattr(takeoff, 'timestamp') and hasattr(landing, 'timestamp'):
        if takeoff.timestamp and landing.timestamp:
            if hasattr(takeoff.timestamp, 'total_seconds') and hasattr(landing.timestamp, 'total_seconds'):
                try:
                    duration_sec = int((landing.timestamp - takeoff.timestamp).total_seconds())
                except (AttributeError, TypeError):
                    pass

    # Calculate simple distance (Haversine)
    distance_km = 0.0
    if takeoff and landing:
        try:
            from math import radians, sin, cos, sqrt, atan2
            R = 6371.0  # Earth radius in km

            lat1 = radians(takeoff.lat)
            lon1 = radians(takeoff.lon)
            lat2 = radians(landing.lat)
            lon2 = radians(landing.lon)

            dlat = lat2 - lat1
            dlon = lon2 - lon1

            a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
            c = 2 * atan2(sqrt(a), sqrt(1-a))
            distance_km = R * c
        except Exception as e:
            print(f"Warning: Could not calculate distance: {e}", file=sys.stderr)

    return {
        "takeoff_datetime": takeoff_datetime,
        "landing_datetime": landing_datetime,
        "takeoff_lat": takeoff.lat if takeoff else 0.0,
        "takeoff_lon": takeoff.lon if takeoff else 0.0,
        "landing_lat": landing.lat if landing else 0.0,
        "landing_lon": landing.lon if landing else 0.0,
        "takeoff_alt": takeoff_alt,
        "max_alt": max_alt,
        "min_alt": min_alt,
        "plaf": max_alt,  # plaf = ceiling = max_alt
        "duration_sec": duration_sec,
        "track_points": len(flight.fixes) if hasattr(flight, 'fixes') else 0,
        "distance_km": distance_km,
        "thermal_count": thermal_count,
        "glide_count": glide_count,
        "avg_climb_rate": avg_climb_rate,
        "max_climb_rate": max_climb_rate,
        "avg_sink_rate": avg_sink_rate,
    }


def parse_with_fallback(file_path: str) -> Dict[str, Any]:
    """
    Fallback parser using parse_igc_extended if libigc not available.

    ⚠️ DEPRECATED FALLBACK ⚠️
    This uses the deprecated parse_igc_extended.py module.
    Install libigc for better results.

    Args:
        file_path: path to IGC file

    Returns:
        dict with flight metrics
    """
    # Warning shown only once per import (via module-level warning in parse_igc_extended.py)
    if not HAS_EXTENDED_PARSER:
        print("Warning: parse_igc_extended not available, using minimal parsing", file=sys.stderr)
        # Minimal fallback
        return {
            "takeoff_datetime": None,
            "landing_datetime": None,
            "takeoff_lat": 0.0,
            "takeoff_lon": 0.0,
            "landing_lat": 0.0,
            "landing_lon": 0.0,
            "takeoff_alt": 0.0,
            "max_alt": 0.0,
            "min_alt": 0.0,
            "plaf": 0.0,
            "duration_sec": 0,
            "track_points": 0,
            "distance_km": 0.0,
            "thermal_count": 0,
            "glide_count": 0,
            "avg_climb_rate": 0.0,
            "max_climb_rate": 0.0,
            "avg_sink_rate": 0.0,
        }

    # Use parse_igc_extended functions
    try:
        # Parse B-records
        track = []
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line.startswith("B") and len(line) >= 35:
                    point = parse_b_record_extended(line)
                    if point:
                        track.append(point)

        # Calculate metrics
        altitude_stats = extract_altitude_stats(track)
        distance_km = calculate_distance(track)

        # Get flight date from basic meta
        flight_date = None
        try:
            basic = parse_igc(file_path)
            flight_date = basic.get("flight_date")
        except:
            pass

        takeoff_datetime = build_takeoff_datetime(flight_date, track)

        # Build result
        return {
            "takeoff_datetime": takeoff_datetime,
            "landing_datetime": None,  # Would need to calculate from last point
            "takeoff_lat": track[0].lat if track else 0.0,
            "takeoff_lon": track[0].lon if track else 0.0,
            "landing_lat": track[-1].lat if track else 0.0,
            "landing_lon": track[-1].lon if track else 0.0,
            "takeoff_alt": altitude_stats["takeoff_alt"],
            "max_alt": altitude_stats["max_alt"],
            "min_alt": altitude_stats["min_alt"],
            "plaf": altitude_stats["plaf"],
            "duration_sec": 0,  # Would need to calculate
            "track_points": len(track),
            "distance_km": distance_km,
            "thermal_count": 0,  # Not available in basic parser
            "glide_count": 0,
            "avg_climb_rate": 0.0,
            "max_climb_rate": 0.0,
            "avg_sink_rate": 0.0,
        }
    except Exception as e:
        print(f"Warning: Fallback parsing failed: {e}", file=sys.stderr)
        raise


def calculate_xc_score_safe(file_path: str) -> Dict[str, Any]:
    """
    Вычислить XC score используя xc_score library (with error handling).

    Args:
        file_path: путь к IGC файлу

    Returns:
        dict с score, distance_km, type
    """
    try:
        # Use process_igc_file from xc_score
        result = process_igc_file(file_path, DEFAULT_SCORING_RULES, optimization=True)

        # Extract relevant fields
        score = result.get("score", 0.0)
        flight_type = result.get("type", "unknown")

        # Get distance
        distance_km = 0.0
        if "properties" in result and "total_distance" in result["properties"]:
            distance_km = result["properties"]["total_distance"]
        elif "max_distance_info" in result:
            distance_km = result["max_distance_info"].get("max_distance", 0.0)

        # Build readable type
        xc_type = flight_type
        if flight_type == "triangle" and "triangle_type" in result:
            xc_type = f"{result['triangle_type']}_triangle"

        return {
            "score": score,
            "distance_km": distance_km,
            "type": xc_type
        }
    except Exception as e:
        print(f"Warning: XC score calculation failed: {e}", file=sys.stderr)
        return {
            "score": 0.0,
            "distance_km": 0.0,
            "type": "error"
        }


# For backward compatibility
def parse_igc_full(file_path: str) -> dict:
    """Alias for compatibility with existing code"""
    return parse_igc_with_libs(file_path)


if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        print("Usage: python parse_igc_with_libs.py <igc_file>", file=sys.stderr)
        sys.exit(1)

    igc_file = sys.argv[1]

    print(f"Parsing {igc_file}...", file=sys.stderr)
    print(f"libigc available: {HAS_LIBIGC}", file=sys.stderr)
    print(f"xc_score available: {HAS_XC_SCORE}", file=sys.stderr)
    print("", file=sys.stderr)

    try:
        result = parse_igc_with_libs(igc_file)
        print(json.dumps(result, indent=2, default=str))
    except Exception as e:
        print(f"Error parsing {igc_file}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
