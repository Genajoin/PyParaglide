#!/usr/bin/env python3
"""
Extended IGC parser for extracting all flight metadata including altitude stats and distance.

⚠️ DEPRECATED ⚠️
================
This module is DEPRECATED. Use `parse_igc_with_libs.py` instead.

`parse_igc_with_libs.py` provides:
- libigc for flight analysis (thermals, glides, statistics)
- xc_score for accurate XC score calculation (FAI triangles, free distance)
- Better error handling

This module is kept only as a fallback for when libigc/xc_score are not available.

---

This module extends the basic IGC parsing from igc_ingest_skygr.py with:
- Full track parsing (all B-records)
- Altitude statistics (takeoff_alt, max_alt, plaf, min_alt)
- Distance calculation using Haversine formula
- Exact takeoff datetime
"""

import math
import sys
import warnings
from dataclasses import dataclass
from typing import Optional, List, Tuple


# Emit deprecation warning when module is imported
warnings.warn(
    "parse_igc_extended.py is DEPRECATED. Use parse_igc_with_libs.py instead. "
    "This module will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

# Import base parsing functions from igc_ingest_skygr
try:
    from igc_ingest_skygr import (
        parse_igc,
        parse_igc_date,
        parse_hhmmss,
        parse_lat,
        parse_lon,
    )
except ImportError:
    # Fallback if running from different directory
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from igc_ingest_skygr import (
        parse_igc,
        parse_igc_date,
        parse_hhmmss,
        parse_lat,
        parse_lon,
    )


@dataclass
class TrackPoint:
    """Точка трека из B-записи IGC файла"""
    time_sec: int          # Секунды с начала дня (0-86399)
    lat: float             # Градусы, -90..90
    lon: float             # Градусы, -180..180
    alt: float             # Метры (приоритет: baro > GPS)
    alt_source: str        # "baro", "gps", "none"
    fix_valid: bool        # True если fix='A'


def parse_altitude_value(s: str) -> Optional[float]:
    """
    Парсит значение высоты из 5-символьной строки.

    Формат: PPPPP или -PPPP (отрицательная высота)

    Examples:
        "01234" -> 1234.0
        "-0050" -> -50.0
        "     " -> None

    Args:
        s: 5-символьная строка высоты из B-записи

    Returns:
        Значение высоты в метрах или None если невалидно
    """
    s = s.strip()
    if not s:
        return None

    # Проверка на отрицательное значение
    negative = s.startswith('-')
    if negative:
        s = s[1:]

    # Проверка что остались только цифры
    if not s.isdigit():
        return None

    alt = float(s)
    return -alt if negative else alt


def parse_altitude(baro_str: str, gps_str: str) -> Tuple[float, str]:
    """
    Извлечь высоту из барометрической или GPS строки.

    Приоритет: baro > GPS > 0.0

    Args:
        baro_str: 5 символов [25:30] из B-записи
        gps_str: 5 символов [30:35] из B-записи

    Returns:
        Tuple (altitude: float, source: str)
        source in ["baro", "gps", "none"]
    """
    # Попытка парсить барометрическую высоту
    baro_alt = parse_altitude_value(baro_str)
    if baro_alt is not None:
        return (baro_alt, "baro")

    # Fallback на GPS высоту
    gps_alt = parse_altitude_value(gps_str)
    if gps_alt is not None:
        return (gps_alt, "gps")

    # Нет данных о высоте
    return (0.0, "none")


def parse_b_record(line: str) -> Optional[TrackPoint]:
    """
    Парсит одну B-запись IGC файла.

    Формат: B HHMMSS DDMMmmmN DDDMMmmmE V PPPPP GGGGG
            [0][1:7] [7:15]   [15:24]  [24][25:30][30:35]

    Args:
        line: строка IGC начинающаяся с 'B'

    Returns:
        TrackPoint или None если запись невалидна
    """
    if len(line) < 35:
        return None

    # Парсинг времени [1:7]
    time_str = line[1:7]
    if not time_str.isdigit():
        return None
    time_sec = parse_hhmmss(time_str)

    # Парсинг координат [7:15], [15:24]
    lat = parse_lat(line[7:15])
    lon = parse_lon(line[15:24])
    if lat is None or lon is None:
        return None

    # Fix validity [24]
    fix_valid = line[24:25] == 'A'

    # Парсинг высот [25:30], [30:35]
    baro_str = line[25:30]
    gps_str = line[30:35]
    alt, alt_source = parse_altitude(baro_str, gps_str)

    return TrackPoint(
        time_sec=time_sec,
        lat=lat,
        lon=lon,
        alt=alt,
        alt_source=alt_source,
        fix_valid=fix_valid
    )


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Вычислить расстояние между двумя точками на сфере (Earth).

    Использует Haversine formula.

    Args:
        lat1, lon1: координаты первой точки (градусы)
        lat2, lon2: координаты второй точки (градусы)

    Returns:
        расстояние в километрах
    """
    R = 6371.0  # Радиус Земли в км

    # Конвертация в радианы
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)

    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) *
         math.sin(dlon / 2) ** 2)

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance


def calculate_distance(track: List[TrackPoint]) -> float:
    """
    Вычислить общую дистанцию по треку используя Haversine formula.

    Учитывает только валидные точки (fix_valid=True) для точности.

    Args:
        track: список точек трека

    Returns:
        дистанция в километрах
    """
    if len(track) < 2:
        return 0.0

    # Фильтр: только валидные fix
    valid_points = [p for p in track if p.fix_valid]
    if len(valid_points) < 2:
        # Fallback: использовать все точки если нет валидных
        valid_points = track

    total_distance = 0.0
    for i in range(len(valid_points) - 1):
        p1 = valid_points[i]
        p2 = valid_points[i + 1]

        # Haversine formula
        distance = haversine(p1.lat, p1.lon, p2.lat, p2.lon)
        total_distance += distance

    return total_distance


def extract_altitude_stats(track: List[TrackPoint]) -> dict:
    """
    Извлечь статистику высот из трека.

    Returns:
        dict с полями:
            - takeoff_alt: float - высота первой валидной точки
            - max_alt: float - максимальная высота
            - min_alt: float - минимальная высота
            - plaf: float - ceiling (= max_alt)
            - altitude_source: str - "baro", "gps", "mixed", "none"
    """
    if not track:
        return {
            "takeoff_alt": 0.0,
            "max_alt": 0.0,
            "min_alt": 0.0,
            "plaf": 0.0,
            "altitude_source": "none"
        }

    # Найти первую валидную точку для takeoff_alt
    takeoff_alt = 0.0
    for point in track:
        if point.fix_valid and point.alt_source != "none":
            takeoff_alt = point.alt
            break

    # Если нет валидных точек, взять первую доступную
    if takeoff_alt == 0.0 and track:
        takeoff_alt = track[0].alt

    # Вычислить min/max только из точек с высотой
    altitudes = [p.alt for p in track if p.alt_source != "none"]

    if altitudes:
        max_alt = max(altitudes)
        min_alt = min(altitudes)
    else:
        max_alt = 0.0
        min_alt = 0.0

    # Определить источник высот
    sources = set(p.alt_source for p in track)
    if "baro" in sources and "gps" in sources:
        altitude_source = "mixed"
    elif "baro" in sources:
        altitude_source = "baro"
    elif "gps" in sources:
        altitude_source = "gps"
    else:
        altitude_source = "none"

    return {
        "takeoff_alt": takeoff_alt,
        "max_alt": max_alt,
        "min_alt": min_alt,
        "plaf": max_alt,  # plaf = ceiling = max_alt
        "altitude_source": altitude_source
    }


def build_takeoff_datetime(flight_date: str, track: List[TrackPoint]) -> Optional[str]:
    """
    Построить полное время старта из даты и первой B-записи.

    Args:
        flight_date: "YYYY-MM-DD"
        track: список точек трека

    Returns:
        "YYYY-MM-DD HH:MM:SS" или None если нет данных
    """
    if not flight_date or not track:
        return None

    # Найти первую валидную точку
    first_point = None
    for point in track:
        if point.fix_valid:
            first_point = point
            break

    # Fallback на первую точку если нет валидных
    if not first_point and track:
        first_point = track[0]

    if not first_point:
        return None

    # Конвертировать time_sec в HH:MM:SS
    hours = first_point.time_sec // 3600
    minutes = (first_point.time_sec % 3600) // 60
    seconds = first_point.time_sec % 60

    return f"{flight_date} {hours:02d}:{minutes:02d}:{seconds:02d}"


def parse_igc_full(file_path: str) -> dict:
    """
    Полный парсинг IGC файла с извлечением всех метрик.

    Returns:
        dict с полями:
        Базовые (из parse_igc):
            - flight_date: str
            - pilot: str
            - glider: str
            - glider_class: str
            - takeoff_lat: float
            - takeoff_lon: float
            - duration_sec: int
            - track_points: int
            - has_baro_alt: bool
            - has_gps_alt: bool

        Новые:
            - takeoff_datetime: str - "YYYY-MM-DD HH:MM:SS"
            - takeoff_alt: float - высота первой точки
            - max_alt: float - максимальная высота
            - plaf: float - ceiling (= max_alt)
            - min_alt: float - минимальная высота
            - distance_km: float - дистанция по треку
            - altitude_source: str - "baro", "gps", "mixed", "none"
    """
    # 1. Базовый парсинг через существующую функцию
    meta = parse_igc(file_path)

    # 2. Парсить трек (все B-записи)
    track = []
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("B") and len(line) >= 35:
                point = parse_b_record(line)
                if point:
                    track.append(point)

    # 3. Вычислить altitude статистику
    altitude_stats = extract_altitude_stats(track)

    # 4. Вычислить distance_km
    distance_km = calculate_distance(track)

    # 5. Построить takeoff_datetime
    takeoff_datetime = build_takeoff_datetime(meta.get("flight_date"), track)

    # 6. Объединить результаты
    result = {
        **meta,  # Базовые поля
        "takeoff_datetime": takeoff_datetime,
        "takeoff_alt": altitude_stats["takeoff_alt"],
        "max_alt": altitude_stats["max_alt"],
        "plaf": altitude_stats["plaf"],
        "min_alt": altitude_stats["min_alt"],
        "distance_km": distance_km,
        "altitude_source": altitude_stats["altitude_source"]
    }

    return result


# Для тестирования
if __name__ == "__main__":
    import json

    if len(sys.argv) < 2:
        print("Usage: python parse_igc_extended.py <igc_file>", file=sys.stderr)
        sys.exit(1)

    igc_file = sys.argv[1]
    try:
        result = parse_igc_full(igc_file)
        print(json.dumps(result, indent=2, default=str))
    except Exception as e:
        print(f"Error parsing {igc_file}: {e}", file=sys.stderr)
        raise
