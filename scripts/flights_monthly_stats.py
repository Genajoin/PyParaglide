#!/usr/bin/env python3
"""Статистика полётов по месяцам/кварталам/годам с топ-10 квадратов bbox стартов.

Использование:
    python flights_monthly_stats.py [--group-by month|quarter|year] [--source SOURCE] [--db-url URL]
    export IGC_DB_URL="postgresql://user:pass@host:5432/db"
    python flights_monthly_stats.py --group-by quarter
"""

import argparse
import os
import sys
from typing import List, Tuple, Dict


try:
    import psycopg
except Exception:
    psycopg = None


class Db:
    """PostgreSQL database wrapper."""

    def __init__(self, conn) -> None:
        self.conn = conn

    def execute(self, sql: str, params: Tuple = ()):
        """Execute SQL with ? placeholders (converted to %s for PostgreSQL)."""
        sql = sql.replace("?", "%s")
        return self.conn.execute(sql, params)

    def close(self):
        self.conn.close()


def connect_db(db_url: str) -> Db:
    """Connect to PostgreSQL database."""
    if psycopg is None:
        raise RuntimeError("psycopg is required. Install with: pip install psycopg[binary]")
    conn = psycopg.connect(db_url)
    return Db(conn)


def get_period_label(period_type: str, **kwargs) -> str:
    """Формирует метку периода для вывода."""
    if period_type == "month":
        return f"{kwargs['year']:04d}-{kwargs['month']:02d}"
    elif period_type == "quarter":
        return f"{kwargs['year']:04d}-Q{kwargs['quarter']}"
    elif period_type == "year":
        return f"{kwargs['year']:04d}"
    return "?"


def get_monthly_stats(db: Db, source: str = None) -> List[dict]:
    """Получить статистику полётов по месяцам.

    Returns:
        Список словарей с полями: year, month, count
    """
    sql = """
        SELECT
            SUBSTRING(flight_date, 1, 4)::int as year,
            SUBSTRING(flight_date, 6, 2)::int as month,
            COUNT(*) as count
        FROM flights
        WHERE flight_date IS NOT NULL
          AND flight_date ~ '^\d{4}-\d{2}-\d{2}'
    """
    params = []
    if source:
        sql += " AND source = ?"
        params = (source,)

    sql += """
        GROUP BY SUBSTRING(flight_date, 1, 4), SUBSTRING(flight_date, 6, 2)
        ORDER BY year DESC, month DESC
    """

    cur = db.execute(sql, tuple(params))
    return [
        {"year": row[0], "month": row[1], "count": row[2]}
        for row in cur.fetchall()
    ]


def get_quarterly_stats(db: Db, source: str = None) -> List[dict]:
    """Получить статистику полётов по кварталам.

    Returns:
        Список словарей с полями: year, quarter, count
    """
    sql = """
        SELECT
            SUBSTRING(flight_date, 1, 4)::int as year,
            CASE
                WHEN SUBSTRING(flight_date, 6, 2)::int <= 3 THEN 1
                WHEN SUBSTRING(flight_date, 6, 2)::int <= 6 THEN 2
                WHEN SUBSTRING(flight_date, 6, 2)::int <= 9 THEN 3
                ELSE 4
            END as quarter,
            COUNT(*) as count
        FROM flights
        WHERE flight_date IS NOT NULL
          AND flight_date ~ '^\d{4}-\d{2}-\d{2}'
    """
    params = []
    if source:
        sql += " AND source = ?"
        params = (source,)

    sql += """
        GROUP BY SUBSTRING(flight_date, 1, 4), quarter
        ORDER BY year DESC, quarter DESC
    """

    cur = db.execute(sql, tuple(params))
    return [
        {"year": row[0], "quarter": row[1], "count": row[2]}
        for row in cur.fetchall()
    ]


def get_yearly_stats(db: Db, source: str = None) -> List[dict]:
    """Получить статистику полётов по годам.

    Returns:
        Список словарей с полями: year, count
    """
    sql = """
        SELECT
            SUBSTRING(flight_date, 1, 4)::int as year,
            COUNT(*) as count
        FROM flights
        WHERE flight_date IS NOT NULL
          AND flight_date ~ '^\d{4}-\d{2}-\d{2}'
    """
    params = []
    if source:
        sql += " AND source = ?"
        params = (source,)

    sql += """
        GROUP BY SUBSTRING(flight_date, 1, 4)
        ORDER BY year DESC
    """

    cur = db.execute(sql, tuple(params))
    return [
        {"year": row[0], "count": row[1]}
        for row in cur.fetchall()
    ]


def get_bbox_stats_month(db: Db, year: int, month: int, source: str = None, limit: int = 10) -> List[dict]:
    """Получить топ-N квадратов bbox для указанного месяца."""
    year_str = f"{year:04d}"
    month_str = f"{month:02d}"

    sql = """
        SELECT
            FLOOR(takeoff_lat) as lat_min,
            FLOOR(takeoff_lon) as lon_min,
            COUNT(*) as count
        FROM flights
        WHERE flight_date IS NOT NULL
            AND flight_date ~ '^\\d{4}-\\d{2}-\\d{2}'
            AND takeoff_lat IS NOT NULL
            AND takeoff_lon IS NOT NULL
            AND SUBSTRING(flight_date, 1, 4) = ?
            AND SUBSTRING(flight_date, 6, 2) = ?
    """
    params = [year_str, month_str]

    if source:
        sql += " AND source = ?"
        params.append(source)

    sql += f"""
        GROUP BY FLOOR(takeoff_lat), FLOOR(takeoff_lon)
        ORDER BY count DESC
        LIMIT {limit}
    """

    cur = db.execute(sql, tuple(params))
    return [
        {
            "bbox": f"[{row[0]:.0f},{row[1]:.0f},{row[0]+1:.0f},{row[1]+1:.0f}]",
            "lat_min": row[0],
            "lon_min": row[1],
            "count": row[2],
        }
        for row in cur.fetchall()
    ]


def get_bbox_stats_quarter(db: Db, year: int, quarter: int, source: str = None, limit: int = 10) -> List[dict]:
    """Получить топ-N квадратов bbox для указанного квартала."""
    year_str = f"{year:04d}"

    # Определяем месяцы квартала
    month_start = (quarter - 1) * 3 + 1
    month_end = quarter * 3

    sql = """
        SELECT
            FLOOR(takeoff_lat) as lat_min,
            FLOOR(takeoff_lon) as lon_min,
            COUNT(*) as count
        FROM flights
        WHERE flight_date IS NOT NULL
            AND flight_date ~ '^\\d{4}-\\d{2}-\\d{2}'
            AND takeoff_lat IS NOT NULL
            AND takeoff_lon IS NOT NULL
            AND SUBSTRING(flight_date, 1, 4) = ?
            AND SUBSTRING(flight_date, 6, 2)::int BETWEEN ? AND ?
    """
    params = [year_str, month_start, month_end]

    if source:
        sql += " AND source = ?"
        params.append(source)

    sql += f"""
        GROUP BY FLOOR(takeoff_lat), FLOOR(takeoff_lon)
        ORDER BY count DESC
        LIMIT {limit}
    """

    cur = db.execute(sql, tuple(params))
    return [
        {
            "bbox": f"[{row[0]:.0f},{row[1]:.0f},{row[0]+1:.0f},{row[1]+1:.0f}]",
            "lat_min": row[0],
            "lon_min": row[1],
            "count": row[2],
        }
        for row in cur.fetchall()
    ]


def get_bbox_stats_year(db: Db, year: int, source: str = None, limit: int = 10) -> List[dict]:
    """Получить топ-N квадратов bbox для указанного года."""
    year_str = f"{year:04d}"

    sql = """
        SELECT
            FLOOR(takeoff_lat) as lat_min,
            FLOOR(takeoff_lon) as lon_min,
            COUNT(*) as count
        FROM flights
        WHERE flight_date IS NOT NULL
            AND flight_date ~ '^\\d{4}-\\d{2}-\\d{2}'
            AND takeoff_lat IS NOT NULL
            AND takeoff_lon IS NOT NULL
            AND SUBSTRING(flight_date, 1, 4) = ?
    """
    params = [year_str]

    if source:
        sql += " AND source = ?"
        params.append(source)

    sql += f"""
        GROUP BY FLOOR(takeoff_lat), FLOOR(takeoff_lon)
        ORDER BY count DESC
        LIMIT {limit}
    """

    cur = db.execute(sql, tuple(params))
    return [
        {
            "bbox": f"[{row[0]:.0f},{row[1]:.0f},{row[0]+1:.0f},{row[1]+1:.0f}]",
            "lat_min": row[0],
            "lon_min": row[1],
            "count": row[2],
        }
        for row in cur.fetchall()
    ]


def format_bbox_line(b: dict) -> str:
    """Форматировать одну строку bbox."""
    ns = "N" if b['lat_min'] >= 0 else "S"
    ew = "E" if b['lon_min'] >= 0 else "W"
    return (f"  {b['bbox']:<20}  ({abs(b['lat_min']):.0f}°{ns}..{abs(b['lat_min']+1):.0f}°{ns}, "
            f"{abs(b['lon_min']):.0f}°{ew}..{abs(b['lon_min']+1):.0f}°{ew})  →  {b['count']} полёт(ов)")


def print_stats(db: Db, group_by: str, source: str = None, bbox_limit: int = 10):
    """Вывести статистику с топ bbox."""

    if group_by == "month":
        stats = get_monthly_stats(db, source)
        period_name = "Месяц"
    elif group_by == "quarter":
        stats = get_quarterly_stats(db, source)
        period_name = "Квартал"
    elif group_by == "year":
        stats = get_yearly_stats(db, source)
        period_name = "Год"
    else:
        raise ValueError(f"Неверный group_by: {group_by}")

    if not stats:
        print("Нет данных о полётах")
        return

    print(f"{period_name:<15} {'Полётов':>10}   Топ-{bbox_limit} bbox (lat_min,lon_min → count)")
    print("=" * 100)

    for period in stats:
        count = period["count"]
        label = get_period_label(group_by, **period)

        # Получаем bbox для периода
        if group_by == "month":
            bboxes = get_bbox_stats_month(db, period["year"], period["month"], source, limit=bbox_limit)
        elif group_by == "quarter":
            bboxes = get_bbox_stats_quarter(db, period["year"], period["quarter"], source, limit=bbox_limit)
        else:  # year
            bboxes = get_bbox_stats_year(db, period["year"], source, limit=bbox_limit)

        # Краткая строка bbox
        bbox_str = ", ".join(
            f"({b['lat_min']:.0f},{b['lon_min']:.0f}→{b['count']})"
            for b in bboxes
        )
        if len(bbox_str) > 65:
            bbox_str = bbox_str[:62] + "..."

        print(f"{label:<15} {count:>10}   {bbox_str}")

        # Детально по каждому bbox
        if bboxes:
            print("  " + "─" * 96)
            for b in bboxes:
                print(format_bbox_line(b))
        else:
            print("  (нет данных с координатами)")
        print()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Статистика полётов по периодам с топ-N квадратов bbox стартов",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  # По месяцам
  %(prog)s --group-by month --source skygr
  # По кварталам
  %(prog)s --group-by quarter --source skygr --top 5
  # По годам
  %(prog)s --group-by year --source skygr
        """
    )
    parser.add_argument(
        "--group-by",
        choices=["month", "quarter", "year"],
        default="month",
        help="Уровень агрегации: month (по умолчанию), quarter, year",
    )
    parser.add_argument(
        "--source",
        help="Фильтр по источнику (paraplan, skygr, etc.)",
    )
    parser.add_argument(
        "--db-url",
        default=os.environ.get("IGC_DB_URL", "postgresql://paraglidable:paraglidable@localhost:5432/paraglidable"),
        help="PostgreSQL connection URL (default: from IGC_DB_URL env or localhost)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=10,
        help="Количество топовых bbox для вывода (default: 10)",
    )
    args = parser.parse_args()

    try:
        db = connect_db(args.db_url)
        print_stats(db, group_by=args.group_by, source=args.source, bbox_limit=args.top)
        db.close()
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
