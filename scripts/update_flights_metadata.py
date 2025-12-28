#!/usr/bin/env python3
"""
Update flight metadata from IGC files.

This script:
1. Adds new columns to the flights table (if not exist)
2. Reads IGC files from disk
3. Parses them using parse_igc_with_libs.py (libigc + xc_score)
4. Updates database with new metadata (XC score, altitudes, thermals, etc.)

Supports incremental updates (only flights where xc_score IS NULL).
"""

import argparse
import os
import sys
from multiprocessing import Pool, cpu_count
from typing import Optional
from tqdm import tqdm

# Import parser
try:
    from parse_igc_with_libs import parse_igc_full
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from parse_igc_with_libs import parse_igc_full

# Import DB connection utilities
try:
    from igc_ingest_skygr import connect_db, Db
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from igc_ingest_skygr import connect_db, Db


def parse_single_flight(args):
    """
    Worker функция для параллельного парсинга одного IGC файла.

    Args:
        args: (flight_id, igc_path) кортеж

    Returns:
        (flight_id, meta_dict, error_msg) кортеж
    """
    flight_id, igc_path = args

    # Импорты внутри функции (требование multiprocessing)
    import os
    from parse_igc_with_libs import parse_igc_full

    # Проверка существования файла
    if not os.path.exists(igc_path):
        return (flight_id, None, f"File not found: {igc_path}")

    # Парсинг
    try:
        meta = parse_igc_full(igc_path)
        return (flight_id, meta, None)
    except Exception as e:
        return (flight_id, None, str(e))


def update_db_schema(db: Db) -> None:
    """
    Добавить новые поля в таблицу flights.

    Adds columns:
    - takeoff_datetime TEXT
    - landing_datetime TEXT
    - takeoff_alt DOUBLE PRECISION
    - max_alt DOUBLE PRECISION
    - plaf DOUBLE PRECISION
    - min_alt DOUBLE PRECISION
    - distance_km DOUBLE PRECISION (if not exists)
    - xc_score DOUBLE PRECISION (XC score points)
    - xc_distance_km DOUBLE PRECISION (XC distance)
    - xc_type TEXT (flight type: triangle, free_distance, etc)
    - thermal_count INTEGER
    - glide_count INTEGER
    - avg_climb_rate DOUBLE PRECISION
    - max_climb_rate DOUBLE PRECISION

    Args:
        db: database connection (PostgreSQL)
    """
    print("Updating database schema...", flush=True)

    sql_additions = """
    ALTER TABLE flights ADD COLUMN IF NOT EXISTS takeoff_datetime TEXT;
    ALTER TABLE flights ADD COLUMN IF NOT EXISTS landing_datetime TEXT;
    ALTER TABLE flights ADD COLUMN IF NOT EXISTS takeoff_alt DOUBLE PRECISION;
    ALTER TABLE flights ADD COLUMN IF NOT EXISTS max_alt DOUBLE PRECISION;
    ALTER TABLE flights ADD COLUMN IF NOT EXISTS plaf DOUBLE PRECISION;
    ALTER TABLE flights ADD COLUMN IF NOT EXISTS min_alt DOUBLE PRECISION;
    ALTER TABLE flights ADD COLUMN IF NOT EXISTS distance_km DOUBLE PRECISION;
    ALTER TABLE flights ADD COLUMN IF NOT EXISTS xc_score DOUBLE PRECISION;
    ALTER TABLE flights ADD COLUMN IF NOT EXISTS xc_distance_km DOUBLE PRECISION;
    ALTER TABLE flights ADD COLUMN IF NOT EXISTS xc_type TEXT;
    ALTER TABLE flights ADD COLUMN IF NOT EXISTS thermal_count INTEGER;
    ALTER TABLE flights ADD COLUMN IF NOT EXISTS glide_count INTEGER;
    ALTER TABLE flights ADD COLUMN IF NOT EXISTS avg_climb_rate DOUBLE PRECISION;
    ALTER TABLE flights ADD COLUMN IF NOT EXISTS max_climb_rate DOUBLE PRECISION;
    """

    for stmt in sql_additions.strip().split(';'):
        if stmt.strip():
            try:
                db.execute(stmt.strip())
            except Exception as e:
                print(f"Warning: {e}", file=sys.stderr)

    db.commit()
    print("Schema updated successfully", flush=True)


def update_flights_from_igc(
    db: Db,
    source: str = "skygr",
    max_files: Optional[int] = None,
    incremental: bool = True,
    workers: Optional[int] = None
) -> int:
    """
    Обновить метаданные полётов из IGC файлов.

    Args:
        db: database connection (PostgreSQL)
        source: источник данных ("skygr", etc)
        max_files: лимит файлов для обработки (None = все)
        incremental: только записи где xc_score IS NULL
        workers: количество процессов (None = auto)

    Returns:
        количество обновлённых записей
    """
    # Построить WHERE условие
    where_clause = "source=? AND igc_path IS NOT NULL AND status IN ('downloaded', 'parsed')"
    if incremental:
        where_clause += " AND xc_score IS NULL"

    # Запрос файлов для обработки
    query = f"""
        SELECT flight_id, igc_path, flight_date
        FROM flights
        WHERE {where_clause}
        ORDER BY id ASC
    """

    if max_files:
        query += f" LIMIT {max_files}"

    cursor = db.execute(query, (source,))
    rows = cursor.fetchall()

    if not rows:
        print("No flights to update", flush=True)
        return 0

    # Определить количество процессов
    if workers is None:
        workers = min(cpu_count(), len(rows))
    else:
        workers = min(workers, len(rows))

    print(f"Processing {len(rows)} flights with {workers} workers...", flush=True)
    print("Press Ctrl+C to gracefully stop (waits for current files to finish)", flush=True)

    # Подготовить аргументы для worker-ов
    parse_args = [(row[0], row[1]) for row in rows]

    # Параллельный парсинг с корректной обработкой прерывания и прогресс-баром
    pool = Pool(processes=workers)
    try:
        results = []
        success_count = 0
        error_count = 0

        with tqdm(total=len(parse_args), desc="Parsing", unit="flight") as pbar:
            for result in pool.imap_unordered(parse_single_flight, parse_args):
                results.append(result)
                if result[2]:  # error
                    error_count += 1
                    pbar.set_postfix(errors=error_count)
                else:
                    success_count += 1
                pbar.update(1)

    except KeyboardInterrupt:
        print("\nInterrupted! Waiting for current tasks to complete...", flush=True)
        raise
    finally:
        # close() предотвращает добавление новых задач
        # join() ждёт завершения текущих задач
        pool.close()
        pool.join()

    # Подготовить SQL для обновления
    update_sql = """
        UPDATE flights
        SET
            takeoff_datetime=?,
            landing_datetime=?,
            takeoff_alt=?,
            max_alt=?,
            plaf=?,
            min_alt=?,
            distance_km=?,
            xc_score=?,
            xc_distance_km=?,
            xc_type=?,
            thermal_count=?,
            glide_count=?,
            avg_climb_rate=?,
            max_climb_rate=?,
            updated_at=NOW()
        WHERE source=? AND flight_id=?
    """

    # Обновить БД результатами
    updated_count = 0
    db_error_count = 0
    parse_error_count = sum(1 for _, _, e in results if e)

    if parse_error_count > 0:
        print(f"Parse errors: {parse_error_count} flights", flush=True)

    with tqdm(total=len(results), desc="Updating DB", unit="rec") as pbar:
        for flight_id, meta, error in results:
            pbar.update(1)

            if error:
                continue  # уже посчитали выше

            try:
                db.execute(
                    update_sql,
                    (
                        meta.get("takeoff_datetime"),
                        meta.get("landing_datetime"),
                        meta.get("takeoff_alt"),
                        meta.get("max_alt"),
                        meta.get("plaf"),
                        meta.get("min_alt"),
                        meta.get("distance_km"),
                        meta.get("xc_score"),
                        meta.get("xc_distance_km"),
                        meta.get("xc_type"),
                        meta.get("thermal_count"),
                        meta.get("glide_count"),
                        meta.get("avg_climb_rate"),
                        meta.get("max_climb_rate"),
                        source,
                        flight_id
                    )
                )

                updated_count += 1

                # Commit каждые 100 записей
                if updated_count % 100 == 0:
                    db.commit()
                    pbar.set_postfix(committed=updated_count)

            except Exception as e:
                db_error_count += 1
                print(f"Error updating flight {flight_id}: {e}", file=sys.stderr, flush=True)
                continue

    # Финальный commit
    db.commit()

    print(f"Successfully updated: {updated_count} flights", flush=True)
    if parse_error_count > 0:
        print(f"Parse errors: {parse_error_count} flights", flush=True)
    if db_error_count > 0:
        print(f"DB errors: {db_error_count} flights", flush=True)

    return updated_count


def print_stats(db: Db, source: str) -> None:
    """
    Напечатать статистику по обновленным данным.

    Args:
        db: database connection (PostgreSQL)
        source: источник данных
    """
    print("\nStatistics:", flush=True)

    # Общая статистика
    query = """
        SELECT
            COUNT(*) as total,
            COUNT(distance_km) as with_distance,
            COUNT(takeoff_alt) as with_takeoff_alt,
            COUNT(plaf) as with_plaf,
            COUNT(takeoff_datetime) as with_datetime
        FROM flights
        WHERE source=?
    """

    cursor = db.execute(query, (source,))
    row = cursor.fetchone()

    if row:
        print(f"  Total flights: {row[0]}", flush=True)
        print(f"  With distance_km: {row[1]}", flush=True)
        print(f"  With takeoff_alt: {row[2]}", flush=True)
        print(f"  With plaf: {row[3]}", flush=True)
        print(f"  With takeoff_datetime: {row[4]}", flush=True)

    # Примеры обновленных данных
    print("\nSample updated flights:", flush=True)
    sample_query = """
        SELECT
            flight_id,
            flight_date,
            takeoff_datetime,
            takeoff_alt,
            max_alt,
            plaf,
            distance_km
        FROM flights
        WHERE source=? AND distance_km IS NOT NULL
        ORDER BY id DESC
        LIMIT 5
    """

    cursor = db.execute(sample_query, (source,))
    rows = cursor.fetchall()

    if rows:
        for row in rows:
            print(f"  {row[0]}: date={row[1]}, datetime={row[2]}, "
                  f"takeoff_alt={row[3]:.1f}m, max_alt={row[4]:.1f}m, "
                  f"plaf={row[5]:.1f}m, distance={row[6]:.1f}km",
                  flush=True)
    else:
        print("  No updated flights found", flush=True)


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Update flight metadata from IGC files"
    )
    parser.add_argument("--db-url",
                       default=os.environ.get("IGC_DB_URL", "postgresql://paraglidable:paraglidable@localhost:5432/paraglidable"),
                       help="PostgreSQL connection URL (default: from IGC_DB_URL env or localhost)")
    parser.add_argument("--source", default="skygr",
                       help="Flight source to update (default: skygr)")
    parser.add_argument("--max-files", type=int, default=None,
                       help="Maximum number of files to process (default: all)")
    parser.add_argument("--full", action="store_true",
                       help="Full reparse (update all flights, not just incremental)")
    parser.add_argument("--stats", action="store_true",
                       help="Show statistics after update")
    parser.add_argument("--workers", type=int, default=None,
                       help="Number of parallel workers (default: auto = CPU count)")

    args = parser.parse_args()

    # Подключение к БД
    print(f"Connecting to database...", flush=True)
    try:
        db = connect_db(args.db_url, None)
    except Exception as e:
        print(f"ERROR: Failed to connect to database: {e}", file=sys.stderr)
        return 1

    try:
        # Обновить схему БД
        update_db_schema(db)

        # Обновить метаданные
        print(f"\nUpdating flight metadata (source={args.source}, "
              f"incremental={not args.full})...", flush=True)
        count = update_flights_from_igc(
            db,
            source=args.source,
            max_files=args.max_files,
            incremental=not args.full,
            workers=args.workers
        )

        if count == 0:
            print("\nNo flights updated. Use --full to reparse all flights.", flush=True)

        # Показать статистику если запрошено
        if args.stats or count > 0:
            print_stats(db, args.source)

        return 0

    except KeyboardInterrupt:
        print("\nInterrupted by user", flush=True)
        return 130

    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    finally:
        db.close()


if __name__ == "__main__":
    raise SystemExit(main())
