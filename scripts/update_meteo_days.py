#!/usr/bin/env python3
"""
Обновляет meteo_days.pkl на основе доступных GFS файлов в data/gfs/anl/.
"""

import pickle
import re
import os
from pathlib import Path
from datetime import date
from collections import defaultdict

# Load environment variables (optional)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).parent.parent))
GFS_DIR = PROJECT_ROOT / os.environ.get("GFS_DIR", "data/gfs/anl")
PKL_DIR = PROJECT_ROOT / os.environ.get("PKL_DIR", "neural_network/bin/data")


def scan_gfs_dates(gfs_dir):
    """Сканирует директорию с GFS файлами и возвращает список дней с полными данными."""
    # Шаблон имени файла: gfsanl_3_YYYYMMDD_HH00_000.grb2
    # Нас интересуют файлы за 06:00, 12:00, 18:00 UTC

    files_by_date = defaultdict(set)

    for month_dir in gfs_dir.iterdir():
        if not month_dir.is_dir():
            continue

        # Парсим год и месяц из имени директории (например, "2023-06")
        match = re.match(r'(\d{4})-(\d{2})', month_dir.name)
        if not match:
            continue

        year, month = int(match.group(1)), int(match.group(2))

        # Сканируем файлы
        for grib_file in month_dir.glob("*.grb2"):
            # Извлекаем дату и час из имени файла
            match = re.match(r'gfsanl_3_(\d{8})_(\d{2})00_000\.grb2', grib_file.name)
            if match:
                day_str = match.group(1)
                hour = int(match.group(2))

                try:
                    day_date = date(year, month, int(day_str[6:8]))
                    files_by_date[day_date].add(hour)
                except ValueError:
                    continue

    # Включаем только дни, где есть все 3 часа: 06, 12, 18
    complete_days = sorted([d for d, hours in files_by_date.items()
                            if {6, 12, 18}.issubset(hours)])

    return complete_days


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Обновляет meteo_days.pkl на основе GFS файлов')
    parser.add_argument('--rebuild', action='store_true',
                       help='Пересоздать файл с нуля (только на основе файлов на диске)')
    args = parser.parse_args()

    print(f"Сканирование GFS файлов в: {GFS_DIR}")

    # Считаем существующие метоо дни
    existing_days = []
    if (PKL_DIR / "meteo_days.pkl").exists():
        with open(PKL_DIR / "meteo_days.pkl", 'rb') as f:
            existing_days = pickle.load(f, encoding='latin1')
        print(f"Существующих дней в meteo_days.pkl: {len(existing_days)}")
        if existing_days:
            print(f"  Период: {existing_days[0]} - {existing_days[-1]}")

    # Сканируем GFS файлы
    scanned_days = scan_gfs_dates(GFS_DIR)
    print(f"Найдено дней с полными GFS данными (06, 12, 18): {len(scanned_days)}")
    if scanned_days:
        print(f"  Период: {scanned_days[0]} - {scanned_days[-1]}")

    # Проверяем на потерянные дни
    existing_set = set(existing_days)
    scanned_set = set(scanned_days)
    missing_on_disk = existing_set - scanned_set
    if missing_on_disk:
        print(f"  ВНИМАНИЕ: {len(missing_on_disk)} дней из pkl отсутствуют на диске")
        if not args.rebuild:
            print(f"    (для удаления используйте --rebuild)")

    if args.rebuild:
        # Пересоздаём с нуля
        all_days = scanned_days
        print(f"\nРежим --rebuild: файл пересоздан с нуля")
    else:
        # Объединяем дни (удаляем дубликаты)
        all_days = sorted(set(existing_days + scanned_days))

    # Сохраняем обновлённый файл
    with open(PKL_DIR / "meteo_days.pkl", 'wb') as f:
        pickle.dump(all_days, f)

    print(f"\nОбновлённый meteo_days.pkl сохранён:")
    print(f"  Всего дней: {len(all_days)}")
    print(f"  Период: {all_days[0]} - {all_days[-1]}")
    if args.rebuild:
        print(f"  Удалено дней (отсутствуют на диске): {len(missing_on_disk)}")
    else:
        print(f"  Добавлено новых дней: {len(all_days) - len(existing_days)}")


if __name__ == "__main__":
    main()
