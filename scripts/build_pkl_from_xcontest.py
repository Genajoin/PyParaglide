#!/usr/bin/env python3
"""
Создаёт PKL файлы для обучения SPOTS модели из xcontest данных.

Генерирует:
- spots.pkl: [('name', lat, lon), ...]
- spots_by_cell.pkl: [[spot_id, ...], ...]
- flights_by_spot.pkl: [(datetime, (score, None, takeoff_alt, lat, lon)), ...]
- flights_by_cell_day_spot.pkl: [[[{spot_id: [...]}, ...], ...], ...]
"""

import argparse
import json
import os
import pickle
from datetime import datetime
from pathlib import Path
from collections import defaultdict

# Константы
THRESHOLD_FLIGHTS = 200  # Минимум полётов на spot для обучения
PROJECT_ROOT = Path("/workspaces/Paraglidable")
MERGED_DIR = PROJECT_ROOT / "data" / "flights" / "merged"
PKL_DIR = PROJECT_ROOT / "neural_network" / "bin" / "data"


def parse_date_ranges(dates_str: str):
    """Парсит строки диапазонов дат типа '2021-06-01:2021-09-30,2022-06-01:2022-09-30'

    Returns: Список кортежей (start_date, end_date) или None для без фильтрации
    """
    if not dates_str:
        return None

    ranges = []
    for part in dates_str.split(','):
        part = part.strip()
        if ':' not in part:
            raise ValueError(f"Неверный диапазон дат: {part}. Ожидалось: YYYY-MM-DD:YYYY-MM-DD")
        start_str, end_str = part.split(':')
        start = datetime.strptime(start_str.strip(), '%Y-%m-%d').date()
        end = datetime.strptime(end_str.strip(), '%Y-%m-%d').date()
        if start > end:
            raise ValueError(f"Начальная дата должна быть раньше конечной: {part}")
        ranges.append((start, end))

    return sorted(ranges)  # Сортировка для согласованности


def is_date_in_ranges(target_date, ranges) -> bool:
    """Проверяет, попадает ли дата в один из указанных диапазонов."""
    if not ranges:
        return True  # Без фильтрации
    for start, end in ranges:
        if start <= target_date <= end:
            return True
    return False


def parse_bbox(bbox_str: str):
    """Парсит bbox строку 'lat_min,lat_max,lon_min,lon_max'."""
    if not bbox_str:
        return None
    parts = [float(x.strip()) for x in bbox_str.split(',')]
    if len(parts) != 4:
        raise ValueError("bbox должен иметь 4 значения: lat_min,lat_max,lon_min,lon_max")
    lat_min, lat_max, lon_min, lon_max = parts
    if lat_min >= lat_max or lon_min >= lon_max:
        raise ValueError("Неверный bbox: минимум должен быть меньше максимума")
    return lat_min, lat_max, lon_min, lon_max


def is_within_bbox(lat: float, lon: float, bbox):
    """Проверяет, попадают ли координаты в bbox."""
    lat_min, lat_max, lon_min, lon_max = bbox
    return lat_min <= lat < lat_max and lon_min <= lon < lon_max


def load_training_data():
    """Загружает training_flights JSON с вычисленными индексами."""
    input_file = MERGED_DIR / "training_flights.json"

    if not input_file.exists():
        raise FileNotFoundError(f"Не найден {input_file}. Сначала запусти extract_training_data.py")

    print(f"Чтение: {input_file}")

    with open(input_file, 'r', encoding='utf-8') as f:
        flights = json.load(f)

    print(f"Загружено полётов: {len(flights)}")
    return flights


def filter_spots_by_flights(flights, min_flights=200, date_ranges=None, bbox=None):
    """Фильтрует spots с минимум min_flights полётов.

    Args:
        flights: Список полётов
        min_flights: Минимум полётов на spot
        date_ranges: Список диапазонов дат для фильтрации [(start, end), ...]
        bbox: Ограничивающий прямоугольник (lat_min,lat_max,lon_min,lon_max)
    """
    # Парсинг фильтров
    parsed_bbox = parse_bbox(bbox) if bbox else None

    # Вывод информации о фильтрах
    if date_ranges:
        print(f"Фильтр по датам: {len(date_ranges)} диапазон(ов)")
        for i, (start, end) in enumerate(date_ranges):
            print(f"  {i+1}. {start} - {end}")
    if parsed_bbox:
        print(f"Фильтр по bbox: {parsed_bbox}")

    # Применяем фильтры к полётам
    filtered_flights = flights
    if parsed_bbox:
        before = len(filtered_flights)
        filtered_flights = [f for f in filtered_flights
                            if is_within_bbox(f['lat'], f['lon'], parsed_bbox)]
        print(f"  Отфильтровано по bbox: {before} -> {len(filtered_flights)}")

    if date_ranges:
        before = len(filtered_flights)
        filtered_flights = [f for f in filtered_flights
                            if is_date_in_ranges(
                                datetime.strptime(f['datetime'], '%Y-%m-%d %H:%M:%S').date(),
                                date_ranges
                            )]
        print(f"  Отфильтровано по датам: {before} -> {len(filtered_flights)}")

    # Подсчёт полётов на каждый spot
    flight_count = defaultdict(int)
    for f in filtered_flights:
        spot_id = f.get('spot_id')
        if spot_id is not None:
            flight_count[spot_id] += 1

    valid_spots = {spot_id for spot_id, count in flight_count.items()
                   if count >= min_flights}

    print(f"Spots с >= {min_flights} полётами: {len(valid_spots)}")

    # Статистика по количеству полётов
    counts = sorted(flight_count.values(), reverse=True)
    print(f"  Максимум полётов: {counts[0] if counts else 0}")
    print(f"  Всего уникальных spots: {len(flight_count)}")

    return valid_spots


def create_spots_pkl(flights, valid_spots):
    """
    Создаёт spots.pkl в формате: [('name', lat, lon), ...]

    Возвращает также маппинг xcontest spot_id -> новый индекс.
    """
    spots = {}  # xcontest spot_id -> (name, lat, lon)

    for f in flights:
        spot_id = f.get('spot_id')
        if spot_id in valid_spots and spot_id not in spots:
            spots[spot_id] = (
                f.get('spot_name') or f"spot_{spot_id}",
                f['lat'],
                f['lon']
            )

    # Создаём непрерывную индексацию
    spots_list = list(spots.values())
    xcontest_to_new_id = {old_id: new_id for new_id, old_id in enumerate(spots.keys())}

    with open(PKL_DIR / "spots.pkl", 'wb') as f:
        pickle.dump(spots_list, f)

    print(f"Сохранено: spots.pkl ({len(spots_list)} spots)")

    return spots_list, xcontest_to_new_id


def create_spots_by_cell_pkl(flights, valid_spots, xcontest_to_new_id, nb_cells):
    """Создаёт spots_by_cell.pkl: [[new_spot_id, ...], ...]"""
    spots_by_cell = [[] for _ in range(nb_cells)]

    # Собираем уникальные пары (new_spot_id, cell_index)
    spot_cells = set()
    for f in flights:
        spot_id = f.get('spot_id')
        cell_idx = f.get('cell_index')
        if spot_id in valid_spots and cell_idx is not None:
            new_spot_id = xcontest_to_new_id[spot_id]
            spot_cells.add((new_spot_id, cell_idx))

    # Заполняем структуру
    for new_spot_id, cell_idx in spot_cells:
        if 0 <= cell_idx < nb_cells:
            spots_by_cell[cell_idx].append(new_spot_id)

    with open(PKL_DIR / "spots_by_cell.pkl", 'wb') as f:
        pickle.dump(spots_by_cell, f)

    cells_with_spots = sum(1 for cell_spots in spots_by_cell if cell_spots)
    print(f"Сохранено: spots_by_cell.pkl ({cells_with_spots}/{nb_cells} ячеек со spots)")


def create_flights_by_spot_pkl(flights, valid_spots, xcontest_to_new_id):
    """Создаёт flights_by_spot.pkl: [(datetime, (score, None, takeoff_alt, lat, lon)), ...]"""
    nb_spots = len(xcontest_to_new_id)
    flights_by_spot = [[] for _ in range(nb_spots)]

    for f in flights:
        spot_id = f.get('spot_id')
        if spot_id in valid_spots:
            new_spot_id = xcontest_to_new_id[spot_id]

            # Формат: (datetime, (score, None, takeoff_alt, lat, lon))
            record = (
                f['datetime'],
                (
                    f['score'] or 0.0,
                    None,  # alt - для SPOTS всегда None
                    f['takeoff_alt'] or 0.0,
                    f['lat'],
                    f['lon']
                )
            )
            flights_by_spot[new_spot_id].append(record)

    with open(PKL_DIR / "flights_by_spot.pkl", 'wb') as f:
        pickle.dump(flights_by_spot, f)

    total_flights = sum(len(flights) for flights in flights_by_spot)
    print(f"Сохранено: flights_by_spot.pkl ({total_flights} полётов)")


def create_flights_by_cell_day_spot_pkl(flights, valid_spots, xcontest_to_new_id, nb_cells, nb_days):
    """Создаёт flights_by_cell_day_spot.pkl: [[[{spot_id: [...]}, ...], ...], ...]"""
    structure = [[{} for _ in range(nb_days)] for _ in range(nb_cells)]

    for f in flights:
        spot_id = f.get('spot_id')
        cell_idx = f.get('cell_index')
        day_idx = f.get('day_index')

        if spot_id in valid_spots and cell_idx is not None and day_idx is not None:
            if 0 <= cell_idx < nb_cells and 0 <= day_idx < nb_days:
                new_spot_id = xcontest_to_new_id[spot_id]

                # Формат: (datetime, (score, None, takeoff_alt, lat, lon))
                record = (
                    f['datetime'],
                    (
                        f['score'] or 0.0,
                        None,  # alt
                        f['takeoff_alt'] or 0.0,
                        f['lat'],
                        f['lon']
                    )
                )

                if new_spot_id not in structure[cell_idx][day_idx]:
                    structure[cell_idx][day_idx][new_spot_id] = []
                structure[cell_idx][day_idx][new_spot_id].append(record)

    with open(PKL_DIR / "flights_by_cell_day_spot.pkl", 'wb') as f:
        pickle.dump(structure, f)

    # Статистика
    total_entries = 0
    cells_with_data = 0
    for c in range(nb_cells):
        for d in range(nb_days):
            if structure[c][d]:
                total_entries += len(structure[c][d])
                cells_with_data += 1

    print(f"Сохранено: flights_by_cell_day_spot.pkl")
    print(f"  Ячеек×дней с данными: {cells_with_data}")
    print(f"  Всего записей spot: {total_entries}")


def main():
    parser = argparse.ArgumentParser(description='Генерирует PKL файлы из xcontest данных')
    # Фильтрация по датам (диапазоны)
    parser.add_argument('--dates', type=str,
                       default=os.environ.get("TRAINING_DATES"),
                       help='Диапазоны дат: YYYY-MM-DD:YYYY-MM-DD,YYYY-MM-DD:YYYY-MM-DD (или TRAINING_DATES env)')
    # Фильтрация по bbox
    parser.add_argument('--bbox', type=str,
                       default=os.environ.get("TRAINING_BBOX"),
                       help='Bounding box: lat_min,lat_max,lon_min,lon_max (или TRAINING_BBOX env)')
    # Минимум полётов на spot
    parser.add_argument('--min-flights', type=int,
                       default=int(os.environ.get("MIN_FLIGHTS_PER_SPOT", str(THRESHOLD_FLIGHTS))),
                       help=f'Минимум полётов на spot (по умолчанию {THRESHOLD_FLIGHTS})')
    args = parser.parse_args()

    # Парсим диапазоны дат
    parsed_dates = None
    if args.dates:
        parsed_dates = parse_date_ranges(args.dates)

    print("=== Генерация PKL файлов из xcontest данных ===\n")

    # Загружаем данные с индексами
    flights = load_training_data()

    # Фильтруем spots по количеству полётов и другим критериям
    valid_spots = filter_spots_by_flights(
        flights,
        min_flights=args.min_flights,
        date_ranges=parsed_dates,
        bbox=args.bbox
    )

    # Загружаем размерности из существующих PKL
    with open(PKL_DIR / "sorted_cells_latlon.pkl", 'rb') as f:
        sorted_cells = pickle.load(f, encoding='latin1')
    nb_cells = len(sorted_cells)

    with open(PKL_DIR / "meteo_days.pkl", 'rb') as f:
        meteo_days = pickle.load(f, encoding='latin1')
    nb_days = len(meteo_days)

    print(f"\nРазмерности:")
    print(f"  nb_cells: {nb_cells}")
    print(f"  nb_days: {nb_days}")

    # Создаём PKL файлы
    print("\nСоздание PKL файлов:")

    _, xcontest_to_new_id = create_spots_pkl(flights, valid_spots)
    create_spots_by_cell_pkl(flights, valid_spots, xcontest_to_new_id, nb_cells)
    create_flights_by_spot_pkl(flights, valid_spots, xcontest_to_new_id)
    create_flights_by_cell_day_spot_pkl(
        flights, valid_spots, xcontest_to_new_id, nb_cells, nb_days
    )

    print("\nГотово! Созданы файлы:")
    print("  - spots.pkl")
    print("  - spots_by_cell.pkl")
    print("  - flights_by_spot.pkl")
    print("  - flights_by_cell_day_spot.pkl")


if __name__ == "__main__":
    main()
