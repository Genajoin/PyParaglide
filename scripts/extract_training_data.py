#!/usr/bin/env python3
"""
Объединяет JSON файлы с полётами и извлекает данные для обучения.

Создаёт файлы со структурой, совместимой с Paraglidable neural network:
- training_flights.json - полёты с вычисленными индексами
- spots.json - точки взлёта
- stats.json - статистика

Формат согласно TRAINING_PROCESS.md:
- flights_by_cell_day_spot: (datetime, (score, None, takeoff_alt, lat, lon))
- flights_by_cell_day: (datetime, (score, alt, plaf, lat, lon, takeoff_alt, mountainess))
"""

import argparse
import json
import os
import re
import math
import pickle
from pathlib import Path
from collections import defaultdict
from datetime import datetime, date

# Импорт ElevationReader
from elevation_reader import ElevationReader

# Пути
PROJECT_ROOT = Path("/home/gena/dev/Paraglidable")
FLIGHTS_DIR = PROJECT_ROOT / "data" / "flights"
MERGED_DIR = FLIGHTS_DIR / "merged"
PKL_DIR = PROJECT_ROOT / "neural_network" / "bin" / "data"


class PklIndexer:
    """Вычисляет cell_index и day_index из существующих PKL файлов."""

    def __init__(self, date_ranges: list[tuple[date, date]] | None = None):
        with open(PKL_DIR / "sorted_cells_latlon.pkl", 'rb') as f:
            self.sorted_cells_latlon = pickle.load(f, encoding='latin1')
        with open(PKL_DIR / "meteo_days.pkl", 'rb') as f:
            all_meteo_days = pickle.load(f, encoding='latin1')

        # Применяем фильтрацию по диапазонам дат
        if date_ranges:
            self.meteo_days = filter_meteo_days(all_meteo_days, date_ranges)
            print(f"  Отфильтровано дней: {len(all_meteo_days)} -> {len(self.meteo_days)}")
        else:
            self.meteo_days = all_meteo_days

        self.day_to_idx = {day: idx for idx, day in enumerate(self.meteo_days)}
        self.nb_cells = len(self.sorted_cells_latlon)
        self.nb_days = len(self.meteo_days)

    def get_cell_index(self, lat: float, lon: float) -> int | None:
        """Вычисляет индекс ячейки по координатам."""
        cell_lat = int(math.floor(lat))
        cell_lon = int(math.floor(lon))
        try:
            return self.sorted_cells_latlon.index((float(cell_lat), float(cell_lon)))
        except ValueError:
            return None

    def get_day_index(self, dt_str: str) -> int | None:
        """Вычисляет индекс дня из строки datetime."""
        flight_date = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S').date()
        return self.day_to_idx.get(flight_date)


def load_json_file(filepath):
    """Загружает JSON файл."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"  Ошибка JSON в {filepath.name}: {e}")
        return None
    except Exception as e:
        print(f"  Ошибка чтения {filepath.name}: {e}")
        return None


def extract_coords_from_link(link):
    """Извлекает координаты из takeoff.link."""
    try:
        match = re.search(r'filter\[point\]=([0-9.-]+)\s+([0-9.-]+)', link)
        if match:
            lon = float(match.group(1))
            lat = float(match.group(2))
            return lon, lat
    except (ValueError, AttributeError):
        pass
    return None, None


def parse_duration(duration_str):
    """Парсит длительность из формата ISO 8601 (PT04H01M35S)."""
    try:
        match = re.match(r'PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?', duration_str)
        if match:
            hours = int(match.group(1) or 0)
            minutes = int(match.group(2) or 0)
            seconds = int(match.group(3) or 0)
            return hours * 3600 + minutes * 60 + seconds
    except (ValueError, AttributeError):
        pass
    return None


def get_cell_from_coords(lat, lon):
    """Вычисляет ячейку 1x1 градус по координатам."""
    try:
        cell_lat = int(lat)
        cell_lon = int(lon)
        return cell_lat, cell_lon
    except (ValueError, TypeError):
        return None, None


def get_flight_date_range(flights):
    """Вычисляет минимальную и максимальную дату из списка полётов.

    Returns:
        (min_date, max_date, count) или (None, None, 0)
    """
    if not flights:
        return None, None, 0

    dates = []
    for flight in flights:
        time_str = flight.get('pointStart', {}).get('time', '')
        dt_formatted = parse_datetime(time_str)
        if dt_formatted:
            flight_date = datetime.strptime(dt_formatted, '%Y-%m-%d %H:%M:%S').date()
            dates.append(flight_date)

    if not dates:
        return None, None, 0

    return min(dates), max(dates), len(dates)


def parse_datetime(dt_str):
    """Парсит datetime строку в формат yyyy-mm-dd hh:mm:ss."""
    try:
        dt = datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        return dt.strftime('%Y-%m-%d %H:%M:%S')
    except (ValueError, AttributeError):
        return None


def parse_date_ranges(dates_str: str) -> list[tuple[date, date]] | None:
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


def is_date_in_ranges(target_date: date, ranges: list[tuple[date, date]]) -> bool:
    """Проверяет, попадает ли дата в один из указанных диапазонов."""
    if not ranges:
        return True  # Без фильтрации
    for start, end in ranges:
        if start <= target_date <= end:
            return True
    return False


def filter_meteo_days(meteo_days: list[date], ranges: list[tuple[date, date]]) -> list[date]:
    """Фильтрует список meteo_days, оставляя только даты в указанных диапазонах."""
    if not ranges:
        return meteo_days
    return [d for d in meteo_days if is_date_in_ranges(d, ranges)]


def parse_bbox(bbox_str: str) -> tuple[float, float, float, float] | None:
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


def is_within_bbox(lat: float, lon: float, bbox: tuple) -> bool:
    """Проверяет, попадают ли координаты в bbox."""
    lat_min, lat_max, lon_min, lon_max = bbox
    return lat_min <= lat < lat_max and lon_min <= lon < lon_max


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Вычисляет расстояние между двумя точками в километрах по формуле Haversine."""
    R = 6371.0  # Радиус Земли в км
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    lon1_rad = math.radians(lon1)
    lon2_rad = math.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = (math.sin(dlat / 2) ** 2 +
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def cluster_nearby_spots(spots: dict, distance_km: float = 2.0):
    """Кластеризует nearby spots и возвращает объединённый словарь и маппинг.

    Args:
        spots: Словарь спотов {spot_id: {id, name, lat, lon, flight_count, ...}}
        distance_km: Максимальное расстояние для кластеризации

    Returns:
        (clustered_spots, spot_id_mapping) - словарь спотов после кластеризации и маппинг old_id -> new_id
    """
    if not spots:
        return spots, {}

    # Конвертируем в список для обработки
    spot_list = list(spots.values())
    if len(spot_list) <= 1:
        return spots, {}

    # Иерархическая кластеризация
    clusters = []
    assigned = set()

    for i, spot in enumerate(spot_list):
        if spot['id'] in assigned:
            continue

        # Создаём новый кластер
        cluster = [spot]
        assigned.add(spot['id'])

        # Находим все соседние споты в радиусе distance_km
        for j, other in enumerate(spot_list):
            if i == j or other['id'] in assigned:
                continue

            dist = haversine_distance(
                spot['lat'], spot['lon'],
                other['lat'], other['lon']
            )

            if dist <= distance_km:
                cluster.append(other)
                assigned.add(other['id'])

        clusters.append(cluster)

    # Для каждого кластера выбираем representative (с max flight_count)
    clustered_spots = {}
    spot_id_mapping = {}  # old_id -> new_id

    new_id = 0
    for cluster in clusters:
        if not cluster:
            continue

        # Сортируем по flight_count (убывание)
        cluster_sorted = sorted(cluster, key=lambda s: s.get('flight_count', 0), reverse=True)
        representative = cluster_sorted[0]

        # Вычисляем центроид (среднее взвешенное по flight_count)
        total_flights = sum(s.get('flight_count', 0) for s in cluster_sorted)
        if total_flights > 0:
            weighted_lat = sum(s['lat'] * s.get('flight_count', 0) for s in cluster_sorted) / total_flights
            weighted_lon = sum(s['lon'] * s.get('flight_count', 0) for s in cluster_sorted) / total_flights
        else:
            weighted_lat = representative['lat']
            weighted_lon = representative['lon']

        # Создаём новый spot
        new_spot_id = str(new_id)
        new_spot = {
            'id': new_spot_id,
            'name': representative['name'],
            'lat': weighted_lat,
            'lon': weighted_lon,
            'countryIso': representative.get('countryIso', ''),
            'cell_lat': int(math.floor(weighted_lat)),
            'cell_lon': int(math.floor(weighted_lon)),
            'flight_count': total_flights,
            'cluster_size': len(cluster_sorted),
            'original_spots': [s['id'] for s in cluster_sorted]
        }

        clustered_spots[new_spot_id] = new_spot

        # Запоминаем маппинг old_id -> new_id
        for old_spot in cluster_sorted:
            spot_id_mapping[old_spot['id']] = new_spot_id

        new_id += 1

    # Статистика
    print(f"Кластеризация спотов:")
    print(f"  Было: {len(spots)} спотов")
    print(f"  Стало: {len(clustered_spots)} кластеров")
    print(f"  Расстояние: {distance_km} км")

    return clustered_spots, spot_id_mapping


def merge_flights_from_source():
    """Объединяет все JSON файлы из data/flights/ в один без дубликатов."""
    MERGED_DIR.mkdir(exist_ok=True)

    # Поиск всех JSON файлов
    json_files = list(FLIGHTS_DIR.glob("*.json"))

    if not json_files:
        print(f"В директории {FLIGHTS_DIR} не найдено JSON файлов.")
        return None

    print(f"Найдено {len(json_files)} JSON файлов:")
    print("=" * 60)

    all_flights = []

    for filepath in sorted(json_files):
        size_mb = filepath.stat().st_size / 1024 / 1024
        print(f"  {filepath.name}: {size_mb:.1f} MB", end='')

        data = load_json_file(filepath)
        if data is None:
            print(" [ошибка]")
            continue

        if not isinstance(data, list):
            print(f" [неверный формат: {type(data)}]")
            continue

        print(f" [{len(data)} записей]")
        all_flights.extend(data)

    print("=" * 60)
    print(f"Всего загружено: {len(all_flights)} записей")

    # Удаление дубликатов по id
    unique_flights = {}
    duplicates = 0

    for flight in all_flights:
        flight_id = flight.get('id')
        if flight_id is None:
            continue
        if flight_id in unique_flights:
            duplicates += 1
        else:
            unique_flights[flight_id] = flight

    merged_flights = list(unique_flights.values())

    print(f"Уникальных записей: {len(merged_flights)}")
    print(f"Удалено дубликатов: {duplicates}")

    return merged_flights


def extract_training_data(flights, indexer, elev_reader, date_ranges=None, bbox=None,
                          cluster_distance_km=None):
    """Извлекает данные для обучения из списка полётов.

    Args:
        flights: Список полётов из xcontest API
        indexer: PklIndexer для получения индексов ячеек и дней
        elev_reader: ElevationReader для получения высоты
        date_ranges: Список диапазонов дат для фильтрации [(start, end), ...]
        bbox: Ограничивающий прямоугольник (lat_min,lat_max,lon_min,lon_max или None)
        cluster_distance_km: Радиус кластеризации спотов в км (None = отключено)
    """

    # Структуры данных
    training_flights = []
    spots = {}

    # Статистика
    stats = {
        'total': len(flights),
        'with_coords': 0,
        'with_valid_time': 0,
        'by_country': defaultdict(int),
        'by_cell': defaultdict(int),
        'spots_created': 0,
        'errors': defaultdict(int),
        'missing_fields': defaultdict(int),
    }

    # Парсинг фильтров
    parsed_bbox = parse_bbox(bbox) if bbox else None

    # Вывод информации о фильтрах
    if date_ranges:
        print(f"Фильтр по датам: {len(date_ranges)} диапазон(ов)")
        for i, (start, end) in enumerate(date_ranges):
            print(f"  {i+1}. {start} - {end}")
    if parsed_bbox:
        print(f"Фильтр по bbox: {parsed_bbox}")

    for flight in flights:
        flight_id = flight.get('id')
        takeoff = flight.get('takeoff', {})
        point_start = flight.get('pointStart', {})
        flight_stats = flight.get('stats', {})
        league = flight.get('league', {})
        glider = flight.get('glider', {})

        # Извлекаем координаты
        link = takeoff.get('link', '')
        lon, lat = extract_coords_from_link(link)

        if lon is None or lat is None:
            stats['errors']['no_coords'] += 1
            continue

        stats['with_coords'] += 1

        # datetime
        time_str = point_start.get('time', '')
        dt_formatted = parse_datetime(time_str)
        if not dt_formatted:
            stats['errors']['invalid_time'] += 1
            continue
        stats['with_valid_time'] += 1

        # Фильтр по bbox
        if parsed_bbox and not is_within_bbox(lat, lon, parsed_bbox):
            stats['errors']['outside_bbox'] += 1
            continue

        # Фильтр по датам
        if date_ranges:
            flight_date = datetime.strptime(dt_formatted, '%Y-%m-%d %H:%M:%S').date()
            if not is_date_in_ranges(flight_date, date_ranges):
                stats['errors']['outside_date_range'] += 1
                continue

        # Вычисляем ячейку
        cell_lat, cell_lon = get_cell_from_coords(lat, lon)
        if cell_lat is None:
            stats['errors']['invalid_cell'] += 1
            continue

        cell_key = f"{cell_lat}_{cell_lon}"
        stats['by_cell'][cell_key] += 1

        # Страна
        country = takeoff.get('countryIso', '')
        if country:
            stats['by_country'][country] += 1

        # Длительность
        duration_str = flight_stats.get('duration', '')
        duration_sec = parse_duration(duration_str)

        # Spot данные
        spot_id = takeoff.get('id')
        if spot_id and spot_id not in spots:
            spots[spot_id] = {
                'id': spot_id,
                'name': takeoff.get('name', ''),
                'lat': lat,
                'lon': lon,
                'countryIso': country,
                'cell_lat': cell_lat,
                'cell_lon': cell_lon,
                'flight_count': 0
            }
            stats['spots_created'] += 1

        if spot_id in spots:
            spots[spot_id]['flight_count'] += 1

        # Данные из xcontest API
        route = league.get('route', {})
        score = route.get('points') if route else None
        avg_speed = route.get('avgSpeed') if route else None
        alt = None
        plaf = avg_speed
        takeoff_alt = elev_reader.get_elevation(lat, lon)
        mountainess = elev_reader.get_mountainess(lat, lon)

        # Индексы
        cell_index = indexer.get_cell_index(lat, lon)
        day_index = indexer.get_day_index(dt_formatted)

        # Фильтр
        if cell_index is None:
            stats['errors']['outside_training_cells'] += 1
            continue
        if day_index is None:
            stats['errors']['outside_training_dates'] += 1
            continue

        # Формируем запись
        training_entry = {
            'datetime': dt_formatted,
            'score': score,
            'alt': alt,
            'plaf': plaf,
            'lat': lat,
            'lon': lon,
            'takeoff_alt': takeoff_alt,
            'mountainess': mountainess,
            'cell_lat': cell_lat,
            'cell_lon': cell_lon,
            'cell_index': cell_index,
            'day_index': day_index,
            'spot_id': spot_id,
            'flight_id': flight_id,
            'spot_name': takeoff.get('name', ''),
            'pilot_id': flight.get('pilot', {}).get('id'),
            'pilot_name': flight.get('pilot', {}).get('name', ''),
            'duration_sec': duration_sec,
            'glider': glider.get('name', ''),
            'glider_class': glider.get('classFAI'),
            'countries': flight.get('countries', []),
            'league': league.get('name', ''),
            'route_distance': route.get('distance') if route else None,
            'route_type': route.get('type') if route else None,
        }

        training_flights.append(training_entry)

        # Статистика
        for field in ['score', 'takeoff_alt', 'mountainess']:
            if training_entry[field] is None:
                stats['missing_fields'][field] += 1
        if score:
            stats['with_score'] = stats.get('with_score', 0) + 1
        if avg_speed:
            stats['with_avg_speed'] = stats.get('with_avg_speed', 0) + 1

    # Кластеризация спотов (после обработки всех полётов)
    if cluster_distance_km and spots:
        print(f"\n--- Кластеризация спотов (радиус {cluster_distance_km} км) ---")
        spots, spot_id_mapping = cluster_nearby_spots(spots, cluster_distance_km)

        # Обновляем spot_id во всех обработанных полётах
        for flight_entry in training_flights:
            old_spot_id = flight_entry.get('spot_id')
            if old_spot_id in spot_id_mapping:
                flight_entry['spot_id'] = spot_id_mapping[old_spot_id]

    return {
        'flights': training_flights,
        'spots': spots,
        'stats': dict(stats),
    }


def print_stats(result):
    """Выводит статистику."""
    stats = result['stats']

    print(f"\nСтатистика:")
    print(f"  Всего полётов: {stats['total']}")
    print(f"  С координатами: {stats['with_coords']}")
    print(f"  С корректным временем: {stats['with_valid_time']}")
    print(f"  Создано spots: {stats['spots_created']}")
    print(f"  С score (points): {stats.get('with_score', 0)}")
    print(f"  С avg_speed: {stats.get('with_avg_speed', 0)}")
    print(f"  Вне области ячеек: {stats['errors'].get('outside_training_cells', 0)}")
    print(f"  Вне области дат: {stats['errors'].get('outside_training_dates', 0)}")

    if stats['by_cell']:
        print(f"\nТоп-10 ячеек по полётам:")
        for cell, count in sorted(stats['by_cell'].items(),
                                  key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {cell}: {count}")

    if stats['missing_fields']:
        print(f"\nНедостающие поля:")
        for field, count in sorted(stats['missing_fields'].items()):
            if count > 0:
                print(f"  {field}: {count}")


def save_results(result):
    """Сохраняет результаты в файлы."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Основной файл с полётами
    output_file = MERGED_DIR / "training_flights.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result['flights'], f, ensure_ascii=False, indent=2)
    size_mb = output_file.stat().st_size / 1024 / 1024
    print(f"\nСохранено: {output_file} ({size_mb:.1f} MB)")

    # Spots
    spots_file = MERGED_DIR / "spots.json"
    with open(spots_file, 'w', encoding='utf-8') as f:
        json.dump(result['spots'], f, ensure_ascii=False, indent=2)
    print(f"Сохранено: {spots_file}")

    # Статистика
    stats_file = MERGED_DIR / "stats.json"
    stats_data = {
        'timestamp': timestamp,
        **result['stats'],
    }
    # Конвертируем defaultdict в dict
    for key in ['by_country', 'by_cell', 'errors', 'missing_fields']:
        if key in stats_data:
            stats_data[key] = dict(stats_data[key])
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=2)
    print(f"Сохранено: {stats_file}")


def main():
    parser = argparse.ArgumentParser(description='Извлекает данные для обучения из xcontest JSON')
    parser.add_argument('--dry-run', action='store_true', help='Вывести статистику без сохранения')
    parser.add_argument('--input', type=str, help='JSON файл с полётами (пропускает объединение)')

    # Фильтрация по датам (диапазоны)
    parser.add_argument('--dates', type=str,
                       default=os.environ.get("TRAINING_DATES"),
                       help='Диапазоны дат: YYYY-MM-DD:YYYY-MM-DD,YYYY-MM-DD:YYYY-MM-DD (или TRAINING_DATES env)')

    # Фильтрация по bbox
    parser.add_argument('--bbox', type=str,
                       default=os.environ.get("TRAINING_BBOX"),
                       help='Bounding box: lat_min,lat_max,lon_min,lon_max (или TRAINING_BBOX env)')

    # Кластеризация спотов
    parser.add_argument('--cluster-distance', type=float,
                       default=float(os.environ.get("SPOT_CLUSTER_DISTANCE_KM", "0")),
                       help='Радиус кластеризации спотов в км (0 = отключено, или SPOT_CLUSTER_DISTANCE_KM env)')

    args = parser.parse_args()

    # Парсим диапазоны дат
    parsed_dates = None
    if args.dates:
        parsed_dates = parse_date_ranges(args.dates)

    print("=== Извлечение данных для обучения ===")

    # Шаг 1: Объединение или загрузка
    if args.input:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Файл не найден: {input_path}")
            return
        print(f"Загрузка: {input_path}")
        flights = load_json_file(input_path)
    else:
        print("\n--- Шаг 1: Объединение JSON файлов ---")
        flights = merge_flights_from_source()
        if flights is None:
            return

    # Шаг 2: Извлечение данных
    print("\n--- Шаг 2: Извлечение данных для обучения ---")

    # Вычисляем период полётов (из JSON данных)
    flight_min, flight_max, flight_count = get_flight_date_range(flights)

    # Индексер для метео-данных
    indexer = PklIndexer(date_ranges=parsed_dates)
    elev_reader = ElevationReader()

    # Вывод информации о периодах
    print(f"Данные полётов:")
    print(f"  Всего записей: {flight_count}")
    if flight_min and flight_max:
        print(f"  Период: {flight_min} - {flight_max}")

    print(f"\nМетео-данные (для обучения):")
    print(f"  nb_cells: {indexer.nb_cells}")
    print(f"  nb_days: {indexer.nb_days}")
    if indexer.nb_days > 0:
        print(f"  Период: {indexer.meteo_days[0]} - {indexer.meteo_days[-1]}")

    result = extract_training_data(flights, indexer, elev_reader,
                                   date_ranges=parsed_dates,
                                   bbox=args.bbox,
                                   cluster_distance_km=args.cluster_distance if args.cluster_distance > 0 else None)

    # Шаг 3: Вывод статистики
    print("\n--- Шаг 3: Статистика ---")
    print_stats(result)

    # Шаг 4: Сохранение
    if not args.dry_run:
        print("\n--- Шаг 4: Сохранение ---")
        MERGED_DIR.mkdir(exist_ok=True)
        save_results(result)
        print("\nГотово!")
    else:
        print("\n[DRY RUN] Файлы не сохранены")


if __name__ == "__main__":
    main()
