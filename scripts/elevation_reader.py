#!/usr/bin/env python3
"""
Чтение elevation tiles для получения высоты точки взлёта.
Аналог PHP функции getElevation() из www/apps/api/get.php
"""

import sys
from pathlib import Path

# Импорт TilesMaths из neural_network
sys.path.insert(0, str(Path(__file__).parent.parent / "neural_network" / "inc"))
from tiles_maths import TilesMaths


class ElevationReader:
    """Читает высоту из elevation tiles по координатам."""

    def __init__(self, elevation_dir: str = None, zoom: int = 7):
        """
        Args:
            elevation_dir: Путь к директории с elevation tiles
            zoom: Уровень зума (по умолчанию 7, доступные: 5, 6, 7)
        """
        if elevation_dir is None:
            project_root = Path(__file__).parent.parent
            elevation_dir = project_root / "tiler" / "_cache" / "elevation"
        self.elevation_dir = Path(elevation_dir)
        self.zoom = zoom

    def get_elevation(self, lat: float, lon: float) -> int | None:
        """
        Возвращает высоту в метрах по координатам.

        Args:
            lat: Широта
            lon: Долгота

        Returns:
            Высота в метрах или None если файл не найден
        """
        # Конвертировать lat/lon в tile coords
        coords = TilesMaths.LatLonToTileCoords(self.zoom, lat, lon)

        # Путь к файлу elevation
        filepath = self.elevation_dir / str(self.zoom) / str(coords['tx']) / f"{coords['ty']}.elev"

        if not filepath.exists():
            return None

        try:
            with open(filepath, 'rb') as f:
                # Прочитать весь файл
                content = f.read()

                # Вычислить смещение (256x256 пикселей, 2 байта на пиксель)
                offset = 2 * (coords['x'] * 256 + coords['y'])

                if offset + 2 > len(content):
                    return None

                # Прочитать 2 байта (big-endian)
                str_val = content[offset:offset+2]
                elevation = (str_val[0] << 8) + str_val[1]

                return elevation
        except (IOError, OSError):
            return None

    def get_mountainess(self, lat: float, lon: float, grid_size: int = 5, radius_km: float = 5.0) -> float:
        """
        Вычисляет гористость местности по высоте вокруг точки.

        Использует сетку grid_size x grid_size точек в круге radius_km.
        Формула: (max_elev - min_elev) / 800, ограничено [0, 1].

        Args:
            lat: Широта центра
            lon: Долгота центра
            grid_size: Размер сетки (5 = 5x5 = 25 точек)
            radius_km: Радиус области анализа в км

        Returns:
            Значение 0.0-1.0 (0 = равнина, 1 = горы)
        """
        import math

        # Коэффициенты перевода градусов в километры
        # 1° широты ≈ 111 км везде
        # 1° долготы ≈ 111 км * cos(широта)
        km_per_deg_lat = 111.0
        km_per_deg_lon = 111.0 * math.cos(math.radians(lat))

        # Шаг сетки в градусах
        step_lat = (radius_km * 2) / (grid_size - 1) / km_per_deg_lat
        step_lon = (radius_km * 2) / (grid_size - 1) / km_per_deg_lon

        # Начальная позиция (левый верхний угол сетки)
        start_lat = lat - radius_km / km_per_deg_lat
        start_lon = lon - radius_km / km_per_deg_lon

        elevations = []

        # Семплируем высоту в точках сетки
        for i in range(grid_size):
            for j in range(grid_size):
                sample_lat = start_lat + i * step_lat
                sample_lon = start_lon + j * step_lon
                elev = self.get_elevation(sample_lat, sample_lon)
                if elev is not None:
                    elevations.append(elev)

        # Если не удалось прочитать ни одной высоты
        if not elevations:
            return 0.0

        # Вычисляем гористость: (max - min) / 800
        elev_range = max(elevations) - min(elevations)
        mountainess = elev_range / 800.0

        # Ограничиваем [0, 1]
        return max(0.0, min(1.0, mountainess))


# Тестирование
if __name__ == "__main__":
    reader = ElevationReader()

    # Тестовые координаты (Ljubljana, Словения)
    test_coords = [
        (46.272567, 13.473567),  # Stol takeoff
        (46.05, 14.50),          # Ljubljana
        (45.43, 12.32),          # Venice
    ]

    print("Тестирование чтения высоты:")
    for lat, lon in test_coords:
        elev = reader.get_elevation(lat, lon)
        print(f"  {lat}, {lon}: {elev} m")
