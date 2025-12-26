# Процесс обучения Paraglidable Neural Network

## Обзор

Документ описывает процесс обучения нейронной сети для прогнозирования условий полёта на планёре.

## Архитектура обучения

### Две модели:

1. **CELLS Model** - Прогноз пригодности для каждой 1°x1° ячейки
   - Использует **55 ячеек** (первые 55 из `sorted_cells_latlon.pkl`)
   - Предсказывает: flyability, crossability, wind-flyability, humidity-flyability

2. **SPOTS Model** - Прогноз для конкретных точек взлёта
   - Использует **80 ячеек** (первые 80)
   - Предсказывает flyability для каждого spot

### Входные данные (метеорология):

**Источник:** GFS (Global Forecast System) от NOAA
- **Разрешение:** 0.25° × 0.25°
- **Уровни:** 1000, 900, 800, 700, 600, 500, 400, 300, 200 hPa
- **Параметры:** 195 значений (65 параметров × 3 временных шага: 06:00, 12:00, 18:00)

**Основные параметры:**
- Temperature (разные уровни)
- Humidity (относительная и относительная)
- Wind (U, V компоненты)
- Geopotential height
- Vorticity
- CAPE, CIN
- и др.

### Целевые данные (полёты):

**Источник:** XC-Leonardo и другие трекинговые системы
**Формат:** IGC файлы (стандарт FAI)

**Структура IGC файла:**
```
HFDTE260425       - Дата (YYMMDD)
HFPLTPilot:Name   - Пилот
HFGTYGlider:Type  - Тип планёра
HFFXA100          - Версия формата
...
B121535453023N00002539WA0045003098  - Трек (время, lat, lon, alt, ...)
...
```

**Ключевые извлекаемые данные:**
1. Дата полёта (из `HFDTE`)
2. Координаты взлёта (первые B-строки)
3. Трек полёта (последовательность B-строк)
4. Балл/очки (постобработка XC-Leonardo)

### Агрегация данных:

**Ячейки:** 1° × 1° (lat × lon)
```python
cell_lat = int(takeoff_lat)  # 例如: 45 для lat=45.7°N
cell_lon = int(takeoff_lon)  # например: 12 для lon=12.3°E
```

**Временная агрегация:** По дням
```python
flights_by_cell_day[cell_index, day_index] = [
    flight_1, flight_2, ..., flight_N
]
```

---

## Обучающие данные

### Файлы в `neural_network/bin/data/`:

| Файл | Размер | Описание |
|------|--------|----------|
| `sorted_cells_latlon.pkl` | 97 | Координаты ячеек (lat, lon) |
| `sorted_cells.pkl` | 97 | Индексы в GRIB сетке |
| `flights_by_cell_day.pkl` | ~ | Полёты по ячейкам/дням |
| `flights_by_cell_day_spot.pkl` | ~ | Полёты по ячейкам/дням/spot |
| `meteo_days.pkl` | ~ | Дни с погодными данными |
| `meteo_params.pkl` | 195 | Описание параметров |
| `meteo_content_by_cell_day.pkl` | ~ | Матрица погодных данных |
| `mountainess_by_cell_alt.pkl` | 97 | Высота рельефа по ячейкам |
| `spots.pkl` | ~ | Точки взлёта |
| `spots_merged.pkl` | ~ | Объединённые spots |
| `flights_by_spot.pkl` | ~ | Полёты по spots |
| `spots_by_cell.pkl` | ~ | Spots по ячейкам |

### Текущие обучающие ячейки (97 штук):

**Диапазон:**
- Latitude: 43°N - 49°N
- Longitude: 4°E - 18°E

**Примеры координат:**
```
Cell 0:  lat=46.0°N, lon=12.0°E  (центральные Альпы)
Cell 54: lat=45.0°N, lon=9.0°E   (северная Италия)
```

---

## Процесс обучения

### Этап 1: Подготовка данных

1. **Сбор полётов:**
   - Скачать IGC файлы с XC-Leonardo для каждой ячейки
   - Период: обычно несколько лет (2018-2024)
   - Скрипт: (нужен для скачивания с XC-Leonardo API)

2. **Сбор погодных данных:**
   - Скачать исторические GFS GRIB файлы
   - Период: совпадает с полётами
   - Разрешение: 0.25°

3. **Агрегация:**
   - Сопоставить полёты с погодными условиями
   - Создать `flights_by_cell_day.pkl`
   - Создать `meteo_content_by_cell_day.pkl`

### Этап 2: Обучение Population Model

**Файл:** `neural_network/train.py`

```python
# Использует 55 ячеек
self.all_cells = list(range(55))

# Процесс:
# 1. Для каждой ячейки:
#    - Загрузить погодные данные для дней с полётами
#    - Загрузить данные о полётах
#    - Обучить модель предсказывать вероятность полёта
#
# 2. Создаётся population_alt_cell_XX.npy для каждой ячейки
#    - Содержит 5 значений (по одному на высоту)
#    - Хранится в bin/models/CLASSIFICATION_1.0.0/weights/
```

**Population weights** (`population_alt_cell_XX.npy`):
- Размер: (5,) - одно значение на высоту
- Смысл: "базовая вероятность полёта" для данной ячейки
- Обучается на данных о том, летали ли в этот день

### Этап 3: Обучение Spots Model

```python
# Использует 80 ячеек
self.all_cells = list(range(80))

# Процесс:
# 1. Для каждого spot в ячейке:
#    - Обучить модель предсказывать flyability
#    - Учесть специфику spot (экспозиция, рельеф и т.д.)
```

---

## Формат метрических данных

### meteo_content_by_cell_day.pkl

**Размерность:** `[nb_days, nb_cells, nb_parameters]`

- `nb_days`: ~2000+ дней
- `nb_cells`: 97 ячеек
- `nb_parameters`: 195 погодных параметров

**Структура параметров (195 = 65 × 3):**
```
[
  # 06:00 UTC (65 параметров)
  temp_1000, temp_900, ..., temp_200,
  rh_1000, ..., rh_200,
  u_1000, ..., u_200,
  v_1000, ..., v_200,
  gh_1000, ..., gh_200,
  ...

  # 12:00 UTC (65 параметров)
  ...

  # 18:00 UTC (65 параметров)
  ...
]
```

### flights_by_cell_day.pkl

**Структура:**
```python
{
  (cell_index, day_index): [
    Flight(flight_id, takeoff_lat, takeoff_lon, distance, score, ...),
    Flight(...),
    ...
  ]
}
```

---

## Модель

### Архитектура:

**Входы:**
- Метеорологические данные (195 параметров)
- День недели (one-hot: 7 значений)
- Дата (нормализованная)
- Сезонные коэффициенты

**Выходы:**
1. **Flyability** - вероятность пригодности (0-1)
2. **Crossability** - вероятность перехода (0-1)
3. **Wind-flyability** - пригодность по ветру (0-1)
4. **Humidity-flyability** - пригодность по влажности (0-1)

Для каждой высоты: 1000, 900, 800, 700, 600 hPa

### Per-cell веса:

**`population_alt_cell_XX.npy`** - (5,) массив
- Индекс 0: вес для 1000 hPa
- Индекс 1: вес для 900 hPa
- Индекс 2: вес для 800 hPa
- Индекс 3: вес для 700 hPa
- Индекс 4: вес для 600 hPa

**Смысл:** Эти веса моделируют "плотность пилотов" на каждой высоте в данной ячейке. Если в ячейке много полётов на 1000 hPa, вес будет выше.

---

## Расширение области (до 37°E)

### Новые ячейки для добавления:

**Текущая область:** 4°E - 18°E (97 ячеек)
**Целевая область:** 4°E - 37°E

**Новые ячейки (примерно 40-80 штук):**
```
Lat: 43°N - 49°N
Lon: 19°E - 37°E
```

### Что нужно для расширения:

1. **Данные о полётах:**
   - Скачать IGC с XC-Leonardo для новой области
   - Период: минимум 2-3 года данных
   - Районы: Карпаты, Балканы, Крым, Кавказ

2. **Погодные данные:**
   - Исторические GFS для новых ячеек
   - GRIB файлы за тот же период

3. **Elevation данные:**
   - Скачать SRTM tiles для новой области

4. **Spots данные:**
   - Extract takeoff spots из IGC файлов
   - Ручная валидация для популярных точек

### Скрипты для создания:

**`scripts/download_flights_xc_leonardo.py`:**
```python
import requests
from datetime import datetime

# XC-Leonardo API (пример, нужен точный endpoint)
BASE_URL = "https://www.xc-league.org/"

def download_flights_for_cell(lat, lon, start_date, end_date):
    """Скачать IGC файлы для ячейки 1x1 градус"""
    params = {
        'minlat': lat,
        'maxlat': lat + 1,
        'minlon': lon,
        'maxlon': lon + 1,
        'datefrom': start_date,
        'dateto': end_date
    }
    # Реализовать запрос к API
    pass

# Пример использования:
for lat in range(43, 50):
    for lon in range(19, 38):
        download_flights_for_cell(lat, lon, '2018-01-01', '2024-12-31')
```

**`scripts/generate_extended_cells.py`:**
```python
import pickle

# Загрузить существующие
with open('neural_network/bin/data/sorted_cells_latlon.pkl', 'rb') as f:
    existing = pickle.load(f)

# Добавить новые
new_cells = []
for lat in range(43, 50):
    for lon in range(19, 38):
        if (lat, lon) not in existing:
            new_cells.append((lat, lon))

all_cells = existing + new_cells

# Сохранить
with open('neural_network/bin/data/sorted_cells_latlon.pkl', 'wb') as f:
    pickle.dump(all_cells, f)
```

---

## Время обучения

**Текущее:**
- Population model (55 ячеек): ~2-4 часа
- Spots model (80 ячеек): ~4-8 часов

**С расширением (140+ ячеек):**
- Population model (120 ячеек): ~6-12 часов
- Spots model (180 ячеек): ~10-20 часов

---

## Проверка качества

После обучения:

1. **Validation loss** - должна уменьшаться
2. **Test forecast** - запустить `python forecast.py`
3. **Visual check** - открыть сайт и проверить карту
4. **Spot prediction** - сравнить с реальными полётами

---

## Полезные команды

```bash
# Запуск обучения
cd neural_network/
python train.py

# Генерация прогноза
python forecast.py

# Проверка ячеек
python3 -c "import pickle; cells=pickle.load(open('neural_network/bin/data/sorted_cells_latlon.pkl','rb')); print(len(cells), cells[:5])"

# Проверка population weights
ls neural_network/bin/models/CLASSIFICATION_1.0.0/weights/population_alt_cell_*.npy | wc -l
```
