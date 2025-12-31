# Процесс обучения Paraglidable Neural Network

## Обзор

Документ описывает процесс обучения нейронной сети для прогнозирования условий полёта на планёре.

## Рабочий подход подготовки данных

**Текущий рабочий метод (2025):**
1. **Данные полётов** → xContest JSON файлы в `data/flights/`
2. **Метео-данные** → AWS S3 через `scripts/download_GFS.py` (2021+)

**Рекомендуемая среда:** Docker Compose (автоматически настраивает volumes, shm_size, монтирование)

---

## Docker Quick Start

### 1. Настройка `.env`

```bash
# Область обучения (bbox) - задаёт количество ячеек
# Формат: lat_min,lat_max,lon_min,lon_max
TRAINING_BBOX=45,47,13,15  # 9 ячеек: Slovenia/northern Italy

# Период обучения (летние месяцы 2021-2025)
# Формат: YYYY-MM-DD:YYYY-MM-DD,YYYY-MM-DD:YYYY-MM-DD,...
TRAINING_DATES=2021-06-01:2021-08-31,2022-06-01:2022-08-31,2023-06-01:2023-08-31,2024-06-01:2024-08-31,2025-06-01:2025-08-31

# Минимум полётов на spot для SPOTS модели
MIN_FLIGHTS_PER_SPOT=200
```

### 2. Docker Compose настройка

**Важно:** `docker-compose.yml` должен содержать `shm_size: 10gb` для multiprocessing:

```yaml
services:
  paraglidable:
    shm_size: 10gb  # Обязательно для build_pkl_dataset.py
    volumes:
      - .:/workspaces/Paraglidable
      - /mnt/backup/Paraglidable:/mnt/backup/Paraglidable  # Если данные на другом диске
```

### 3. Запуск контейнера

```bash
# Первый запуск или после изменений docker-compose.yml
docker compose up -d

# Вход в контейнер
docker exec -it paraglidable bash
```

### 4. Полный pipeline обучения (внутри Docker)

```bash
# === Шаг 1: Сборка метаданных и метеоданных ===
python3 /workspaces/Paraglidable/scripts/build_pkl_dataset.py

# === Шаг 2: Восстановление meteo_days.pkl (если нужно) ===
python3 /workspaces/Paraglidable/scripts/update_meteo_days.py --rebuild

# === Шаг 3: Извлечение данных из xContest ===
python3 /workspaces/Paraglidable/scripts/extract_training_data.py

# === Шаг 4: Генерация PKL из xContest данных ===
python3 /workspaces/Paraglidable/scripts/build_pkl_from_xcontest.py

# === Шаг 5: Обучение ===
cd /workspaces/Paraglidable/neural_network
python train.py
```

---

## Архитектура обучения

**Альтернативные методы (не работают / не протестированы):**
- ⚠️ **IGC scraping** — DEPRECATED, не работает
- ❓ **UCAR RDA** — не протестирован (требует регистрацию)

## Архитектура обучения

### Две модели:

1. **CELLS Model** - Прогноз пригодности для каждой 1°x1° ячейки
   - Использует **динамически определённое количество ячеек** (из `sorted_cells_latlon.pkl`)
   - Количество ячеек задаётся через `TRAINING_BBOX` в `.env`
   - Предсказывает: flyability, crossability, wind-flyability, humidity-flyability

2. **SPOTS Model** - Прогноз для конкретных точек взлёта
   - Использует все доступные ячейки из датасета
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

**Источник:** xContest API через JSON
- **Формат:** JSON файлы с данными о полётах
- **Расширение:** `extensions/xcontest_data_collector/`

**Ключевые извлекаемые данные:**
1. Дата и время полёта
2. Координаты взлёта (lat, lon)
3. Высота точки взлёта
4. Балл/очки (score)
5. Информация о пилоте и планёре

### Агрегация данных:

**Ячейки:** 1° × 1° (lat × lon)
```python
cell_lat = int(takeoff_lat)  # например: 45 для lat=45.7°N
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

| Файл | Описание |
|------|----------|
| `sorted_cells_latlon.pkl` | Координаты ячеек (lat, lon) — задаётся через `TRAINING_BBOX` |
| `sorted_cells.pkl` | Индексы в GRIB сетке |
| `meteo_days.pkl` | Дни с погодными данными — фильтруется по `TRAINING_DATES` |
| `meteo_params.pkl` | Описание параметров (195 штук) |
| `meteo_content_by_cell_day.pkl` | Матрица погодных данных `[nb_days*nb_cells, 195]` |
| `mountainess_by_cell_alt.pkl` | Гористость по ячейкам `[nb_cells, 5]` |
| `flights_by_cell_day.pkl` | Полёты по ячейкам/дням (из PostgreSQL, опционально) |

### Настройка области обучения (`.env`)

```bash
# Область обучения (bbox)
TRAINING_BBOX=45,47,13,15  # 4 ячейки: Slovenia/Julian Alps

# Период обучения (летние месяцы 2021-2025)
TRAINING_DATES=2021-06-01:2021-08-31,2022-06-01:2022-08-31,2023-06-01:2023-08-31,2024-06-01:2024-08-31,2025-06-01:2025-08-31
```

---

## Подготовка данных (детальный pipeline)

### Шаг 0: Сбор исходных данных (один раз)

#### 0.1 Сбор данных полётов (xContest)

**Расширение:** `extensions/xcontest_data_collector/`

Собирает JSON файлы с данными о полётах через браузерное расширение xContest.

Результат: файлы в `data/flights/`:
- `xcontest_flights_YYYY-MM-DD-sl-XX.json` (Slovenia)
- `xcontest_flights_YYYY-MM-DD-ru-XX.json` (Russia)

#### 0.2 Скачивание GFS данных

**Скрипт:** `scripts/download_GFS.py`

```bash
python3 scripts/download_GFS.py \
  --start-date 2021-06-01 \
  --end-date 2025-08-31 \
  --data-dir data/gfs/anl \
  --hours 6,12,18 \
  --filter
```

Результат: GRIB файлы в `data/gfs/anl/` (несколько сотен GB)

---

### Шаг 1: Генерация базовых PKL файлов

**Скрипт:** `scripts/build_pkl_dataset.py`

**Выполняется внутри Docker:**
```bash
docker exec paraglidable python3 /workspaces/Paraglidable/scripts/build_pkl_dataset.py
```

**Что делает:**
1. Читает настройки из `.env` (`TRAINING_BBOX`, `TRAINING_DATES`)
2. Создаёт `sorted_cells_latlon.pkl` на основе bbox
3. Сканирует GFS файлы и создаёт `meteo_days.pkl`
4. Извлекает погодные данные из GRIB → `meteo_content_by_cell_day.pkl`
5. Вычисляет гористость → `mountainess_by_cell_alt.pkl`

**Создаваемые файлы:**
- `sorted_cells_latlon.pkl` — координаты ячеек
- `sorted_cells.pkl` — индексы в GRIB сетке
- `meteo_days.pkl` — список дней с данными
- `meteo_params.pkl` — описание 195 параметров
- `meteo_content_by_cell_day.pkl` — матрица погодных данных
- `mountainess_by_cell_alt.pkl` — гористость по ячейкам

**Важно:** По умолчанию пропускает данные полётов (`--include-flights` не указан).

---

### Шаг 2: Восстановление meteo_days.pkl (при необходимости)

**Скрипт:** `scripts/update_meteo_days.py`

```bash
# Пересоздать с нуля на основе файлов на диске
docker exec paraglidable python3 /workspaces/Paraglidable/scripts/update_meteo_days.py --rebuild
```

**Когда нужно:**
- После изменения `TRAINING_DATES` в `.env`
- Если `meteo_days.pkl` был случайно очищен

---

### Шаг 3: Извлечение данных из xContest

**Скрипт:** `scripts/extract_training_data.py`

```bash
docker exec paraglidable python3 /workspaces/Paraglidable/scripts/extract_training_data.py
```

**Что делает:**
1. Объединяет все JSON файлы из `data/flights/`
2. Вычисляет `mountainess` для каждого полёта
3. Фильтрует по bbox и датам из `.env`
4. Создаёт `data/flights/merged/training_flights.json`

**Выводит статистику:**
- Всего полётов
- Полётов с score (очки XC)
- Полётов вне области ячеек
- Полётов вне диапазона дат

---

### Шаг 4: Генерация PKL из xContest данных

**Скрипт:** `scripts/build_pkl_from_xcontest.py`

```bash
docker exec paraglidable python3 /workspaces/Paraglidable/scripts/build_pkl_from_xcontest.py
```

**Что делает:**
1. Читает `data/flights/merged/training_flights.json`
2. Создаёт SPOTS-специфичные PKL файлы

**Создаваемые файлы:**
- `spots.pkl` — список точек взлёта
- `spots_by_cell.pkl` — распределение spots по ячейкам
- `flights_by_spot.pkl` — полёты по точкам взлёта
- `flights_by_cell_day_spot.pkl` — полёты по (ячейка, день, spot)

---

### Шаг 5: Обучение моделей

**Скрипт:** `neural_network/train.py`

```bash
docker exec paraglidable bash -c "cd /workspaces/Paraglidable/neural_network && python train.py"
```

**Процесс:**
1. Обучает CELLS модель (по ячейкам)
2. Обучает SPOTS модель (по точкам взлёта)
3. Сохраняет веса в `neural_network/bin/models/CLASSIFICATION_2.0.0/weights/`

**Ожидаемое время:**
- 1 ячейка: ~10 минут
- 9 ячеек: ~1-2 часа (CPU)

---

## Обучающие данные

### Файлы в `neural_network/bin/data/`:

| Файл | Создаётся шагом | Описание |
|------|-----------------|----------|
| `sorted_cells_latlon.pkl` | 1 | Координаты ячеек (lat, lon) — из `TRAINING_BBOX` |
| `sorted_cells.pkl` | 1 | Индексы в GRIB сетке |
| `meteo_days.pkl` | 1,2 | Дни с погодными данными — фильтруется по `TRAINING_DATES` |
| `meteo_params.pkl` | 1 | Описание 195 параметров погоды |
| `meteo_content_by_cell_day.pkl` | 1 | Матрица `[nb_days*nb_cells, 195]` |
| `mountainess_by_cell_alt.pkl` | 1 | Гористость `[nb_cells, 5]` |
| `spots.pkl` | 4 | Список точек взлёта (для SPOTS модели) |
| `spots_by_cell.pkl` | 4 | Распределение spots по ячейкам |
| `flights_by_spot.pkl` | 4 | Полёты по точкам взлёта |
| `flights_by_cell_day_spot.pkl` | 4 | Полёты по (ячейка, день, spot) |

---

## Troubleshooting

### Ошибка: "No space left on device"

**Проблема:** `/dev/shm` переполнен при multiprocess-обработке GRIB.

**Решение:**
```bash
# 1. Остановить контейнер
docker compose down

# 2. Добавить в docker-compose.yml:
#    shm_size: 10gb

# 3. Перезапустить
docker compose up -d
```

### Ошибка: "ValueError: ... is not in list" при обучении

**Проблема:** `meteo_params.pkl` пустой или несовместим.

**Решение:** Пересобрать метеданные:
```bash
docker exec paraglidable python3 /workspaces/Paraglidable/scripts/build_pkl_dataset.py
```

### Мало спотов для обучения

**Проверьте:**
```bash
# Статистика по ячейкам
docker exec paraglidable bash -c "cat /workspaces/Paraglidable/data/flights/merged/stats.json"

# Проверить spots
docker exec paraglidable python3 /workspaces/Paraglidable/scripts/build_pkl_from_xcontest.py
```

**Решения:**
- Увеличить `TRAINING_BBOX` для большего покрытия
- Уменьшить `MIN_FLIGHTS_PER_SPOT` в `.env`
- Собрать больше данных полётов

### Плохой loss (>0.8)

**Причины:**
1. Мало обучающих данных (<1000 полётов)
2. Слишком узкая область (меньше 4 ячеек)
3. Данные не覆盖ют разные погодные условия

**Решения:**
- Увеличить период `TRAINING_DATES`
- Расширить `TRAINING_BBOX`
- Проверить качество данных (полноту score)

---

## Полезные команды

```bash
# === Проверка данных ===
# Размеры ячеек и период
docker exec paraglidable python3 -c "
import pickle
cells = pickle.load(open('/workspaces/Paraglidable/neural_network/bin/data/sorted_cells_latlon.pkl', 'rb'))
days = pickle.load(open('/workspaces/Paraglidable/neural_network/bin/data/meteo_days.pkl', 'rb'))
print(f'Ячеек: {len(cells)}')
print(f'Период: {days[0]} - {days[-1]}')
print(f'Дней: {len(days)}')
"

# Статистика по flights
docker exec paraglidable bash -c "cat /workspaces/Paraglidable/data/flights/merged/stats.json"

# === Пересборка данных ===
# Только метеданные (без flights)
docker exec paraglidable python3 /workspaces/Paraglidable/scripts/build_pkl_dataset.py

# Обновление meteo_days
docker exec paraglidable python3 /workspaces/Paraglidable/scripts/update_meteo_days.py --rebuild

# Пересборка flights из xContest
docker exec paraglidable python3 /workspaces/Paraglidable/scripts/extract_training_data.py
docker exec paraglidable python3 /workspaces/Paraglidable/scripts/build_pkl_from_xcontest.py

# === Обучение ===
# Полное обучение
docker exec paraglidable bash -c "cd /workspaces/Paraglidable/neural_network && python train.py"

# === Диагностика ===
# Проверить spots по ячейкам
docker exec paraglidable bash -c "cd /workspaces/Paraglidable/neural_network && python3 -c '
from inc.dataset import SpotsData
s = SpotsData()
for i in range(9):
    spots = s.getSpots([i])
    if spots and spots[0]:
        print(f\"Cell {i}: {len(spots[0])} spots\")
'"

# Проверить /dev/shm
docker exec paraglidable df -h /dev/shm
```

---

## Формат метрических данных

### meteo_content_by_cell_day.pkl

**Размерность:** `[nb_days * nb_cells, 195]`

- `nb_days`: дней из `TRAINING_DATES`
- `nb_cells`: ячеек из `TRAINING_BBOX`
- `nb_parameters`: 195 погодных параметров (65 × 3 часа)

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

### mountainess (гористость)

**Вычисление:** `scripts/elevation_reader.py`

```python
def get_mountainess(lat, lon) -> float:
    # Семплирует высоту в сетке 5×5 (радиус ~5км)
    # Формула: (max_elev - min_elev) / 800
    # Ограничивается [0, 1]
```

**Значения:**
- `0.0` — равнина
- `0.4-0.6` — холмистая местность
- `1.0` — горы

---

## Скачивание GFS данных

### Источники данных

| Источник | Период | Статус | Примечания |
|----------|--------|--------|------------|
| **AWS S3** | 2021+ | ✅ Работает | Публичный, рекомендуется |
| **NCAR RDA** | 2000-2019+ | ❓ Не протестирован | Требует регистрацию |
| **dynamical.org** | 2015-2024 | ❓ Не протестирован | Zarr формат |
| **NOMADS** | ~30 дней | ❓ Не протестирован | Только последние дни |

### Способ 1: AWS S3 (для данных 2021+)

**Скрипт:** `scripts/download_GFS.py`

**Пример использования:**

```bash
# Скачивание с фильтрацией по параметрам
python3 scripts/download_GFS.py \
  --start-date 2025-06-01 \
  --end-date 2025-08-31 \
  --data-dir data/gfs/anl \
  --hours 6,12,18 \
  --filter
```

### Способ 2: NCAR RDA (для исторических данных 2000-2020+)

**⚠️ НЕ ПРОТЕСТИРОВАНО** — требует регистрацию на https://rda.ucar.edu/

**Скрипт:** `scripts/download_GFS_rda.py`

**Настройка в `.env`:**
```bash
UCAR_EMAIL=your@email.com
UCAR_PASS=your_password
```

**Пример использования:**
```bash
python3 scripts/download_GFS_rda.py 2012-06-01 2012-08-31 data/gfs/anl
```

---

## Обучение

### Файл: `neural_network/train.py`

```bash
cd neural_network/
python train.py
```

**Процесс:**
1. Для каждой ячейки:
   - Загрузить погодные данные для дней с полётами
   - Загрузить данные о полётах
   - Обучить модель предсказывать вероятность полёта

2. Создаётся `population_alt_cell_XX.npy` для каждой ячейки:
   - Содержит 5 значений (по одному на высоту)
   - Хранится в `bin/models/CLASSIFICATION_1.0.0/weights/`

---

## Полезные команды

```bash
# Генерация PKL файлов
python3 scripts/build_pkl_dataset.py --skip-flights

# Извлечение данных из xContest
python3 scripts/extract_training_data.py --cluster-distance 15

# Обновление meteo_days.pkl
python3 scripts/update_meteo_days.py

# Обновление meteo_days.pkl с пересозданием
python3 scripts/update_meteo_days.py --rebuild

# Запуск обучения
cd neural_network/
python train.py

# Генерация прогноза
python forecast.py

# Проверка ячеек и периодов
python3 -c "
import pickle
cells = pickle.load(open('neural_network/bin/data/sorted_cells_latlon.pkl', 'rb'))
days = pickle.load(open('neural_network/bin/data/meteo_days.pkl', 'rb'))
print(f'Ячеек: {len(cells)}')
print(f'Период: {days[0]} - {days[-1]}')
print(f'Дней: {len(days)}')
"
```

---

## IGC scraping (DEPRECATED)

⚠️ **НЕ ИСПОЛЬЗУЕТСЯ** — метод парсинга IGC файлов больше не поддерживается.

**Причины:**
- Требует scraping множества сайтов
- Нестабильные API
- Сложно поддерживать

Используйте **xContest API** вместо этого.

---

## Обновление метаданных

### Скрипт: `scripts/update_meteo_days.py`

**Назначение:** Обновляет `meteo_days.pkl` при добавлении новых GFS файлов

```bash
# Добавить новые дни (слияние с существующими)
python3 scripts/update_meteo_days.py

# Пересоздать с нуля (удалить несуществующие на диске)
python3 scripts/update_meteo_days.py --rebuild
```

**Читает настройки из `.env`:**
- `PROJECT_ROOT` — корень проекта
- `GFS_DIR` — директория с GFS файлами
- `PKL_DIR` — директория для PKL файлов
