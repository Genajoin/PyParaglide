# PKL структуры и пайплайн IGC → dataset

Полное руководство по созданию датасета для обучения нейронной сети из IGC файлов и метеоданных.

---

## Часть 1: PKL структуры

Формальное описание структур `*.pkl` в `neural_network/bin/data/`.

Все PKL читаются через `BinObj.load()` (`neural_network/inc/bin_obj.py`), путь по умолчанию — `neural_network/bin/data`.

### 1.1 `sorted_cells_latlon.pkl`

Список ячеек обучения в градусах.

- Тип: `List[Tuple[float, float]]`
- Значение: `(lat, lon)` для центра 1°×1° ячейки
- Порядок: должен быть стабильным и совпадать для всех связанных pkl

### 1.2 `sorted_cells.pkl`

Индексы ячеек в GRIB‑сетке.

- Тип: `List[Tuple[int, int]]`
- Значение: `(row, col)` индексы в `grb.values` (lat/lon сетка GFS)
- Порядок совпадает с `sorted_cells_latlon.pkl`

### 1.3 `meteo_days.pkl`

Список дней, для которых есть метео‑данные.

- Тип: `List[datetime.date]`
- Порядок: по возрастанию даты

### 1.4 `meteo_params.pkl`

Полный список погодных параметров в разрезе 3 временных шагов.

- Тип: `List[Tuple[int, str, List[Tuple[str, int]]]]`
- Формат элемента: `(hour, param_name, level_list)`
  - `hour` ∈ {6, 12, 18}
  - `param_name` как в GRIB (например `"Temperature"`)
  - `level_list` как в `GfsData.parameters_vector_all`

В сумме 195 параметров = 65 параметров × 3 часа.

### 1.5 `meteo_content_by_cell_day.pkl`

Матрица погодных параметров для каждого дня и ячейки.

- Тип: `np.ndarray`
- Размер: `[nb_days * nb_cells, 195]`
- Порядок строк: **day‑major**:
  - индекс строки = `day_idx * nb_cells + cell_idx`

### 1.6 `mountainess_by_cell_alt.pkl`

Гео‑фактор для модели (учёт рельефа).

- Тип: `List[List[float]]`
- Размер: `[nb_cells][5]`
- Значения в диапазоне `[0, 1]`, обычно одинаковые для 5 высот

### 1.7 `flights_by_cell_day.pkl`

Список полётов по ячейкам и дням.

- Тип: `List[List[Tuple[str, Tuple[float, float, float]]]]`
- Размер списка: `nb_days * nb_cells`
- Индекс: `day_idx * nb_cells + cell_idx`
- Один полёт:
  - `("YYYY-MM-DD HH:MM:SS", (score, lat, lon))`
    - `score`: XC score (баллы) из xContest API, используется для crossability (порог 60)
    - `lat`, `lon`: координаты старта (используются для super-resolution позиционирования)

**Примечание:** Altitude binning удалён. Ранее структура включала `alt`, `plaf`, `takeoff_alt`, `mountainess`:
- `alt` и `plaf` были из IGC файлов (недоступны в xContest JSON)
- `takeoff_alt` использовался для биннинга по высотам (теперь агрегированы)
- `mountainess` теперь берётся из отдельного файла `mountainess_by_cell_alt.pkl` (усреднённая)

### 1.8 `spots.pkl`, `spots_merged.pkl`

Список стартов (спотов).

- Тип: `List[Tuple[str, float, float]]`
- Формат: `(name, lat, lon)`

`spots_merged` — результат слияния близких стартов (`SpotsData.__fusion_of_close_spots`).

### 1.9 `spots_by_cell.pkl`

Маппинг спотов к ячейкам.

- Тип: `List[List[int]]`
- Размер: `[nb_cells]`
- Значение: список индексов спотов в ячейке

### 1.10 `flights_by_spot.pkl`

Список полётов по спотам.

- Тип: `List[List[Tuple[str, Tuple[float, float, float, float, float]]]]`
- Формат полёта:
  - `("YYYY-MM-DD HH:MM:SS", (score, alt, plaf, lat, lon))`

### 1.11 `flights_by_cell_day_spot.pkl`

Полетные записи на пересечении (ячейка, день, спот).

- Тип: `List[List[Dict[int, List[Tuple[str, Tuple[float, float, float, float, float]]]]]]`
- Размер: `[nb_cells][nb_days]`
- Ключ словаря: `spot_id`, значение — список полётов

---

## Часть 2: Пайплайн создания PKL датасета

```
┌─────────────────────────────────────────────────────────────┐
│ 1. Сбор IGC файлов                                          │
│    └─ igc_ingest_skygr.py → PostgreSQL                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. Расширенный парсинг IGC                                  │
│    └─ update_flights_metadata.py → БД flights              │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. Создание PKL структур                                    │
│    ├─ build_pkl_dataset.py                                 │
│    ├─ Ячейки: sorted_cells*.pkl                            │
│    ├─ Метео: meteo_*.pkl (из GFS GRIB)                     │
│    ├─ Полёты: flights_by_cell_day.pkl                      │
│    └─ Рельеф: mountainess_by_cell_alt.pkl                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. Обучение нейронной сети                                  │
│    └─ neural_network/train.py                              │
└─────────────────────────────────────────────────────────────┘
```

### Шаг 0: Подготовка окружения

#### Установка зависимостей

```bash
# Python зависимости
pip install tqdm numpy psycopg[binary] pygrib

# Для работы с GRIB файлами
# Ubuntu/Debian:
sudo apt-get install libeccodes-dev
pip install pygrib

# macOS:
brew install eccodes
pip install pygrib
```

#### Структура директорий

```
Paraglidable/
├── data/
│   └── igc/
│       └── skygr/          # IGC файлы
│           └── YYYY/MM/flight_id/
├── data/gfs/anl/           # GFS GRIB файлы
│   └── YYYY-MM/
│       └── gfsanl_3_YYYYMMDD_HH00_000.grb2
├── tiler/_cache/elevation/ # Elevation tiles
│   └── 7/tx/ty.mountainess
├── neural_network/bin/data/ # Выходные PKL файлы
└── scripts/                # Скрипты
```

### Шаг 1: Сбор IGC файлов

#### Автоматическая загрузка с Leonardo

```bash
cd scripts/

# Загрузить список полётов (с пагинацией)
python igc_ingest_skygr.py \
  --db-url "postgresql://user:pass@localhost/igc" \
  --source skygr \
  --max-pages 10 \
  --list-only

# Загрузить IGC файлы
python igc_ingest_skygr.py \
  --db-url "postgresql://user:pass@localhost/igc" \
  --source skygr \
  --max-flights 1000 \
  --out-dir ../data/igc/skygr
```

### Шаг 2: Расширенный парсинг IGC

```bash
# Обновить только файлы где distance_km IS NULL
python update_flights_metadata.py \
  --db-url "postgresql://user:pass@localhost/igc" \
  --source skygr \
  --stats

# Полный reparse (если нужно)
python update_flights_metadata.py \
  --db-url "postgresql://..." \
  --source skygr \
  --full \
  --stats
```

### Шаг 3: Скачивание GFS Analysis данных

Для построения PKL датасета нужны исторические GFS Analysis данные из NOAA.

**Скрипт:** `scripts/download_GFS.py`

**Источник:** AWS S3 Open Data (https://registry.opendata.aws/noaa-gfs-bdp-pds/)

**Формат файлов:** `gfsanl_3_YYYYMMDD_HH00_000.grb2` (legacy naming)
- Часы: 0, 6, 12, 18 UTC
- Формат: GRIB2 (0.25° resolution)
- Размер: ~490MB на файл (полный), ~190MB (отфильтрованный)
- Содержимое: 696 параметров (полный) → 233 (отфильтрованный)

**Структура хранения:**
```
data/gfs/anl/
└── YYYY-MM/
    └── gfsanl_3_YYYYMMDD_HH00_000.grb2
```

**Примеры использования:**

```bash
# Скачать данные для тестирования
python scripts/download_GFS.py \
  --start-date 2025-12-21 \
  --end-date 2025-12-26 \
  --data-dir data/gfs/anl

# Скачать для конкретного месяца
python scripts/download_GFS.py \
  --start-date 2025-06-01 \
  --end-date 2025-06-30 \
  --data-dir data/gfs/anl

# Скачать только определенные часы
python scripts/download_GFS.py \
  --start-date 2025-12-01 \
  --end-date 2025-12-31 \
  --data-dir data/gfs/anl \
  --hours 6,18
```

### Шаг 3.1: Оптимизация хранения GFS данных (фильтрация)

**Проблема:** Полные GRIB файлы содержат ~700 параметров, но для обучения используется только ~65 (9 переменных × 11 уровней давления).

**Решение:** Фильтрация GRIB файлов для экономии ~70% дискового пространства.

#### Скрипт фильтрации: `scripts/filter_gfs.py`

```bash
# Dry-run (проверка без изменений)
python scripts/filter_gfs.py data/gfs/anl --dry-run

# Фильтрация всех файлов
python scripts/filter_gfs.py data/gfs/anl

# Фильтрация конкретного месяца
python scripts/filter_gfs.py data/gfs/anl/2025-06 -v
```

**Результаты фильтрации:**
- 696 messages → 233 messages (33%)
- 488 MB → 187 MB на файл
- Экономия: ~70% дискового пространства

**Используемые параметры** (соответствуют `neural_network/inc/dataset.py`):
- Temperature, U/V wind, Relative humidity
- Geopotential Height, Vertical velocity, Absolute vorticity
- Precipitable water, Cloud water
- Уровни: 200-1000 hPa + entire atmosphere

#### Автоматическая фильтрация при скачивании

```bash
# Скачать данные с автоматической фильтрацией
python scripts/download_GFS.py \
  --start-date 2025-06-01 \
  --end-date 2025-06-30 \
  --filter
```

**Совместимость:** Отфильтрованные файлы полностью совместимы с `GribReader` (`neural_network/inc/grib_reader.py`).

### Шаг 4: Создание PKL датасета

#### Базовая команда

```bash
cd scripts/

python build_pkl_dataset.py \
  --db-url "postgresql://user:pass@localhost/igc" \
  --bbox "43,49,4,37" \
  --gfs-dir ../data/gfs/anl \
  --elevation-dir ../tiler/_cache/elevation \
  --out-dir ../neural_network/bin/data \
  --source skygr
```

#### Параметры команды

| Параметр | Описание | Пример |
|----------|----------|--------|
| `--db-url` | PostgreSQL connection URL | `postgresql://user:pass@localhost/igc` |
| `--bbox` | Bounding box: lat_min,lat_max,lon_min,lon_max | `43,49,4,37` |
| `--gfs-dir` | Путь к GFS GRIB файлам | `../data/gfs/anl` |
| `--elevation-dir` | Путь к elevation tiles | `../tiler/_cache/elevation` |
| `--out-dir` | Выходная директория для PKL | `../neural_network/bin/data` |
| `--source` | Источник полётов | `skygr` |
| `--skip-meteo` | Пропустить обработку метео | flag |
| `--skip-flights` | Пропустить обработку полётов | flag |

#### Вывод скрипта

```
=== PKL Dataset Builder ===
Bbox: (43.0, 49.0, 4.0, 37.0)
Output directory: /path/to/neural_network/bin/data

Connecting to database...

=== Step 1: Building cells ===
  Generated 198 cells
  Lat range: 43°..49°
  Lon range: 4°..37°
  Mapping cells to GRIB grid...
  Saving sorted_cells_latlon.pkl...
  Saving sorted_cells.pkl...

=== Step 2: Scanning GFS data ===
  Found 365 days with complete GFS data
  Building meteo_params...
    Total parameters: 195
  Saving meteo_days.pkl...
  Saving meteo_params.pkl...

=== Step 3: Extracting weather data ===
  Matrix shape: (72270, 195)  # 365 days * 198 cells
  Saving meteo_content_by_cell_day.pkl...

=== Step 4: Building flights_by_cell_day ===
  Querying flights from database...
  Processing 5432 flights...
  Processing: 100%|████████| 5432/5432 [00:15<00:00]
  Successfully processed: 4823 flights
  Skipped (not in meteo_days): 509 flights
  Skipped (outside bbox): 100 flights
  Saving flights_by_cell_day.pkl...

=== Step 5: Building mountainess_by_cell_alt ===
  Processing cells: 100%|████| 198/198 [00:02<00:00]
  Saving mountainess_by_cell_alt.pkl...

=== Validation ===
  ✓ sorted_cells_latlon.pkl
  ✓ sorted_cells.pkl
  ✓ meteo_days.pkl
  ✓ meteo_params.pkl
  ✓ meteo_content_by_cell_day.pkl
  ✓ flights_by_cell_day.pkl
  ✓ mountainess_by_cell_alt.pkl

  Dimensions:
    Cells: 198
    Days: 365
    Parameters: 195
    Meteo shape: (72270, 195)
    Flights: 72270 (days*cells)
    Mountainess: 198x5
    Total flights: 4823

✓ Dataset built successfully!
```

### Шаг 5: Проверка PKL файлов

```bash
cd neural_network/

# Проверить загрузку
python3 -c "
from inc.bin_obj import BinObj

cells = BinObj.load('sorted_cells_latlon')
days = BinObj.load('meteo_days')
flights = BinObj.load('flights_by_cell_day')

print(f'Cells: {len(cells)}')
print(f'Days: {len(days)}')
print(f'Flights structure: {len(flights)} entries')
print(f'Total flights: {sum(len(dc) for dc in flights)}')
"

# Проверить DatasetParams
python3 -c "
from inc.dataset import DatasetParams

params = DatasetParams()
print(f'nb_cells: {params.nb_cells}')
print(f'nb_days: {params.nb_days}')
"
```

### Шаг 6: Обучение нейронной сети

```bash
cd neural_network/

# Запуск обучения
python train.py
```

---

## Часть 3: Восстановленный пайплайн IGC → PKL

### 3.1 Выбор области и разрешения

Параметры:
- `lat_min`, `lat_max`, `lon_min`, `lon_max`
- Разрешение ячейки = 1° (текущее)

Алгоритм:
- Формируем список ячеек: `lat = floor(lat_min) .. ceil(lat_max)-1`,
  `lon = floor(lon_min) .. ceil(lon_max)-1`
- Порядок фиксируем (например, `lat` по возрастанию, внутри `lon` по возрастанию)
- `sorted_cells_latlon` = список `(lat, lon)` в этом порядке

### 3.2 Построение `sorted_cells.pkl`

Нужен любой GRIB файл из GFS (0.25°):

1. Открываем файл через `pygrib` или `GribReader.getInfos()`
2. Берём массивы `distinctLatitudes`, `distinctLongitudes`
3. Для каждой ячейки `(lat, lon)` ищем ближайшие индексы:
   `row = argmin(|lats - lat|)`, `col = argmin(|lons - lon|)`
4. Сохраняем `(row, col)` в том же порядке, что `sorted_cells_latlon`

### 3.3 Список дней `meteo_days.pkl`

На основании архива GFS analysis:

- Для каждого дня в диапазоне, проверяем наличие файлов для 06:00, 12:00, 18:00
- В `meteo_days` включаем только дни, где доступны все 3 файла
- Храним как `datetime.date`

### 3.4 Список параметров `meteo_params.pkl`

Используем `TrainedModel.meteoParams()`:

- Параметры для humidity, wind, other
- Каждый элемент = `(hour, name, level_list)`
- Порядок должен соответствовать метрике обучения (строго фиксированный)

### 3.5 Матрица `meteo_content_by_cell_day.pkl`

Для каждого дня:
1. Загружаем 3 GRIB файла (06, 12, 18)
2. Для каждого часа извлекаем `GfsData.parameters_vector_all`
   по всем выбранным ячейкам
3. Формируем матрицу `[nb_cells, 65*3]`
4. Складываем в общий массив в порядке `day_idx * nb_cells + cell_idx`

### 3.6 Формирование `flights_by_cell_day.pkl`

Источники:
- таблица `flights` в Postgres (результат ingestion)

Шаги:
1. Для каждого скачанного IGC используем `parse_igc_with_libs.py`:
   - Парсинг через `parse_igc_full(igc_path)` → возвращает метаданные
   - `takeoff_datetime` → время взлёта (ISO формат)
   - `takeoff_lat`, `takeoff_lon` → координаты старта
   - `takeoff_alt` → высота старта (баро или GPS fallback)
   - `plaf` → max_alt (потолок)
   - `xc_score` → XC score (баллы, FAI triangles/free distance)
2. Фильтруем по bbox (lat/lon)
3. Ищем `day_idx` в `meteo_days` по `takeoff_datetime`
4. Определяем `cell_idx` по `(takeoff_lat, takeoff_lon)` (floor)
5. Добавляем запись в `flights_by_cell_day[day_idx * nb_cells + cell_idx]`

### 3.7 `mountainess_by_cell_alt.pkl`

Берём `tiler/_cache/elevation`:

1. Для каждого `(lat, lon)` вычисляем `TilesMaths.LatLonToTileCoords`
2. Читаем `*.mountainess`, получаем значение `[0..1]`
3. Дублируем для 5 высот

### 3.8 Spots‑пакет

Если нужны спотовые модели:

1. Генерируем базовые `spots.pkl`:
   - По `takeoff_name` (если доступен),
   - или кластеризуем координаты (DBSCAN / grid clustering),
   - создаём синтетические имена `spot_<id>`
2. Запускаем `SpotsData.__compute_spots_information`:
   - создаёт `spots_merged`, `spots_by_cell`, `flights_by_spot`,
     `flights_by_cell_day_spot`

---

## Часть 4: Расширение области обучения

### Новая область: Альпы + Восточная Европа

```bash
# Текущая: 43°-49°N, 4°-18°E (97 ячеек)
# Новая: 43°-49°N, 4°-37°E (198 ячеек)

python build_pkl_dataset.py \
  --db-url "postgresql://..." \
  --bbox "43,49,4,37" \
  --gfs-dir ../data/gfs/anl \
  --elevation-dir ../tiler/_cache/elevation \
  --out-dir ../neural_network/bin/data \
  --source skygr

# Новые регионы:
# - Карпаты (19°-25°E)
# - Балканы (20°-24°E)
# - Крым (33°-37°E)
# - Часть Кавказа (37°E)
```

### Обновление train.py для новой области

```python
# В neural_network/train.py:

# Старое:
self.all_cells = list(range(55))  # Только Альпы

# Новое:
self.all_cells = list(range(120))  # Альпы + Восток
```

---

## Часть 5: Troubleshooting

### Проблема: "No meteo days found"

**Решение:** Проверьте структуру GFS директории и наличие файлов:

```bash
# Должна быть структура:
data/gfs/anl/YYYY-MM/gfsanl_3_YYYYMMDD_HH00_000.grb2

# Проверить:
ls -R data/gfs/anl/
```

### Проблема: "distance_km IS NULL for all flights"

**Решение:** Сначала запустите `update_flights_metadata.py`:

```bash
python update_flights_metadata.py \
  --db-url "postgresql://..." \
  --source skygr
```

### Проблема: "Mountainess file not found"

**Решение:** Скачайте elevation tiles:

```bash
cd scripts/
python download_elevation_tiles.py
```

### Проблема: "pygrib not found"

**Решение:** Установите ECCODES и pygrib:

```bash
# Ubuntu/Debian
sudo apt-get install libeccodes-dev
pip install pygrib

# macOS
brew install eccodes
pip install pygrib
```

---

## Часть 6: Связанные документы

- `specs/TRAINING_PROCESS.md` - Процесс обучения
- `specs/MULTIPROCESSING.md` - Многопроцессорная обработка
- `specs/IGC_PARSING.md` - Детали парсинга IGC файлов

---

## Контрольный список

### Перед созданием датасета:

- [ ] IGC файлы скачаны (`igc_ingest_skygr.py`)
- [ ] Метаданные обновлены (`update_flights_metadata.py`)
- [ ] GFS GRIB файлы доступны (06, 12, 18 UTC)
- [ ] Elevation tiles скачаны
- [ ] БД содержит поля: `distance_km`, `takeoff_alt`, `plaf`, `takeoff_datetime`

### После создания датасета:

- [ ] Все 7 PKL файлов созданы
- [ ] Валидация прошла успешно
- [ ] `DatasetParams()` корректно загружает данные
- [ ] Запуск `train.py` без ошибок
