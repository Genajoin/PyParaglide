# Парсинг IGC файлов

Эти скрипты извлекают полные метаданные из IGC файлов, включая высоты, дистанцию, XC score и статистику термиков/скольжений.

## Файлы

### 1. `parse_igc_with_libs.py` ⭐ основной парсер

Современный IGC парсер использующий библиотеки:

- **libigc** — анализ полёта (термики, скольжения, статистика)
- **xc_score** — расчёт XC score (FAI треугольники, free distance, XContest-style)
- **parse_igc_extended** — fallback если библиотеки недоступны (deprecated)

**Извлекаемые данные:**

Базовые (через `igc_ingest_skygr.py`):
- `flight_date` - дата полёта
- `pilot` - имя пилота
- `glider` - тип планера
- `glider_class` - класс планера
- `takeoff_lat`, `takeoff_lon` - координаты старта
- `landing_lat`, `landing_lon` - координаты посадки

Метрики полёта (libigc):
- `takeoff_datetime` - точное время старта ("YYYY-MM-DD HH:MM:SS")
- `landing_datetime` - точное время посадки
- `takeoff_alt` - высота точки старта (метры)
- `max_alt` - максимальная высота полёта (метры)
- `min_alt` - минимальная высота (метры)
- `plaf` - потолок полёта (= max_alt)
- `duration_sec` - длительность полёта (секунды)
- `track_points` - количество точек трека
- `distance_km` - дистанция по треку (километры, haversine formula)

Термическая активность (libigc):
- `thermal_count` - количество термов
- `glide_count` - количество скольжений
- `avg_climb_rate` - средняя скорость набора (м/с)
- `max_climb_rate` - максимальная скорость набора (м/с)
- `avg_sink_rate` - средняя скорость снижения (м/с)

XC Score (xc_score):
- `xc_score` - очки XC (с учётом множителя за тип полёта)
- `xc_distance_km` - дистанция XC (км)
- `xc_type` - тип полёта ("free_distance", "FAI_triangle", "flat_triangle", etc)

**Использование:**

```bash
# Парсинг одного файла
python scripts/parse_igc_with_libs.py path/to/flight.igc

# Вывод в JSON
python scripts/parse_igc_with_libs.py flight.igc | jq .
```

**Пример вывода:**

```json
{
  "flight_date": "2025-02-02",
  "pilot": "Tsoukas",
  "glider": "EN-C",
  "takeoff_datetime": "2025-02-02 13:51:43",
  "landing_datetime": "2025-02-02 15:21:13",
  "takeoff_lat": 38.064416666666666,
  "takeoff_lon": 23.37535,
  "landing_lat": 38.1052,
  "landing_lon": 23.5125,
  "takeoff_alt": 666.0,
  "max_alt": 1247.0,
  "plaf": 1247.0,
  "min_alt": 130.0,
  "duration_sec": 5370,
  "track_points": 1771,
  "distance_km": 47.52,
  "thermal_count": 12,
  "glide_count": 11,
  "avg_climb_rate": 1.8,
  "max_climb_rate": 3.2,
  "avg_sink_rate": -1.1,
  "xc_score": 57.02,
  "xc_distance_km": 47.52,
  "xc_type": "free_flight"
}
```

**Зависимости:**

```bash
# Рекомендуемые библиотеки
pip install libigc

# xc_score включён в scripts/xc_score/ (локальная копия)
```

### 2. `parse_igc_extended.py` ⚠️ DEPRECATED

⚠️ **Этот модуль УСТАРЕЛ. Используйте `parse_igc_with_libs.py` вместо него.**

Оставлен только как fallback для случаев когда libigc/xc_score недоступны.

Не предоставляет:
- Анализ термов и скольжений
- XC score с множителями за тип полёта
- Точную статистику вертикальных скоростей

### 3. `update_flights_metadata.py`

Скрипт для обновления БД из IGC файлов.

**Функционал:**
1. Добавляет новые поля в таблицу `flights` (если не существуют)
2. Читает IGC файлы из `igc_path`
3. Парсит через `parse_igc_with_libs.py`
4. Обновляет БД новыми метаданными

**Новые поля в БД:**
- `takeoff_datetime TEXT`
- `landing_datetime TEXT`
- `takeoff_alt DOUBLE PRECISION`
- `max_alt DOUBLE PRECISION`
- `plaf DOUBLE PRECISION`
- `min_alt DOUBLE PRECISION`
- `distance_km DOUBLE PRECISION`
- `xc_score DOUBLE PRECISION` — XC score (points)
- `xc_distance_km DOUBLE PRECISION` — XC distance (km)
- `xc_type TEXT` — тип XC полёта
- `thermal_count INTEGER` — количество термов
- `glide_count INTEGER` — количество скольжений
- `avg_climb_rate DOUBLE PRECISION` — средняя скорость набора (м/с)
- `max_climb_rate DOUBLE PRECISION` — максимальная скорость набора (м/с)

**Использование:**

```bash
# Incremental update (только где xc_score IS NULL)
python scripts/update_flights_metadata.py \
  --db-url "postgresql://user:pass@localhost/igc" \
  --source skygr

# Full reparse (все файлы)
python scripts/update_flights_metadata.py \
  --db-url "postgresql://user:pass@localhost/igc" \
  --source skygr \
  --full

# С лимитом для тестирования
python scripts/update_flights_metadata.py \
  --db-url "postgresql://..." \
  --max-files 100 \
  --stats

# Многопоточная обработка
python scripts/update_flights_metadata.py \
  --db-url "postgresql://..." \
  --workers 8

# Использование переменной окружения для БД
export IGC_DB_URL="postgresql://user:pass@localhost/igc"
python scripts/update_flights_metadata.py --source skygr
```

**Параметры:**

| Параметр | Описание |
|----------|----------|
| `--db-url` | PostgreSQL connection URL (по умолчанию из `IGC_DB_URL` или `localhost`) |
| `--source` | Источник данных (по умолчанию: `skygr`) |
| `--max-files` | Лимит файлов для обработки (по умолчанию: все) |
| `--full` | Полный reparse (иначе incremental - только где `xc_score IS NULL`) |
| `--stats` | Показать статистику после обновления |
| `--workers` | Количество параллельных процессов (по умолчанию: CPU count) |

**Пример вывода:**

```
Connecting to database...
Updating database schema...
Schema updated successfully

Processing 1523 flights with 8 workers...
Press Ctrl+C to gracefully stop (waits for current files to finish)

Successfully updated: 1523 flights

Statistics:
  Total flights: 5432
  With distance_km: 1523
  With xc_score: 1523
  With thermal_count: 1523

Sample updated flights:
  115155: date=2025-02-02, datetime=2025-02-02 13:11:55, takeoff_alt=658.0m, max_alt=1451.0m, plaf=1451.0m, distance=49.6km, xc_score=59.52, thermals=15
  115080: date=2025-02-02, datetime=2025-02-02 13:51:43, takeoff_alt=666.0m, max_alt=1247.0m, plaf=1247.0m, distance=47.5km, xc_score=57.02, thermals=12
```

## Алгоритмы

### libigc анализ

libigc автоматически обнаруживает:
- **Термики** — участки постоянного набора высоты
- **Скольжения** — участки снижения между термиками
- **Статистику** — средние/максимальные вертикальные скорости

### XC Scoring (xc_score)

Использует XContest-style правила с множителями:

| Тип полёта | Множитель | Описание |
|------------|-----------|----------|
| `free_flight` | 1.0 | Свободный полёт (максимальная дистанция) |
| `flat` | 1.2 | Плоский треугольник |
| `FAI` | 1.4 | FAI треугольник (закрытый, 28% правило) |
| `closedFlat` | 1.4 | Закрытый плоский треугольник |
| `closedFAI` | 1.6 | Закрытый FAI треугольник |

```
score = distance_km × multiplier
```

### Приоритет высот

При парсинге B-записей используется приоритет:
1. **Барометрическая высота** (позиции 25-30)
2. **GPS высота** (позиции 30-35) - fallback
3. **0.0** - если оба поля пустые

Результат помечается в `altitude_source`:
- `"baro"` - все точки с барометрической высотой
- `"gps"` - все точки с GPS высотой
- `"mixed"` - часть точек baro, часть GPS
- `"none"` - нет данных о высотах

### Вычисление distance_km

Используется **Haversine formula** для вычисления расстояния между последовательными точками трека:

**Формула:**
```
a = sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)
c = 2 * atan2(√a, √(1-a))
distance = R * c  (R = 6371 км - радиус Земли)
```

## Валидация после обновления

```sql
-- Проверить заполненность новых полей
SELECT
    COUNT(*) as total,
    COUNT(distance_km) as with_distance,
    COUNT(takeoff_alt) as with_takeoff_alt,
    COUNT(plaf) as with_plaf,
    COUNT(xc_score) as with_xc_score,
    COUNT(thermal_count) as with_thermals
FROM flights
WHERE source='skygr';

-- Примеры данных с XC score
SELECT
    flight_id,
    flight_date,
    takeoff_datetime,
    takeoff_alt,
    max_alt,
    plaf,
    distance_km,
    xc_score,
    xc_type,
    thermal_count
FROM flights
WHERE source='skygr' AND xc_score IS NOT NULL
ORDER BY xc_score DESC
LIMIT 10;
```

## Edge Cases

### 1. Нет библиотек (libigc/xc_score)

**Поведение:** Fallback на deprecated `parse_igc_extended.py`
**Вывод:** Warning в stderr
**Ограничения:** Нет анализа термов, XC score = distance_km

### 2. Нет барометрических высот

**Поведение:** Fallback на GPS высоту
**Результат:** `altitude_source = "gps"`

### 3. Нет валидных fix (все V)

**Поведение:**
- Для distance: используются все точки
- Для takeoff: используется первая доступная точка

### 4. Отрицательные высоты

**Поведение:** Корректно парсятся (полёты ниже уровня моря)
**Пример:** Dead Sea flights (~-400m)

### 5. Файл поврежден

**Поведение:**
- Ошибка логируется в stderr
- Файл пропускается
- Обработка продолжается

## Зависимости

```bash
# Рекомендуемые
pip install libigc

# Для PostgreSQL
pip install psycopg[binary]

# Для progress bar (встроен в update_flights_metadata.py)
pip install tqdm
```

## Связь с PKL датасетом

Эти скрипты являются первым этапом пайплайна создания `flights_by_cell_day.pkl`:

```
IGC файлы
  ↓
parse_igc_with_libs.py  (извлечение метрик + XC score)
  ↓
update_flights_metadata.py  (обновление БД)
  ↓
build_pkl_dataset.py  (создание PKL структур)
  ↓
flights_by_cell_day.pkl
```

### Использование данных в PKL

```python
# Формат flights_by_cell_day.pkl:
[
    [  # day 0, cell 0
        (
            "2025-02-02 13:51:43",  # takeoff_datetime
            (
                57.02,    # score (xc_score или distance_km как fallback)
                666.0,    # alt (= takeoff_alt)
                1247.0,   # plaf
                38.064,   # lat
                23.375,   # lon
                666.0,    # takeoff_alt
                0.85      # mountainess (из elevation tiles)
            )
        ),
        ...
    ],
    ...
]
```

## Примечания

1. **XC score > distance:** XC score может быть больше distance_km за счёт множителей (до 1.6x)
2. **Incremental по умолчанию:** Обновляются только записи где `xc_score IS NULL`
3. **Многопоточность:** Использует multiprocessing для ускорения обработки
4. **Commit каждые 100:** Данные коммитятся порциями для производительности
5. **UTF-8 с errors='ignore':** Парсинг устойчив к кодировке IGC файлов

## Авторы

Создано на основе спецификации IGC v1.00 (specs/IGC-Spec_v1.00.pdf) и существующего парсера `igc_ingest_skygr.py`.
