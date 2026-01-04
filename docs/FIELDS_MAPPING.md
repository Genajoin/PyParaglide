# Маппинг полей: xcontest → Paraglidable Training Format

## Важно: IGC файлы недоступны

Для этих данных **нет IGC файлов**, поэтому мы используем данные из xcontest API:
- `score` = `league.route.points`

---

## Требуемый формат (из TRAINING_PROCESS.md)

### flights_by_cell_day.pkl
```python
flights_by_cell_day[cell_index][day_index] = [
    ('yyyy-mm-dd hh:mm:ss', (score, lat, lon)),
    ...
]
```

**Поля записи (CELLS модель):**
| Позиция | Поле | Тип | Описание | Источник xcontest |
|---------|------|-----|----------|-------------------|
| 0 | datetime | str | `'yyyy-mm-dd hh:mm:ss'` | ✅ `pointStart.time` |
| 1 | score | float | Очки/балл полёта | ✅ `league.route.points` |
| 2 | lat | float | Широта точки взлёта | ✅ `takeoff.link` |
| 3 | lon | float | Долгота точки взлёта | ✅ `takeoff.link` |

---

## Наши поля из xcontest API

| Поле xcontest | Маппинг на Paraglidable | Статус |
|---------------|------------------------|--------|
| `pointStart.time` | `datetime` | ✅ |
| `takeoff.link` | `lat`, `lon` | ✅ (парсинг URL) |
| `league.route.points` | `score` | ✅ |
| `league.route.distance` | `route_distance` | ✅ (дополнительно) |
| `league.route.type` | `route_type` | ✅ (дополнительно) |
| `stats.duration` | `duration_sec` | ✅ |
| `pilot.id` | `pilot_id` | ✅ |
| `glider.name` | `glider` | ✅ |
| **mountainess** | `mountainess` | ❌ **НЕТ** (можно вычислить из SRTM) |
| **cell_index** | `cell_index` | ❌ **НЕТ** (нужен sorted_cells.pkl) |
| **day_index** | `day_index` | ❌ **НЕТ** (нужен meteo_days.pkl) |

---

## Что не хватает и как получить

### 1. **mountainess** (гористость 0.0-1.0)
- **Статус:** Не приходит в xcontest API
- **Решение:**
  - Вариант A: Взять из существующего `neural_network/bin/data/mountainess_by_cell_alt.pkl`
  - Вариант B: Вычислить из SRTM данных (разница высот в ячейке)
- **Приоритет:** НИЗКИЙ для CELLS модели

### 2. **cell_index** (индекс ячейки в sorted_cells.pkl)
- **Статус:** Не вычисляется
- **Решение:** Найти индекс ячейки (cell_lat, cell_lon) в `neural_network/bin/data/sorted_cells.pkl`
- **Приоритет:** **ВЫСОКИЙ** - необходим для структуры данных

### 3. **day_index** (индекс дня относительно meteo_days.pkl)
- **Статус:** Не вычисляется
- **Решение:** Сопоставить дату с индексом из `neural_network/bin/data/meteo_days.pkl`
- **Приоритет:** **ВЫСОКИЙ** - необходим для структуры данных

---

## Структура данных для обучения (CELLS модель)

```python
# Формат flights_by_cell_day:
flights_by_cell_day[cell_index][day_index] = [
    ('2023-10-01 10:12:58', (105.2, 46.272567, 13.473567)),
    #                          score   lat         lon
]

# Для наших данных:
{
    'datetime': '2023-10-01 10:12:58',
    'score': 105.2,              # Из league.route.points
    'lat': 46.272567,
    'lon': 13.473567,
    'cell_lat': 46,
    'cell_lon': 13,
    'cell_index': None,          # TODO: из sorted_cells.pkl
    'day_index': None,           # TODO: из meteo_days.pkl
    'mountainess': None,         # TODO: из SRTM
}
```

---

## План действий

### Этап 1: Добавить cell_index и day_index (КРИТИЧНО)
```python
# Загрузить существующие данные
with open('neural_network/bin/data/sorted_cells.pkl', 'rb') as f:
    sorted_cells = pickle.load(f)

with open('neural_network/bin/data/meteo_days.pkl', 'rb') as f:
    meteo_days = pickle.load(f)

# Вычислить индексы
cell_to_index = {cell: i for i, cell in enumerate(sorted_cells)}
day_to_index = {day: i for i, day in enumerate(meteo_days)}

cell_index = cell_to_index.get((cell_lat, cell_lon))
day_date = datetime.strptime(dt_formatted, '%Y-%m-%d %H:%M:%S').date()
day_index = day_to_index.get(day_date)
```

### Этап 2: Создать PKL файлы
```python
# Конвертация JSON в PKL формат для обучения
import pickle

flights_by_cell_day = ...  # структура из наших данных

with open('neural_network/bin/data/flights_by_cell_day.pkl', 'wb') as f:
    pickle.dump(flights_by_cell_day, f)
```

---

## Статистика по данным

Текущий набор данных (xcontest):
- Всего полётов: ~11,759
- Уникальных spots: ~252
- Основной регион: Словения (SI), Северная Италия
- Диапазон дат: 2023-2025 (нужно проверить)

Топ-3 ячейки:
- 46_13 (46°N, 13°E) — 6,393 полёта
- 45_13 (45°N, 13°E) — 3,468 полётов
- 46_14 (46°N, 14°E) — 1,049 полётов

---

## Крупное обновление: Удаление altitude binning (2026-01-03)

### Изменения в модели CELLS

**Удалён altitude binning** - модель больше не разделяет полёты по 5 высотным уровням (600/700/800/900/1000 hPa).

**До (после Issue #16 PR #17):**
```python
(datetime, (score, lat, lon, takeoff_alt, mountainess))
#         [0]    [1][0] [1][1] [1][2] [1][3]      [1][4]
```

**После (текущее):**
```python
(datetime, (score, lat, lon))
#         [0]    [1][0] [1][1] [1][2]
```

### Убраны поля:
- **`takeoff_alt`** - использовался для altitude binning, теперь агрегирован
- **`mountainess`** - перелётное значение, теперь из `mountainess_by_cell_alt.pkl` (усреднённое)

### Модельные изменения:
1. **`nb_altitudes`**: 5 → 1
2. **Формы тензоров:**
   - Входы: `(batch, nb_cells, nb_altitudes, ...)` → `(batch, nb_cells, 1, ...)`
   - Выходы: `(batch, nb_cells, nb_altitudes)` → `(batch, nb_cells, 1)`
3. **Wind:** усредняется по 5 altitude levels
4. **Mountainess:** усредняется по 5 altitude levels

### Файлы изменены:
- `flight_processor.py` - tuple: 7 полей → 3 поля
- `dataset.py` - агрегация без altitude binning
- `trainer.py` - `nb_altitudes = 1`, обновлены формы тензоров
- `model_cells.py` - input/output shapes для nb_altitudes=1
- `layers.py` - удалён altitude tiling во всех блоках
- `forecast.py` - nb_altitudes=1, агрегация в inference

### Влияние на обучение:
- Модель упрощена: 20 выходов (4×5) → 4 выхода
- Потеряна информация о altitude-specific условиях
- Требуется полное переобучение модели

### TODO:
- Оптимизация: mountainess хранится дублированно (per-flight в PKL и per-cell в отдельном файле)
- Можно вычислять mountainess на лету из DEM/lat/lon

