# Руководство по развёртыванию Paraglidable

Этот документ описывает пошаговый процесс развёртывания проекта Paraglidable.

## Предварительные требования

- Установленный Docker на хостовой системе
- Минимум 100ГБ свободного дискового пространства (для GFS данных)
- Порт 8001 доступен на хосте

## Рабочая инфраструктура

**Docker Compose (рекомендуется):**

```bash
# Запуск контейнера
docker compose up -d

# Остановка
docker compose stop
```

**Порты:**
- `http://localhost:8001` — веб-интерфейс
- `http://localhost:8888` — Jupyter (после `sh scripts/start_jupyter.sh` внутри)

## Подготовка данных для обучения

### Шаг 1: Сбор данных полётов (xContest)

```bash
cd extensions/xcontest_data_collector/
pip install -r requirements.txt
python main.py --collect-flights
```

Создаёт файлы в `data/flights/`:
- `xcontest_flights_YYYY-MM-DD-sl-XX.json`

### Шаг 2: Скачивание GFS данных

```bash
# Лето 2021
python3 scripts/download_GFS.py \
  --start-date 2021-06-01 \
  --end-date 2021-08-31 \
  --data-dir data/gfs/anl \
  --hours 6,12,18 \
  --filter

# Повторить для 2022, 2023, 2024, 2025
```

**Расчет размера:** ~70 GB за сезон с фильтром

### Шаг 3: Настройка `.env`

```bash
# Копируем пример
cp .env.example .env

# Редактируем .env
nano .env
```

**Ключевые настройки:**
```bash
TRAINING_BBOX=45,47,13,15           # 4 ячейки (Slovenia)
TRAINING_DATES=2021-06-01:2021-08-31,2022-06-01:2022-08-31,...
GFS_DIR=data/gfs/anl
PKL_DIR=neural_network/bin/data
```

### Шаг 4: Генерация PKL файлов

```bash
python3 scripts/build_pkl_dataset.py --skip-flights
```

**Создаёт:**
- `sorted_cells_latlon.pkl` — ячейки по bbox
- `sorted_cells.pkl` — индексы в GRIB сетке
- `meteo_days.pkl` — дни (фильтруется по `TRAINING_DATES`)
- `meteo_content_by_cell_day.pkl` — матрица погодных данных
- `mountainess_by_cell_alt.pkl` — гористость

### Шаг 5: Извлечение данных обучения

```bash
python3 scripts/extract_training_data.py --cluster-distance 15
```

**Создаёт:**
- `data/flights/merged/training_flights.json`
- `data/flights/merged/spots.json`
- `data/flights/merged/stats.json`

---

## Сборка C++ Tiler

```bash
cd scripts/
sh build_tiler.sh
```

**Результат:** `tiler/Tiler/Tiler`

---

## Генерация прогноза

```bash
cd neural_network/
python forecast.py
```

**Что делает:**
1. Скачивает GFS данные на ближайшие 10 дней
2. Запускает ML-прогноз через TensorFlow
3. Генерирует PNG тайлы через C++ Tiler

**Время выполнения:** 10-20 минут

---

## Проверка

```bash
# Проверка PKL файлов
python3 -c "
import pickle
cells = pickle.load(open('neural_network/bin/data/sorted_cells_latlon.pkl', 'rb'))
days = pickle.load(open('neural_network/bin/data/meteo_days.pkl', 'rb'))
print(f'Ячеек: {len(cells)}')
print(f'Период: {days[0]} - {days[-1]}')
print(f'Дней: {len(days)}')
"

# Проверка тайлов
find www/data/tiles/ -name '*.png' | wc -l

# Проверка Apache
curl -s http://localhost:8001/ | head -10
```

---

## Источники данных

### Данные полётов

| Источник | Статус | Описание |
|----------|--------|----------|
| **xContest API** | ✅ Работает | `extensions/xcontest_data_collector/` |
| IGC scraping | ⚠️ Deprecated | Не поддерживается |

### Метео-данные (GFS)

| Источник | Период | Статус | Примечания |
|----------|--------|--------|------------|
| **AWS S3** | 2021+ | ✅ Работает | `scripts/download_GFS.py` |
| NCAR RDA | 2000+ | ❓ Не протестирован | Требует регистрацию |

---

## Troubleshooting

### Проблема: Нет GFS данных для некоторых дней

**Решение:** Скрипт `build_pkl_dataset.py` покажет недостающие дни и предложит команды для скачивания.

### Проблема: mountess = None

**Решение:** Убедитесь что `tiler/_cache/elevation/` содержит данные высот.

### Проблема: Ячейки не совпадают с bbox

**Решение:** Проверьте `TRAINING_BBOX` в `.env` — используется для создания `sorted_cells_latlon.pkl`.
