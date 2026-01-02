# Исследование кодовой базы Paraglidable

## Цель

Кратко описать структуру репозитория, ключевые подсистемы, потоки данных и
выявленные пробелы/риски, чтобы понимать, что нужно для запуска обучения на
собранных IGC.

## Структура репозитория (верхний уровень)

- `neural_network/` — обучение и прогноз (TensorFlow, обработка GFS/GRIB, модели).
- `scripts/` — утилиты загрузки данных, запусков, ingestion IGC, bridge‑сервер.
- `tiler/` — C++ утилита генерации тайлов прогнозов.
- `www/` — фронтенд (HTML/CSS/JS) и PHP API сайта.
- `extensions/` — браузерное расширение для сбора ссылок (paraplan.ru).
- `docker/` и `docker-compose.yml` — окружение и локальный Postgres.
- `specs/` — документация процессов и развёртывания.

## Ключевые подсистемы

### 1) Ingestion IGC (скрипт)

Файл: `scripts/igc_ingest_skygr.py`

Назначение:
- Обходит списки Leonardo (например sky.gr), собирает `show_flight` ссылки.
- Из каждой страницы извлекает `igc_url`.
- Скачивает IGC, парсит метаданные (дата, пилот, планер, координаты старта,
  длительность, кол-во точек и т.д.).
- Пишет всё в БД (PostgreSQL), отмечает дубликаты по `sha256`.

Ключевые таблицы:
- `flights` — основная таблица (статусы, `igc_url`, метаданные, `sha256`, `updated_at`).
- `crawl_state` / `crawl_state_list` — прогресс по страницам списка.

Жизненный цикл статуса (упрощенно):
`new` -> `queued` -> `downloading` -> `downloaded`/`parsed`
и `failed` при ошибках.

### 2) Bridge‑сервер + расширение

Файл сервера: `scripts/igc_bridge_server.py`  
Расширение: `extensions/paraplan_igc/*`

Назначение:
- Расширение собирает ссылки на `show_flight` и IGC, запрашивает backend.
- Backend хранит весь state в БД; расширение не использует локальную БД.

REST‑эндпоинты:
- `GET /stats?source=...`
- `POST /links`
- `GET /resolve/next`, `POST /resolve`
- `GET /downloads/next`, `POST /downloads`

### 3) Обучение и датасет

Файлы:
- `neural_network/train.py` — основной вход для обучения.
- `neural_network/inc/*.py` — загрузка данных, модели, признаки.
- `neural_network/bin/data/*.pkl` — фактический датасет.
- `scripts/download_data.py` — скачивание датасета из Google Drive.

Фактический датасет сейчас хранится в pkl‑файлах:
- `flights_by_cell_day.pkl`, `meteo_content_by_cell_day.pkl`, `meteo_days.pkl`,
  `sorted_cells_latlon.pkl`, и т.д.

Важно: текущий код обучения **не использует** IGC‑таблицы из Postgres.
Он зависит от готовых pkl‑файлов (исторические данные).

### 4) Прогноз и тайлы

Файлы:
- `neural_network/forecast.py` — скачивание GFS, прогноз, подготовка данных для tiler.
- `tiler/` — C++ генерация PNG тайлов из prediction файлов.
- `www/` — сайт и API для отображения результатов.

Пайплайн прогноза:
GFS -> `forecast.py` -> `predictions.txt` + `tilerArguments.json` -> tiler -> tiles -> сайт.

### 5) Инфраструктура и утилиты

- `docker/Dockerfile` и `docker/python_requirements.txt` — окружение (TF 2.0.0).
- `scripts/start_server.sh`, `scripts/start_jupyter.sh`.
- `scripts/cron_tasks/` — задачи по обновлению прогнозов.

## Потоки данных (схема)

```
IGC sources (sky.gr/paraplan) 
  -> ingestion (igc_ingest_skygr.py / browser extension + bridge server)
  -> Postgres table flights (metadata + igc_url + download status)

Training data (historical)
  -> scripts/download_data.py (Google Drive)
  -> neural_network/bin/data/*.pkl
  -> train.py

Forecast
  -> GFS download (forecast.py)
  -> prediction files
  -> tiler (C++)
  -> www/data/tiles + spots.json
```

## Форматы и ключевые артефакты

- IGC: парсинг `HFDTE`, `HFPLT`, `HFGTY`, `HFGCL/HFCCL`, B‑строки.
- База `flights` в Postgres: метаданные + статусы + `sha256`.
- bin/data (*.pkl): агрегаты по ячейкам/дням, набор параметров GFS.
- Тайлы прогнозов: PNG + `spots.json` в `www/data/tiles/`.

## Точки входа (основные команды)

- Обучение: `python neural_network/train.py`
- Прогноз: `python neural_network/forecast.py`
- Скачивание обучающих pkl: `python scripts/download_data.py`
- IGC ingestion: `python scripts/igc_ingest_skygr.py ...`
- Bridge сервер: `python scripts/igc_bridge_server.py --db-url ...`
