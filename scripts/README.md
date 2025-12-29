# Scripts

## Unified CLI (новый интерфейс)

Начиная с 2025 года, все основные скрипты объединены в единый CLI интерфейс на базе Typer.

### Быстрый старт

```bash
# Показать все команды
python cli.py --help

# IGC ingestion from sky.gr
python cli.py ingest list --years 2020-2025
python cli.py ingest download --max-flights 100
python cli.py ingest stats

# Flight statistics
python cli.py stats monthly --group-by quarter --top 5
python cli.py stats yearly

# Dataset building
python cli.py dataset build --years 2020-2025

# Scan local files
python cli.py scan paraplan
```

### Подкоманды ingest (IGC из sky.gr)

| Команда | Описание |
|---------|----------|
| `list` | Собрать ссылки на IGC файлы за период |
| `links` | Извлечь IGC URL со страниц полётов |
| `download` | Скачать IGC файлы |
| `reparse` | Перепарсить уже скачанные файлы |
| `stats` | Статистика ingestion |

```bash
# Примеры использования
python cli.py ingest list --years 2020-2025 --max-pages 5
python cli.py ingest links --max-links 100
python cli.py ingest download --max-flights 200 --min-delay 1.0 --max-delay 1.5
python cli.py ingest reparse --max-files 1000
```

### Подкоманды stats (статистика)

| Команда | Описание |
|---------|----------|
| `monthly` | Статистика по периодам (месяц/квартал/год) |
| `quarterly` | Квартальная статистика |
| `yearly` | Годовая статистика |

```bash
# По месяцам с топ-10 bbox
python cli.py stats monthly --source skygr --top 10

# По кварталам
python cli.py stats quarterly --source skygr --top 5

# По годам
python cli.py stats yearly --source skygr
```

### Подкоманды dataset (PKL датасеты)

```bash
# Построить training dataset
python cli.py dataset build --years 2020-2025 --source skygr
```

### Подкоманды scan (локальные файлы)

```bash
# Сканировать локальные paraplan файлы
python cli.py scan paraplan --igc-dir /path/to/igc
```

### Legacy скрипты (обратная совместимость)

Старые скрипты доступны в `legacy/` для обратной совместимости:

```bash
# Используйте legacy-версии для:
# - bridge: требует Python 3.7+ (ThreadingHTTPServer)
# - metadata: требует Python 3.9+ (xc_score type hints)

python legacy/igc_ingest_skygr.py --list-only --years 2020-2025
python legacy/igc_bridge_server.py
python legacy/update_flights_metadata.py
python legacy/scan_paraplan_files.py
```

---

## Краткая справка по lib/ модулям

Общие модули, используемые всеми скриптами:

| Модуль | Описание |
|--------|----------|
| `lib/db.py` | Db класс, connect_db(), PG_SCHEMA |
| `lib/config.py` | Config класс с загрузкой .env |
| `lib/cli.py` | Typer app, общие опции для CLI |
| `lib/igc_parser.py` | parse_igc(), parse_igc_date() |

Пример использования в собственных скриптах:

```python
from lib.db import Db, connect_db, ensure_db
from lib.config import Config, get_default_db_url
from lib.igc_parser import parse_igc

# Получить конфигурацию из .env
cfg = Config()
db = connect_db(cfg.db_url)

# Распарсить IGC файл
meta = parse_igc("/path/to/flight.igc")
```

---

## Оригинальная документация

## To be executed once

```bash
python download_data.py             # Download training weather and flights data (200MB)
python download_elevation_tiles.py  # Download elevation data (260MB)
python download_background_tiles.py # Download background tiles (facultative) (180MB)
sh build_tiler.sh                   # Build the C++ tiler

python download_GFS.py # Optional, download the source .grib weather data files from GFS
```

## To be executed once per session

```bash
sh start_server.sh  # Start Apache server
sh start_jupyter.sh # Start Jupyter server for the neural network documentation
```

## To be executed if needed

```bash
sh update_nn_README.sh # If you have modified the neural network documentation
```

## IGC ingestion (sky.gr) - Legacy interface

```bash
# Postgres only; set --db-url or IGC_DB_URL
export IGC_DB_URL="postgresql://paraglidable:paraglidable@localhost:5432/paraglidable"

# Collect IGC links for a year range (resumable per year)
python legacy/igc_ingest_skygr.py --links-only --years 2010-2025 --continue --max-pages 0 --max-links 0

# Download IGC files slowly (resumes automatically)
python legacy/igc_ingest_skygr.py --download-only --max-flights 0 --min-delay 2 --max-delay 3

# Reparse metadata for already downloaded files
python legacy/igc_ingest_skygr.py --reparse-only --max-reparse 0
```

## IGC bridge server (browser -> DB)

```bash
# Требуется Python 3.7+ для ThreadingHTTPServer
python legacy/igc_bridge_server.py --db-url postgresql://paraglidable:paraglidable@localhost:5432/paraglidable
```

## Browser extension (paraplan.ru)

```bash
# Start the local backend
python legacy/igc_bridge_server.py --db-url postgresql://paraglidable:paraglidable@localhost:5432/paraglidable
```

Then load the extension from `extensions/paraplan_igc` and configure the server
URL in the popup. See `extensions/README.md` for the full usage flow.
