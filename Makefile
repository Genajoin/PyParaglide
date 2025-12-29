# Paraglidable Makefile
# Основные команды для разработки и эксплуатации

.PHONY: help build up down restart logs shell db-shell
.PHONY: cli ingest stats dataset scan
.PHONY: install-deps start-server start-jupyter
.PHONY: clean clean-data clean-igc clean-gfs

# Docker Compose
COMPOSE := docker compose
CONTAINER := paraglidable

# Python в контейнера
PYTHON := docker exec $(CONTAINER) python3
BASH := docker exec $(CONTAINER) bash

# ============================================================================
# Help
# ============================================================================

help:  ## Показать эту справку
	@echo "Paraglidable Makefile"
	@echo ""
	@echo "Docker:"
	@echo "  make build          - Собрать Docker образ"
	@echo "  make up             - Запустить контейнеры"
	@echo "  make down           - Остановить контейнеры"
	@echo "  make restart        - Перезапустить контейнеры"
	@echo "  make logs           - Показать логи"
	@echo "  make shell          - Запустить bash в контейнере"
	@echo "  make db-shell       - Подключиться к PostgreSQL"
	@echo ""
	@echo "CLI команды:"
	@echo "  make cli           - Показать CLI справку"
	@echo "  make ingest        - IGC ingestion"
	@echo "  make stats         - Статистика полётов"
	@echo "  make dataset        - Построить dataset"
	@echo "  make scan          - Сканировать файлы"
	@echo ""
	@echo "Сервер:"
	@echo "  make install-deps  - Установить зависимости (однократно)"
	@echo "  make start-server  - Запустить Apache"
	@echo "  make start-jupyter - Запустить Jupyter"
	@echo ""
	@echo "Очистка:"
	@echo "  make clean         - Удалить временные файлы"
	@echo "  make clean-data    - Удалить данные нейросети"
	@echo "  make clean-igc     - Удалить IGC файлы"
	@echo "  make clean-gfs     - Удалить GFS данные"

# ============================================================================
# Docker команды
# ============================================================================

build:  ## Собрать Docker образ
	$(COMPOSE) build

up:  ## Запустить контейнеры в фоне
	$(COMPOSE) up -d

down:  ## Остановить и удалить контейнеры
	$(COMPOSE) down

restart:  ## Перезапустить контейнеры
	$(COMPOSE) restart

logs:  ## Показать логи всех контейнеров
	$(COMPOSE) logs -f

logs-f:  ## Показать логи Paraglidable контейнера
	$(COMPOSE) logs -f paraglidable

logs-db:  ## Показать логи PostgreSQL
	$(COMPOSE) logs -f postgres

shell:  ## Запустить bash в контейнере
	$(BASH)

db-shell:  ## Подключиться к PostgreSQL
	docker exec -it paraglidable-postgres psql -U paraglidable -d paraglidable

db-backup:  ## Экспорт базы данных
	docker exec paraglidable-postgres pg_dump -U paraglidable paraglidable > backup_$$(date +%Y%m%d_%H%M%S).sql

db-restore:  ## Импорт базы данных (использование: make db-restore FILE=backup.sql)
	docker exec -i paraglidable-postgres psql -U paraglidable paraglidable < $(FILE)

# ============================================================================
# CLI команды
# ============================================================================

cli:  ## Показать CLI справку
	$(PYTHON) scripts/cli.py --help

# --- ingest команды ---
ingest:  ## Справка по ingest командам
	$(PYTHON) scripts/cli.py ingest --help

ingest-list:  ## Собрать IGC ссылки за период (используйте YEARS=2020-2025)
	$(PYTHON) scripts/cli.py ingest list --years "$(YEARS)"

ingest-links:  ## Извлечь IGC URLs (используйте MAX=100)
	$(PYTHON) scripts/cli.py ingest links --max-links $(MAX)

ingest-download:  ## Скачать IGC файлы (используйте MAX=200)
	$(PYTHON) scripts/cli.py ingest download --max-flights $(MAX)

ingest-reparse:  ## Перепарсить файлы (используйте MAX=1000)
	$(PYTHON) scripts/cli.py ingest reparse --max-files $(MAX)

ingest-stats:  ## Статистика ingestion
	$(PYTHON) scripts/cli.py ingest stats

# --- stats команды ---
stats:  ## Справка по stats командам
	$(PYTHON) scripts/cli.py stats --help

stats-monthly:  ## Статистика по месяцам (используйте GROUP=quarter, TOP=10)
	$(PYTHON) scripts/cli.py stats monthly --group-by "$(GROUP)" --top $(TOP)

stats-quarterly:  ## Квартальная статистика (используйте TOP=10)
	$(PYTHON) scripts/cli.py stats quarterly --top $(TOP)

stats-yearly:  ## Годовая статистика (используйте TOP=10)
	$(PYTHON) scripts/cli.py stats yearly --top $(TOP)

# --- dataset команды ---
dataset:  ## Справка по dataset командам
	$(PYTHON) scripts/cli.py dataset --help

dataset-build:  ## Построить PKL dataset (используйте YEARS=2020-2025)
	$(PYTHON) scripts/cli.py dataset build --years "$(YEARS)"

# --- scan команды ---
scan:  ## Справка по scan командам
	$(PYTHON) scripts/cli.py scan --help

scan-paraplan:  ## Сканировать paraplan файлы (используйте DIR=/home/gena/par)
	$(PYTHON) scripts/cli.py scan paraplan --igc-dir "$(DIR)"

# ============================================================================
# Сервер и зависимости
# ============================================================================

install-deps:  ## Установить зависимости (однократно)
	@echo "Скачивание training data..."
	$(PYTHON) scripts/download_data.py
	@echo "Скачивание elevation tiles..."
	$(PYTHON) scripts/download_elevation_tiles.py
	@echo "Скачивание background tiles..."
	$(PYTHON) scripts/download_background_tiles.py
	@echo "Сборка C++ tiler..."
	sh scripts/build_tiler.sh

start-server:  ## Запустить Apache web сервер
	docker exec $(CONTAINER) service apache2 start

start-jupyter:  ## Запустить Jupyter notebook
	docker exec -d $(CONTAINER) bash -c "cd /workspaces/Paraglidable && jupyter notebook --no-browser --ip=0.0.0.0 --port=8888"

# ============================================================================
# Legacy команды (через legacy обёртки)
# ============================================================================

legacy-ingest:  ## Legacy igc_ingest_skygr.py (полная совместимость)
	$(PYTHON) scripts/legacy/igc_ingest_skygr.py

legacy-stats:  ## Legacy flights_monthly_stats.py
	$(PYTHON) scripts/legacy/flights_monthly_stats.py

legacy-bridge:  ## Legacy igc_bridge_server.py
	$(PYTHON) scripts/legacy/igc_bridge_server.py

# ============================================================================
# Очистка
# ============================================================================

clean:  ## Удалить временные файлы Python
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

clean-data:  ## Удалить данные нейросети
	rm -rf neural_network/bin/data/*
	rm -rf neural_network/bin/data_test/*

clean-igc:  ## Удалить скачанные IGC файлы
	rm -rf data/igc/*

clean-gfs:  ## Удалить GFS данные
	rm -rf data/gfs/*

clean-all: clean clean-data clean-igc clean-gfs  ## Полная очистка

# ============================================================================
# Форейcasting
# ============================================================================

forecast:  ## Запустить генерацию прогноза (требует install-deps)
	docker exec $(CONTAINER) bash -c "cd /workspaces/Paraglidable/neural_network && python forecast.py"

# ============================================================================
# Переменные по умолчанию
# ============================================================================

YEARS ?= 2020-2025
GROUP ?= month
TOP ?= 10
MAX ?= 200
DIR ?= /home/gena/par
FILE ?= backup.sql
