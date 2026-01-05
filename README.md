# PyParaglide

<div align="center">

**AI-based paragliding flyability forecasting with TensorFlow 2.x**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15+](https://img.shields.io/badge/tensorflow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-120%20passing-green.svg)](tests/)

</div>

---

## Overview

PyParaglide is a modernized fork of Paraglidable — an AI-powered forecasting system that predicts paragliding flying conditions based on weather data.

### Key Features

- **Neural Network Forecasts** — Uses TensorFlow 2.x to predict flyability for 1°×1° grid cells
- **2 Output Model** — Predicts overall flyability and cross-country potential
- **5 Altitude Levels** — Separate predictions for 600hPa, 700hPa, 800hPa, 900hPa, 1000hPa
- **GFS Weather Data** — Uses NOAA GFS Analysis/Forecast data
- **CLI-First Design** — Modern command-line interface with Typer + Rich
- **Browser Extension** — Collect flight data from xContest with [xcontest_data_collector](extensions/xcontest_data_collector/)

## Installation

### Requirements

- Python 3.12 or higher
- TensorFlow 2.15+ (automatically installed)
- ~1TB free disk space for training data

### Install from Source

```bash
git clone https://github.com/Genajoin/PyParaglide.git
cd PyParaglide
pip install -e .
```

## Quick Start

### 1. Flight Data Collection

PyParaglide uses historical flight data from xContest for training. The recommended way to collect this data is using the browser extension [xcontest_data_collector](extensions/xcontest_data_collector/):

1. **Install the extension** (Chrome/Firefox)
2. **Visit xcontest.org** and navigate to your region
3. **Click extension** to download flight data as JSON
4. **Place files** in `data/flights/` directory

**See [extensions/xcontest_data_collector/README.md](extensions/xcontest_data_collector/README.md)** for installation and usage details.

### 2. Analyze & Determine Parameters

Review flight distribution, determine optimal bbox and training date ranges

```bash
pyparaglide analyze flights
```

### 3. Configure Environment

```bash
cp .env.example .env
```

Edit .env to set your training dates, bbox, and data directories. The `.env` file uses this format for date ranges:
```bash
# Multiple ranges supported (comma-separated)
PYPARAGLIDE_TRAINING_DATES=2024-06-01:2024-08-31,2025-06-01:2025-08-31
```

### 2. Download Training Data

```bash
# Use dates from .env (TRAINING_DATES)
pyparaglide dl analysis
pyparaglide analyze meteo
pyparaglide dl elevation
```

### 3. Build Dataset

Create PKL files from GRIB + flights

```bash
# Use dates from .env
pyparaglide build-dataset
```

### 4. Train Model

CELLS model (grid-based, 1°×1°)

```bash
pyparaglide train
pyparaglide evaluate --year 2025 --threshold 0.1
```

### 5. Generate Forecast

```bash
pyparaglide dl forecast --days 10
pyparaglide forecast
```

### 8. Interpret Results

Output files in `output/forecasts/`. Each day has predictions for different hours.

## CLI Commands

| Command | Description |
|---------|-------------|
| `pyparaglide version` | Show version information |
| `pyparaglide config` | Show current configuration |
| `pyparaglide info` | Show system information |
| `pyparaglide dl analysis` | Download GFS analysis data (historical) |
| `pyparaglide dl forecast` | Download GFS forecast data (predictions) |
| `pyparaglide dl elevation` | Download SRTM elevation data |
| `pyparaglide build-dataset` | Build PKL dataset from GRIB + flights |
| `pyparaglide analyze` | Analyze flights and weather data |
| `pyparaglide train` | Train CELLS neural network model |
| `pyparaglide forecast` | Generate flyability forecast |

### Data Download

#### GFS Analysis Data (Historical Weather)

GFS Analysis data is historical weather data used for **training** models:

```bash
# Use dates from .env (TRAINING_DATES)
pyparaglide dl analysis

# Single range
pyparaglide dl analysis --dates 2024-06-01:2024-08-31

# Multiple ranges
pyparaglide dl analysis --dates "2024-06-01:2024-08-31,2025-06-01:2025-08-31"

# With parallel downloads and filtering
pyparaglide dl analysis --dates 2024-06-01:2024-08-31 --workers 4 --filter
```

**Date Format Priority:**
1. `--dates` (new unified format, supports multiple ranges)
2. `.env` TRAINING_DATES (default)

#### GFS Forecast Data (Predictions)

GFS Forecast data is used for **generating predictions**:

```bash
# Download next 10 days from latest forecast run (default)
pyparaglide dl forecast --days 10

# Download for specific date
pyparaglide dl forecast --date 2026-01-05

# Download for date range
pyparaglide dl forecast --start 2026-01-05 --end 2026-01-15

# Force re-download (overwrite existing files)
pyparaglide dl forecast --days 10 --force
```

**How it works:**
- Automatically finds the latest available GFS forecast run from NOMADS
- Calculates correct forecast offsets (f006, f012, f018, etc.)
- Downloads for 3 hours per day: 06:00, 12:00, 18:00 UTC
- Files are named by **valid time**: `YYYYMMDD-HH.grib2`

#### Elevation Data

```bash
# Download SRTM elevation data (uses BBOX from .env)
pyparaglide dl elevation

# Download with custom bbox
pyparaglide dl elevation --bbox 45,47,13,15
```

### Example: Training

```bash
# Train CELLS model (all cells at once)
pyparaglide train

# Train with specific learning rate
pyparaglide train --lr-init 0.01 --lr-end 0.001

# Train without validation
pyparaglide train --no-validation
```

### Example: Forecast

```bash
# Generate forecast (requires GFS forecast data in data/gfs/forecasts/)
pyparaglide forecast

# Generate for specific date
pyparaglide forecast --date 2026-01-05
pyparaglide forecast --output-dir /path/to/output

# Show debug information
pyparaglide forecast --verbose
```

### Example: Analyze Data

```bash
# Analyze flight distribution (shows clusters and recommends optimal bbox)
pyparaglide analyze flights

# Analyze with bbox filter and min flights threshold
pyparaglide analyze flights --bbox 45,47,13,16 --min-flights 50

# Analyze GFS data completeness
pyparaglide analyze meteo

# Check specific date ranges
pyparaglide analyze meteo --dates 2024-06-01:2024-08-31

# Build with auto-analysis
pyparaglide build-dataset --analyze
```

**Note:** Analysis results are recommendations based on your data. You may have intentionally chosen different settings.

## Configuration

PyParaglide uses environment variables with the `PYPARAGLIDE_` prefix:

```bash
# Training configuration
PYPARAGLIDE_TRAINING_DATES=2024-06-01:2024-08-31,2025-06-01:2025-08-31
PYPARAGLIDE_BBOX=45,47,13,15  # lat_min,lat_max,lon_min,lon_max

# Data directories
PYPARAGLIDE_GFS_DIR=data/gfs/anl
PYPARAGLIDE_FLIGHTS_DIR=data/flights
PYPARAGLIDE_MODELS_DIR=data/models
PYPARAGLIDE_PKL_DIR=data/pkl
PYPARAGLIDE_OUTPUT_DIR=output/forecasts

# Processing
PYPARAGLIDE_MIN_FLIGHTS_PER_SPOT=200
PYPARAGLIDE_SPOT_CLUSTER_DISTANCE_KM=15.0
```

## Project Structure

```
PyParaglide/
├── src/pyparaglide/
│   ├── __init__.py
│   ├── cli.py              # Main CLI entry point
│   ├── config/             # Configuration (Settings)
│   ├── data/               # Dataset loading and normalization
│   ├── downloads/          # GFS data downloader
│   ├── inference/          # GRIB reader, forecast generation
│   ├── models/             # Neural network models
│   ├── preprocessing/      # Dataset building
│   └── training/           # Training logic
└── tests/                  # Unit tests (43 tests)
```

## Model Architecture

PyParaglide uses a custom neural network with:

1. **Wind Processing Block** — Adjusts wind for mountainous terrain
2. **Flyability Block** — Combines wind, weather, and humidity data
3. **Crossability Block** — Predicts cross-country potential
4. **Population Block** — Models pilot behavior and probability

### Outputs (2 outputs, per altitude level)

| Output | Description |
|--------|-------------|
| `flown` | Overall flight probability |
| `crossed` | Cross-country potential |

**Note:** Altitude levels are aggregated (nb_altitudes=1). The model uses a CELLS-only architecture (grid-based, 1°×1°).

## Documentation

- **[Architecture](docs/ARCHITECTURE.md)** — Detailed model architecture
- **[API Reference](docs/API.md)** — Complete API documentation
- **[Training Guide](docs/TRAINING.md)** — How to train models
- **[Contributing](CONTRIBUTING.md)** — Contribution guidelines

## Data Sources

### Weather Data (GFS)

PyParaglide uses NOAA GFS (Global Forecast System) weather data from two sources:

#### GFS Analysis (Historical - for Training)

GFS Analysis is **historical weather data** used for **training** models:

| Source | Years | Access | Download Command |
|--------|-------|--------|------------------|
| AWS S3 (NOAA Open Data) | 2021+ | Public, free | `pyparaglide dl analysis` |
| NCAR RDA | 2000-2021 | Requires free registration | Manual download required |

**Usage:** Train models with historical flight data
```bash
pyparaglide dl analysis --dates 2024-06-01:2024-08-31
```

#### GFS Forecast (Predictions - for Forecasting)

GFS Forecast is **future weather prediction** used for **generating forecasts**:

| Source | Latency | Access | Download Command |
|--------|---------|--------|------------------|
| NOMADS API | ~6 hours | Public, free | `pyparaglide dl forecast` |

**Usage:** Generate flyability predictions for upcoming days
```bash
# Download next 10 days from latest GFS run
pyparaglide dl forecast --days 10
```

**How it works:**
- NOMADS provides GFS forecast runs 4 times daily (00, 06, 12, 18 UTC)
- Each run contains predictions up to 16 days ahead (f000 to f384)
- PyParaglide automatically finds the latest run and calculates correct offsets
- Downloads 3 forecast hours per day: 06:00, 12:00, 18:00 UTC

### Elevation Data (SRTM)

PyParaglide supports global elevation data via SRTM:

- **SRTM3** (90m resolution): Default, from CGIAR-CSI

**Coverage:** ±60° latitude (covers most inhabited regions)

**Supported Regions:**
- Alps: `PYPARAGLIDE_BBOX=45,47,6,10`
- Patagonia: `PYPARAGLIDE_BBOX=-55,-40,-75,-65`
- Himalayas: `PYPARAGLIDE_BBOX=27,30,85,88`
- New Zealand: `PYPARAGLIDE_BBOX=-47,-34,166,179`

Elevation data downloads during `pyparaglide dl eleavation`.


## Acknowledgments

- **Original [Paraglidable](https://github.com/AntoineMeler/Paraglidable)** by Antoine de Mandre
- **GFS Data** — NOAA/NCEP GFS model
- **xContest** — Paragliding flight data
- **CGIAR-CSI** — SRTM elevation data

## License

This project is licensed under GPL v3. The original Paraglidable project was also GPL v3.
