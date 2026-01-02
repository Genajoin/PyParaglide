# PyParaglide

<div align="center">

![PyParaglide Logo](www/imgs/logo/logo.svg" width="100")

**AI-based paragliding flyability forecasting with TensorFlow 2.x**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.15+](https://img.shields.io/badge/tensorflow-2.15+-orange.svg)](https://www.tensorflow.org/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-43%20passing-green.svg)](tests/)

**Live Site:** https://paraglidable.com

</div>

---

## Overview

PyParaglide is a modernized fork of [Paraglidable](https://github.com/Genajoin/Paraglidable) — an AI-powered forecasting system that predicts paragliding flying conditions based on weather data.

### Key Features

- **Neural Network Forecasts** — Uses TensorFlow 2.x to predict flyability for 1°×1° grid cells
- **Multiple Outputs** — Predicts overall flyability, cross-country potential, wind-based and rain-based indicators
- **5 Altitude Levels** — Separate predictions for 600hPa, 700hPa, 800hPa, 900hPa, 1000hPa
- **GFS Weather Data** — Uses NOAA GFS Analysis/Forecast data
- **CLI-First Design** — Modern command-line interface with Typer + Rich

### What's New in PyParaglide

- ✅ **TensorFlow 2.15+** — Migrated from TF 1.15
- ✅ **Python 3.12+** — Modern Python with type hints
- ✅ **No Docker Required** — Direct installation via pip
- ✅ **No C++ Tiler** — Pure Python implementation
- ✅ **pyproject.toml** — Standard Python packaging
- ✅ **Unit Tests** — 43 tests with pytest
- ✅ **GPL v3** — Open source license

## Installation

### Requirements

- Python 3.12 or higher
- TensorFlow 2.15+ (automatically installed)
- ~1GB free disk space for training data

### Install from PyPI (Future)

```bash
pip install pyparaglide
```

### Install from Source

```bash
git clone https://github.com/Genajoin/PyParaglide.git
cd PyParaglide
pip install -e .
```

### Development Installation

```bash
git clone https://github.com/Genajoin/PyParaglide.git
cd PyParaglide
pip install -e ".[dev]"
```

## Quick Start

### 1. Configure Environment

```bash
cp .env.example .env
# Edit .env to set your training dates, bbox, and data directories
```

The `.env` file uses this format for date ranges:
```bash
# Multiple ranges supported (comma-separated)
PYPARAGLIDE_TRAINING_DATES=2024-06-01:2024-08-31,2025-06-01:2025-08-31
```

### 2. Download Training Data

```bash
# Use dates from .env (TRAINING_DATES)
pyparaglide download

# Override with specific range
pyparaglide download --dates 2024-06-01:2024-08-31

# Multiple ranges
pyparaglide download --dates "2024-06-01:2024-08-31,2025-06-01:2025-08-31"

# Legacy format (single range)
pyparaglide download --start 2024-06-01 --end 2024-08-31
```

### 3. Build Dataset

```bash
# Use dates from .env
pyparaglide build-dataset

# Override with specific range
pyparaglide build-dataset --dates 2024-06-01:2024-08-31
```

### 4. Train Model

```bash
pyparaglide train --cells 10 --epochs 55
```

### 5. Generate Forecast

```bash
pyparaglide forecast
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `pyparaglide version` | Show version information |
| `pyparaglide config` | Show current configuration |
| `pyparaglide info` | Show system information |
| `pyparaglide download` | Download GFS weather data |
| `pyparaglide build-dataset` | Build PKL dataset from GRIB + flights |
| `pyparaglide analyze` | Analyze flights and weather data |
| `pyparaglide train` | Train neural network model |
| `pyparaglide forecast` | Generate flyability forecast |

### Example: Data Download

```bash
# Use dates from .env (TRAINING_DATES)
pyparaglide download

# Single range
pyparaglide download --dates 2024-06-01:2024-08-31

# Multiple ranges
pyparaglide download --dates "2024-06-01:2024-08-31,2025-06-01:2025-08-31"

# With parallel downloads
pyparaglide download --dates 2024-06-01:2024-08-31 --workers 4 --filter
```

**Date Format Priority:**
1. `--dates` (new unified format, supports multiple ranges)
2. `--start/--end` (legacy format, single range)
3. `.env` TRAINING_DATES (default)

### Example: Training

```bash
# Train with 10 cells for 55 epochs
pyparaglide train --cells 10 --epochs 55 --batch-size 32

# Train with specific learning rate
pyparaglide train --cells 5 --lr-init 0.01 --lr-end 0.001

# Train without validation
pyparaglide train --cells 10 --no-validation
```

### Example: Forecast

```bash
# Generate 10-day forecast (default)
pyparaglide forecast

# Generate with custom output directory
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
5. **Separate Indicators** — Wind-flyability and humidity-flyability

### Outputs (per altitude level)

| Output | Description |
|--------|-------------|
| `flown` | Overall flight probability |
| `flown fufu` | Cross-country potential |
| `flown of wind` | Wind-based flyability |
| `flown of rain` | Humidity/rain-based flyability |

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Code Formatting

```bash
ruff check src/
ruff format src/
black src/
```

### Type Checking

```bash
mypy src/
```

## Migration from Original Paraglidable

| Original | PyParaglide |
|----------|-------------|
| TensorFlow 1.15 | TensorFlow 2.15+ |
| Python 3.6 | Python 3.12+ |
| Docker required | pip install |
| C++ tiler | Pure Python |
| Custom training scripts | Unified CLI |
| `scripts/train.py` | `pyparaglide train` |
| `neural_network/forecast.py` | `pyparaglide forecast` |

## Documentation

- **[Architecture](docs/ARCHITECTURE.md)** — Detailed model architecture
- **[API Reference](docs/API.md)** — Complete API documentation
- **[Training Guide](docs/TRAINING.md)** — How to train models
- **[Contributing](CONTRIBUTING.md)** — Contribution guidelines

## Acknowledgments

- **Original Paraglidable** by Antoine de Mandre — https://github.com/Genajoin/Paraglidable
- **GFS Data** — NOAA/NCEP GFS model
- **xContest** — Paragliding flight data

## License

This project is licensed under GPL v3. The original Paraglidable project was also GPL v3.

---

<div align="center">

**[Original Project](https://github.com/Genajoin/Paraglidable)** • **[Issues](https://github.com/Genajoin/PyParaglide/issues)**

</div>
