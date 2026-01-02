# PyParaglide Architecture

## System Overview

PyParaglide is an AI-based paragliding flyability forecasting system that uses neural networks to predict flying conditions based on weather data.

```
┌─────────────────┐     ┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  GFS Weather    │────▶│  GribReader │────▶│   Dataset    │────▶│    Model    │
│  Data (GRIB)    │     │  (pygrib)   │     │   (PKL)      │     │  (Keras)    │
└─────────────────┘     └─────────────┘     └──────────────┘     └─────────────┘
                                                                  │
                                                                  ▼
                                                           ┌─────────────┐
                                                           │  Forecast   │
                                                           │  (JSON)      │
                                                           └─────────────┘
```

## Module Structure

```
src/pyparaglide/
├── __init__.py              # Package init, version info
├── cli.py                   # Main CLI entry point (Typer app)
│
├── config/                  # Configuration
│   ├── __init__.py
│   └── settings.py          # Pydantic Settings (env vars)
│
├── data/                    # Data loading
│   ├── __init__.py
│   ├── dataset.py           # Dataset class (PKL loader)
│   └── normalization.py    # Normalization functions
│
├── downloads/               # Data download
│   ├── __init__.py
│   └── gfs_downloader.py    # GFS data from NOAA AWS
│
├── inference/               # Forecast generation
│   ├── __init__.py
│   ├── grib_reader.py       # GRIB file reader (pygrib)
│   └── forecast.py          # Forecaster class
│
├── models/                  # Neural network
│   ├── __init__.py
│   ├── enums.py             # ModelType, ProblemFormulation
│   ├── layers.py            # Custom Keras layers
│   ├── model_cells.py       # CELLS model (grid-based)
│   └── model_spots.py       # SPOTS model (spot-based)
│
├── preprocessing/           # Dataset building
│   ├── __init__.py
│   └── dataset_builder.py   # Build PKL from GRIB + flights
│
└── training/                # Training logic
    ├── __init__.py
    ├── callbacks.py         # Training callbacks
    └── trainer.py           # Trainer class
```

## Neural Network Architecture

### Model Types

**CELLS Model** (Grid-based):
- Predicts flyability for 1°×1° grid cells
- Input: `(batch, nb_cells, nb_altitudes, 3, wind_dim)` + other weather data
- Output: 4 predictions × 5 altitudes = 20 total outputs

**SPOTS Model** (Spot-based):
- Predicts flyability per take-off spot
- Spot-specific wind direction weights
- Input: Spot coordinates + weather data
- Output: Per-spot flyability prediction

### Model Blocks

```
┌─────────────────────────────────────────────────────────────────┐
│                        ModelCells                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Inputs:                                                         │
│  ├── date (batch, 1)                                            │
│  ├── dow (batch, 7)  - day of week one-hot                     │
│  ├── mountainess (batch, nb_cells, nb_altitudes)                │
│  ├── other (batch, nb_cells, 3, other_dim)                      │
│  ├── humidity (batch, nb_cells, 3, humidity_dim)                │
│  └── wind (batch, nb_cells, nb_altitudes, 3, wind_dim)          │
│                                                                  │
│  ┌─────────────────┐    ┌──────────────────────────────────┐    │
│  │  WindBlockCells │────▶│  Wind Prediction                │    │
│  └─────────────────┘    │  (batch, nb_cells, nb_altitudes, 3)│    │
│                         └──────────────────────────────────┘    │
│                                       │                         │
│                                       ▼                         │
│  ┌─────────────────────────────────────────────────────┐       │
│  │  _encapsulate_flyability()                           │       │
│  │  - Reshape & tile other/humidity over altitudes    │       │
│  │  - FlyabilityBlock Dense layers                     │       │
│  │  - Output: (batch, nb_cells, nb_altitudes)          │       │
│  └─────────────────────────────────────────────────────┘       │
│                                       │                         │
│                                       ▼                         │
│  ┌─────────────────┐    ┌──────────────────────────────────┐    │
│  │ FlyabilityBlock │────▶│  Flyability Prediction           │    │
│  └─────────────────┘    │  (batch, nb_cells, nb_altitudes)  │    │
│                         └──────────────────────────────────┘    │
│                                       │                         │
│                  ┌────────────────┼────────────────┐           │
│                  ▼                ▼                ▼           │
│  ┌──────────────┐  ┌───────────────┐  ┌─────────────────┐   │
│  │ Crossability │  │   Wind Fly    │  │  Humidity Fly   │   │
│  │    Block     │  │  AbilityBlock │  │  AbilityBlock   │   │
│  └──────┬───────┘  └───────┬───────┘  └────────┬────────┘   │
│         │                  │                    │             │
│         └──────────────────┼────────────────────┘             │
│                            ▼                                  │
│  ┌─────────────────────────────────────────────────────┐      │
│  │  PopulationBlock (4 separate instances)             │      │
│  │  - Expands prediction by super_resolution^2        │      │
│  │  - Applies pilot population model                 │      │
│  │  - Date & day-of-week factors                      │      │
│  └─────────────────────────────────────────────────────┘      │
│                            │                                  │
│                  ┌───────────┼───────────┐                    │
│                  ▼           ▼           ▼                    │
│  Outputs: (4 × nb_cells × super_resolution^2 × nb_altitudes) │
│  ├── flown (overall flight probability)                  │
│  ├── crossed (cross-country potential)                   │
│  ├── wind_flown (wind-based flyability)                  │
│  └── humidity_flown (rain-based flyability)              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Custom Layers

| Layer | Purpose | Input Shape | Output Shape |
|-------|---------|-------------|--------------|
| `WindBlockCells` | Wind terrain adjustment | `(batch, nb_cells, nb_altitudes, 3, wind_dim)` | `(batch, nb_cells, nb_altitudes, 3)` |
| `WindFlyabilityBlock` | Wind-based flyability | `(batch, nb_cells, nb_altitudes, 3)` | `(batch, nb_cells, nb_altitudes)` |
| `HumidityFlyabilityBlock` | Rain-based flyability | `(batch, nb_cells, 3, humidity_dim)` | `(batch, nb_cells, nb_altitudes)` |
| `FlyabilityBlock` | Combined flyability | `(3,)`, `(3*other_dim,)`, `(3*humidity_dim,)` | `(1,)` |
| `CrossabilityBlock` | Cross-country potential | Multiple inputs | `(batch, nb_cells, nb_altitudes)` |
| `PopulationBlock` | Pilot behavior model | `(pred, date, dow)` | `(batch, nb_cells*sr^2, nb_altitudes)` |

## Data Flow

### Training Pipeline

```
┌──────────────────┐
│  xContest Flights│
│  (JSON files)    │
└────────┬─────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  DatasetBuilder.build()                                 │
│  - Parse flight data from JSON                           │
│  - Download GRIB files from GFS                          │
│  - Extract weather parameters                            │
│  - Create cell/spot definitions                          │
│  - Build mountainess data                               │
│  - Save to PKL files                                     │
└────────┬───────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  PKL Files                                               │
│  - meteo_days.pkl                                       │
│  - sorted_cells.pkl                                     │
│  - meteo_params.pkl                                     │
│  - meteo_content_by_cell_day.pkl                        │
│  - flights_by_cell_day.pkl                              │
│  - mountainess_by_cell_alt.pkl                          │
└────────┬───────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  Trainer.prepare_data()                                  │
│  - Load PKL files                                        │
│  - Compute normalization                                │
│  - Prepare input tensors (date, dow, weather)            │
│  - Prepare output tensors (flights by altitude)          │
└────────┬───────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  Model.fit()                                             │
│  - binary_crossentropy loss                             │
│  - Adam optimizer                                       │
│  - Learning rate schedule                                │
│  - Validation split                                     │
└────────┬───────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  Saved Weights                                           │
│  - cells.weights.h5                                     │
│  - normalization_*.pkl                                  │
└────────────────────────────────────────────────────────────┘
```

### Forecast Pipeline

```
┌──────────────────┐
│  Download GFS     │
│  (NOAA AWS S3)   │
└────────┬─────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  GribReader.open()                                       │
│  - Read GRIB2 files                                     │
│  - Extract parameters by name/level                      │
│  - Get lat/lon grids                                     │
└────────┬───────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  Forecaster.predict()                                    │
│  - Load model weights                                    │
│  - Prepare inputs                                        │
│  - Run model prediction                                  │
│  - Apply denormalization                                 │
└────────┬───────────────────────────────────────────────┘
         │
         ▼
┌────────────────────────────────────────────────────────────┐
│  Forecast JSON                                           │
│  - Predictions for each cell/altitude                    │
│  - Date/time info                                        │
│  - Model metadata                                       │
└────────────────────────────────────────────────────────────┘
```

## Altitude Levels

The model predicts for 5 pressure levels (hPa):

| Level | Approx. Altitude | Description |
|-------|-----------------|-------------|
| 600 | ~4200m | High mountain flights |
| 700 | ~3000m | Mountain thermal altitude |
| 800 | ~2000m | Mid-level thermal |
| 900 | ~1000m | Low thermal |
| 1000 | ~100m | Valley/basin |

## Output Naming Convention

The model outputs follow this naming pattern:

```
flown [fufu] [of wind|of rain] <altitude>
```

Examples:
- `flown 1000` — Overall flight probability at 1000hPa
- `flown fufu 800` — Cross-country potential at 800hPa
- `flown of wind 600` — Wind-based flyability at 600hPa
- `flown of rain 900` — Rain-based flyability at 900hPa

## Configuration

All configuration is done via environment variables with the `PYPARAGLIDE_` prefix:

```bash
# Required
PYPARAGLIDE_TRAINING_DATES=2024-06-01:2024-08-31,2025-06-01:2025-08-31
PYPARAGLIDE_BBOX=45,47,13,15
PYPARAGLIDE_GFS_DIR=data/gfs/anl

# Optional (with defaults)
PYPARAGLIDE_PKL_DIR=data/pkl
PYPARAGLIDE_FLIGHTS_DIR=data/flights
PYPARAGLIDE_MODELS_DIR=data/models
PYPARAGLIDE_OUTPUT_DIR=output/forecasts
PYPARAGLIDE_MIN_FLIGHTS_PER_SPOT=200
PYPARAGLIDE_SPOT_CLUSTER_DISTANCE_KM=15.0
```

## Dependencies

### Core
- **tensorflow** >= 2.15 — Neural network framework
- **numpy** >= 1.26 — Numerical computing
- **pygrib** >= 2.1 — GRIB file reading

### Data Processing
- **pandas** >= 2.0 — Data manipulation
- **scipy** >= 1.11 — Scientific computing
- **pyproj** >= 3.5 — Coordinate transformations

### CLI
- **typer** >= 0.12 — CLI framework
- **rich** >= 13.0 — Terminal output
- **tqdm** >= 4.66 — Progress bars
- **python-dotenv** >= 1.0 — Environment variables
- **pydantic** >= 2.0 — Configuration validation
