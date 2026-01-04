# Changelog

All notable changes to PyParaglide will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.0] - 2026-01-04

### Breaking Changes
- **Removed redundant indicator outputs** from CELLS model
  - `wind_flown` and `humidity_flown` outputs removed
  - CELLS model now has 2 outputs: `flown` and `crossed` (previously 4)
  - **Existing model checkpoints are incompatible** - retraining required

### Removed
- `WindFlyabilityBlock` layer (redundant - `FlyabilityBlock` already considers all factors)
- `HumidityFlyabilityBlock` layer (redundant - `FlyabilityBlock` already considers all factors)
- CLI `--output wind_flown` and `--output humidity_flown` options

### Changed
- `ModelCells.output_names()` returns `['flown', 'crossed']`
- `Dataset.get_flights_by_altitude()` returns 2 arrays instead of 4
- Forecaster JSON output no longer includes `wind_flyability` and `humidity_flyability` fields
- Training callbacks now track 2 losses instead of 4
- Updated documentation to reflect 2-output architecture

### Benefits
- Simpler model architecture
- Reduced memory usage (2 outputs vs 4)
- Faster training/inference (fewer computations)
- Cleaner API

## [2.0.0] - 2025-01-01

### Added
- **Complete modernization from original Paraglidable**
- TensorFlow 2.15+ support (migrated from TF 1.15)
- Python 3.12+ support with type hints
- Modern CLI with Typer + Rich
- pyproject.toml packaging standard
- 43 unit tests with pytest
- Separate population blocks for each output type
- Multi-range date support in TRAINING_DATES
- PYPARAGLIDE_ prefixed environment variables

### Changed
- **Removed Docker dependency** — runs directly with pip install
- **Removed C++ tiler** — pure Python implementation
- Refactored code into src/pyparaglide/ structure
- Updated all Keras layers to TF2 API patterns
- Replaced tf.keras.backend.* with tf.* equivalents
- Standardized on Python packaging conventions

### Fixed
- WindFlyabilityBlock reshape issue (batch size mismatch)
- PopulationBlock tile operation (day_factor_vector)
- Dataset.get_date() returning wrong shape
- GribReader RuntimeError with pygrib
- Data shape issues in trainer._prepare_outputs()

### Removed
- Old Docker-based deployment (moved to original repo)
- C++ Qt tiler (use original repo for map tiles)
- Deprecated IGC parsing scripts
- Custom BinObj dependency (now using pickle)

---

## [1.0.0] - Original Paraglidable

### Features
- TensorFlow 1.15 neural network
- Docker deployment
- C++ Qt tiler for map generation
- Web interface with Leaflet.js
- Training scripts
- GFS weather data processing
- xContest flight data integration

See [Original Paraglidable](https://github.com/Genajoin/Paraglidable) for full history.
