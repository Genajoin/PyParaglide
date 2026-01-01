# Training Guide

This guide explains how to train PyParaglide models from scratch.

## Prerequisites

Before training, make sure you have:

1. **Downloaded GFS weather data** for your training dates
2. **Downloaded xContest flight data** for your region
3. **Built the PKL dataset** from GRIB files and flights

See [Quick Start](../README.md#quick-start) for details.

## Training Pipeline

### 1. Configure Environment

Set your training parameters in `.env`:

```bash
# Training dates (multiple seasons recommended)
PYPARAGLIDE_TRAINING_DATES=2024-06-01:2024-08-31,2025-06-01:2025-08-31

# Bounding box (Alps example)
PYPARAGLIDE_BBOX=45,47,13,15

# Data directories
PYPARAGLIDE_PKL_DIR=data/pkl
PYPARAGLIDE_MODELS_DIR=data/models
```

### 2. Build Dataset

```bash
# Use dates from .env
pyparaglide build-dataset

# Or specify date range
pyparaglide build-dataset --dates 2024-06-01:2024-08-31
```

This creates PKL files in `data/pkl/`:
- `meteo_days.pkl`
- `sorted_cells.pkl`
- `meteo_params.pkl`
- `meteo_content_by_cell_day.pkl`
- `flights_by_cell_day.pkl`
- `mountainess_by_cell_alt.pkl`

### 3. Train Model

```bash
# Full training (recommended)
pyparaglide train --cells 10 --epochs 55 --batch-size 32

# Quick test (1 cell, 2 epochs)
pyparaglide train --cells 1 --epochs 2 --batch-size 8
```

**Parameters:**
- `--cells` — Number of grid cells to train (use `10` for full Alps, `1` for testing)
- `--epochs` — Number of training epochs (`55` is optimal)
- `--batch-size` — Batch size (`32` is default)
- `--lr-init` — Initial learning rate (default: `0.008`)
- `--lr-end` — Final learning rate (default: `0.0007`)
- `--no-validation` — Skip validation split (not recommended)

### 4. Model Outputs

Training saves:
- `cells.weights.h5` — Model weights
- `normalization_*.pkl` — Normalization coefficients

To `data/models/` by default (configurable via `PYPARAGLIDE_MODELS_DIR`).

## Training Tips

### Recommended Training Configuration

```bash
# Full Alps training (55 epochs)
pyparaglide train --cells 10 --epochs 55 --batch-size 32
```

### Learning Rate Schedule

The model uses exponential decay from `0.008` to `0.0007` over 55 epochs. This schedule is optimized for the CELLS model.

### Number of Cells

- **Testing**: Use `--cells 1` for quick experiments
- **Production**: Use `--cells 10` (or more) for full coverage
- The Alps region is divided into ~10 cells at 1°×1° resolution

### Monitoring Training

Watch for these indicators:
- **Loss** should decrease steadily
- **Validation loss** should track training loss (small gap = good generalization)
- Training typically takes 10-30 minutes depending on hardware

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
pyparaglide train --cells 10 --epochs 55 --batch-size 16
```

### Poor Convergence

- Ensure you have sufficient training data (multiple seasons)
- Check that flight data quality is good
- Verify bounding box covers your target region
- Try different random seed by re-shuffling data

### Model Not Saving

Check that output directory is writable:
```bash
ls -la data/models/
```

Or specify a different output directory:
```bash
pyparaglide train --models-dir /path/to/models
```

## Advanced Usage

### Custom Learning Rates

```bash
pyparaglide train --cells 10 --lr-init 0.01 --lr-end 0.001 --epochs 55
```

### Resume Training

```bash
pyparaglide train --cells 10 --load-weights --epochs 55
```

### Super-Resolution

For higher spatial resolution (experimental):
```bash
pyparaglide train --cells 10 --super-res 2 --epochs 55
```

## See Also

- [Architecture](ARCHITECTURE.md) — Model architecture details
- [API Reference](API.md) — Complete API documentation
- [Contributing](../CONTRIBUTING.md) — Development guidelines
