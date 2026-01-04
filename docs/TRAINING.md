# Training Guide

This guide explains how to train PyParaglide models from scratch.

## Prerequisites

Before training, make sure you have:

1. **Downloaded GFS weather data** for your training dates
2. **Downloaded xContest flight data** for your region
3. **Built the PKL dataset** from GRIB files and flights

See [Quick Start](../README.md#quick-start) for details.

## Model Types

PyParaglide uses the CELLS model:

### CELLS Model
- **Grid-based** flyability prediction (1°×1° cells)
- Trains all cells at once
- More data per model → better generalization
- **Use for**: Regional overview, when you need predictions for entire area

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

### 3. Train CELLS Model

```bash
# Full training (recommended for production)
pyparaglide train --model cells --epochs 100 --batch-size 32

# With early stopping (recommended)
pyparaglide train --model cells --epochs 600 --early-stopping-patience 30

# Quick test
pyparaglide train --model cells --epochs 2 --batch-size 8
```

### 4. Evaluate Model

```bash
# Evaluate on test year
pyparaglide evaluate --year 2023 --model cells

# With custom threshold
pyparaglide evaluate --year 2023 --model cells --threshold 0.3
```

## Validation Strategies

### Choosing Validation Split Method

PyParaglide supports two validation methods:

#### Alternating Days (`--validation-split 0` or default)
```bash
pyparaglide train --model cells --epochs 100
```

**Use when:**
- Working with time-series data
- Want to avoid temporal leakage
- Data has seasonal patterns

**How it works:** Even days (0,2,4...) for validation, odd days (1,3,5...) for training

#### Keras Random Split (`--validation-split 0.2`)
```bash
pyparaglide train --model cells --epochs 200 --validation-split 0.2
```

**Use when:**
- Training with limited data
- Want train/val to have similar distribution
- Don't care about temporal order

**How it works:** Randomly splits 20% of samples for validation

### Recommendations

| Model | Validation Method | Early Stopping |
|-------|-------------------|----------------|
| CELLS | Alternating days (default) | ✅ Recommended (patience=30) |

## Understanding Training Metrics

### Train Loss vs Validation Loss

**Key principle: Validation loss is more important!**

| Scenario | Train Loss | Val Loss | Diagnosis |
|----------|------------|----------|-----------|
| **Ideal** | 0.25 | 0.28 | ✅ Good generalization |
| **Overfitting** | 0.25 | 0.80 | ❌ Model memorized training data |
| **Underfitting** | 0.80 | 0.82 | ⚠️ Model needs more training |

**Rule of thumb:**
- Gap < 0.1: Excellent
- Gap 0.1-0.2: Good
- Gap > 0.3: Overfitting

### When to Use Early Stopping

**✅ Use Early Stopping for CELLS:**
```bash
pyparaglide train --model cells --epochs 600 --early-stopping-patience 30
```
- Plenty of data
- Validation metrics are reliable
- Prevents overfitting

## Recommended Training Configurations

### CELLS Production Training

```bash
# With early stopping (recommended)
pyparaglide train --model cells \
  --epochs 600 \
  --batch-size 32 \
  --early-stopping-patience 30

# Expected results:
# - Train loss: ~0.28
# - Val loss: ~0.28-0.36
# - Stops at epoch 80-120 typically
```

## Model Evaluation

### Evaluating on Test Data

```bash
# Evaluate CELLS
pyparaglide evaluate --year 2023 --model cells

# Output:
# - Confusion Matrix (TP, TN, FP, FN)
# - ROC AUC Score
# - Precision, Recall, F1
```

### Understanding Results

**Good model indicators:**
- ROC AUC > 0.85
- Precision > 0.7 (few false alarms)
- Recall > 0.7 (few missed days)

**Threshold tuning:**
```bash
# Conservative (few false alarms)
pyparaglide evaluate --year 2023 --threshold 0.7

# Balanced
pyparaglide evaluate --year 2023 --threshold 0.5

# Aggressive (catch most flyable days)
pyparaglide evaluate --year 2023 --threshold 0.3
```

## Parameter Tuning Guide

### Learning Rate

```bash
# Default (works well)
--lr-init 0.008 --lr-end 0.0007

# For faster convergence (may be unstable)
--lr-init 0.01 --lr-end 0.001

# For fine-tuning (slower but stable)
--lr-init 0.005 --lr-end 0.0005
```

### Batch Size

```bash
# Default (good balance)
--batch-size 32

# For larger datasets (faster training)
--batch-size 64

# If out of memory
--batch-size 16
```

### Epochs

| Model | Minimum | Recommended | With Early Stopping |
|-------|---------|-------------|---------------------|
| CELLS | 55 | 100-200 | 600 |

## Troubleshooting

### Out of Memory

Reduce batch size:
```bash
pyparaglide train --model cells --epochs 100 --batch-size 16
```

### Poor Convergence

- Ensure you have sufficient training data (multiple seasons)
- Check that flight data quality is good
- Verify bounding box covers your target region
- Try different learning rates

### Validation Loss Too High

**For CELLS:**
- Try alternating days validation (default)
- Increase training epochs
- Check for data quality issues

## Model Outputs

Training saves:
- `cells.weights.h5` — CELLS model weights
- `normalization_*.pkl` — Normalization coefficients

To `data/models/` by default (configurable via `PYPARAGLIDE_MODELS_DIR`).

## Advanced Usage

### Custom Learning Rates

```bash
pyparaglide train --model cells --lr-init 0.01 --lr-end 0.001 --epochs 100
```

### Resume Training

```bash
pyparaglide train --model cells --load-weights --epochs 100
```

### Super-Resolution

For higher spatial resolution (experimental):
```bash
pyparaglide train --model cells --super-res 2 --epochs 100
```

## Quick Reference

```bash
# Complete training pipeline
pyparaglide build-dataset                                    # Build dataset
pyparaglide train --model cells --epochs 600 -p 30          # Train CELLS
pyparaglide evaluate --year 2023 --model cells               # Evaluate
```

## See Also

- [Architecture](ARCHITECTURE.md) — Model architecture details
- [API Reference](API.md) — Complete API documentation
- [Contributing](../CONTRIBUTING.md) — Development guidelines
