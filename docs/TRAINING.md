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

# Bounding box following GeoJSON RFC 7946 (Alps example: lon_min,lat_min,lon_max,lat_max)
PYPARAGLIDE_BBOX=13,45,15,47

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
pyparaglide train --epochs 100 --batch-size 16
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
pyparaglide train --lr-init 0.01 --lr-end 0.001 --epochs 100
```

### Resume Training

```bash
pyparaglide train --load-weights --epochs 100
```

### Super-Resolution

For higher spatial resolution (experimental):
```bash
pyparaglide train --super-res 2 --epochs 100
```

## Quick Reference

```bash
# Complete training pipeline
pyparaglide build-dataset                     # Build dataset
pyparaglide train --epochs 600 -p 30          # Train CELLS
pyparaglide evaluate --year 2023              # Evaluate
```

## Experiment Workflow

This section describes the recommended workflow for developing model improvements through experiments.

### Overview

The experiment workflow combines:
- **Git branches** — Isolate changes per experiment
- **Experiments tracking** — Save metrics and configurations
- **A/B comparison** — Compare against baseline

### Workflow Steps

#### 1. Establish Baseline

Before starting experiments, establish a baseline:

```bash
# Train baseline model
source .venv/bin/activate && pyparaglide train \
  --epochs 100 \
  --experiment baseline_v1 \
  --notes "Baseline CELLS model with default config"

# Verify baseline
source .venv/bin/activate && pyparaglide experiments --show baseline_v1
```

#### 2. Create Experiment Branch

Create a new branch for each experiment:

```bash
# Name convention: feat/exp-, exp-, impro-
git checkout -b feat/exp-add-attention-layer
# or
git checkout -b exp/learning-rate-schedule
```

#### 3. Make Changes

Implement your improvement:
- Modify model architecture (`src/pyparaglide/models/`)
- Change training parameters (`src/pyparaglide/training/`)
- Add new features or preprocessing

#### 4. Train Experiment

Train with experiment tracking:

```bash
source .venv/bin/activate && pyparaglide train \
  --epochs 100 \
  --experiment attention_v1 \
  --notes "Added self-attention layer after FlyabilityBlock"
```

This saves:
- Weights to `data/models/experiments/attention_v1/cells.weights.h5`
- Metrics to `data/models/experiments/attention_v1/metrics.json`
- Config to `data/models/experiments/attention_v1/config.json`
- Git branch and commit are automatically recorded

#### 5. Evaluate Experiment

```bash
# Run evaluation on test set
source .venv/bin/activate && pyparaglide evaluate \
  --year 2023 \
  --model-path data/models/experiments/attention_v1/cells.weights.h5
```

#### 6. Compare with Baseline

```bash
# Compare experiments
source .venv/bin/activate && pyparaglide experiments --compare baseline_v1,attention_v1

# Or list all experiments
source .venv/bin/activate && pyparaglide experiments --list
```

Output shows:
```
baseline_v1 vs attention_v1
┌──────────────┬──────────┬────────────┬──────────┬─────────┐
│ Metric       │ Baseline │ Experiment │ Delta    │ Status  │
├──────────────┼──────────┼────────────┼──────────┼─────────┤
│ roc_auc      │ 0.8234   │ 0.8456     │ ↑0.0222  │ Better  │
│ precision    │ 0.7123   │ 0.7234     │ ↑0.0111  │ Better  │
│ recall       │ 0.7456   │ 0.7567     │ ↑0.0111  │ Better  │
│ f1           │ 0.7286   │ 0.7398     │ ↑0.0112  │ Better  │
│ val_loss     │ 0.3234   │ 0.3123     │ ↓0.0111  │ Better  │
└──────────────┴──────────┴────────────┴──────────┴─────────┘

Winner: EXPERIMENT
```

#### 7a. Successful Experiment → Merge

If experiment shows improvement:

```bash
# Rebase to keep clean history
git checkout master
git pull
git checkout feat/exp-add-attention-layer
git rebase master

# Merge with --no-ff to preserve experiment history
git checkout master
git merge --no-ff feat/exp-add-attention-layer

# Optional: Tag the merge
git tag -a exp/attention-v1-merged -m "Merged attention layer experiment"

# Push
git push origin master
git push origin exp/attention-v1-merged

# Delete feature branch
git branch -d feat/exp-add-attention-layer
```

#### 7b. Failed Experiment → Archive

If experiment shows no improvement or regression:

```bash
# Save the experiment data for reference
# (already saved in data/models/experiments/)

# Switch back to master
git checkout master

# Delete experiment branch
git branch -d feat/exp-add-attention-layer

# Document findings (optional)
# Add to EXPERIMENTS.md or create experiment notes
```

### Best Practices

#### Naming Conventions

| Element | Convention | Example |
|---------|------------|---------|
| Git branch | `feat/exp-*`, `exp/*`, `impro/*` | `feat/exp-attention-layer` |
| Experiment name | Descriptive with version | `attention_v1`, `lr_schedule_v2` |
| Notes | Concise description | "Added attention layer" |

#### Metrics for Comparison

Primary metrics for deciding success:
- **ROC AUC** — Overall ranking quality (most important)
- **Val Loss** — Generalization gap
- **F1 Score** — Balance of precision/recall

Minimum improvement thresholds:
- ROC AUC: +0.01 (1%)
- Val Loss: -0.02
- F1: +0.01

#### Multiple Iterations

For iterative experiments:

```bash
# v1
git checkout -b feat/exp-attention-v1
pyparaglide train --experiment attention_v1

# Iterate (still in same branch)
# Modify code...
pyparaglide train --experiment attention_v2

# If v2 successful
git commit -am "v2: improved attention mechanism"

# Only merge the final version
git checkout master && git merge --no-ff feat/exp-attention-v1
```

#### Documentation

Keep experiment notes:

```bash
# Save with detailed notes
pyparaglide train \
  --experiment attention_v1 \
  --notes "Added self-attention (4 heads). Key changes: attention position after FlyabilityBlock, dropout 0.1."

# View notes later
pyparaglide experiments --show attention_v1
```

### Experiment Directory Structure

```
data/models/experiments/
├── baseline_v1/
│   ├── cells.weights.h5
│   ├── metrics.json        # All metrics + git info
│   └── config.json         # Model config only
├── attention_v1/
│   ├── cells.weights.h5
│   ├── metrics.json
│   └── config.json
└── lr_schedule_v2/
    ├── cells.weights.h5
    ├── metrics.json
    └── config.json
```

### Common Experiment Patterns

#### 1. Architecture Changes

```bash
git checkout -b feat/exp-add-residual-connections
# Edit model architecture
pyparaglide train --experiment residual_v1 --epochs 150
pyparaglide experiments --compare baseline_v1,residual_v1
```

#### 2. Hyperparameter Tuning

```bash
git checkout -b exp/learning-rate-tuning
# Modify learning rate in code or via config
pyparaglide train --experiment lr_high_v1 --epochs 100 --lr-init 0.02
pyparaglide train --experiment lr_low_v1 --epochs 100 --lr-init 0.002
pyparaglide experiments --compare baseline_v1,lr_high_v1
```

#### 3. Data Augmentation

```bash
git checkout -b feat/exp-data-augmentation
# Add augmentation in DatasetBuilder
pyparaglide train --experiment aug_v1 --epochs 100
pyparaglide evaluate --year 2023 --model-path data/models/experiments/aug_v1/cells.weights.h5
```

## See Also

- [Architecture](ARCHITECTURE.md) — Model architecture details
- [API Reference](API.md) — Complete API documentation
- [Contributing](../CONTRIBUTING.md) — Development guidelines
