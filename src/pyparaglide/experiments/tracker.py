"""
Experiment tracker for saving and loading experiment metrics.

This module provides functionality to:
- Save experiment results with metrics and configuration
- Load experiment metrics for comparison
- Manage experiment directory structure
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import git
from pydantic import BaseModel, Field


@dataclasses.dataclass
class ExperimentConfig:
    """Configuration for an experiment."""

    model: str = "cells"
    dropout: float | None = None
    layers: str | None = None
    learning_rate: float | None = None
    batch_size: int | None = None
    epochs: int | None = None
    optimizer: str | None = None
    loss: str | None = None


@dataclasses.dataclass
class TestMetrics:
    """Test metrics for an experiment."""

    roc_auc: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    threshold: float = 0.5
    accuracy: float | None = None
    confusion_matrix: dict[str, int] | None = None


class ExperimentMetrics(BaseModel):
    """Metrics and metadata for an experiment."""

    experiment_name: str = Field(description="Name of the experiment")
    date: str = Field(description="Date of the experiment (YYYY-MM-DD)")
    git_branch: str | None = Field(default=None, description="Git branch name")
    git_commit: str | None = Field(default=None, description="Git commit hash")
    config: dict[str, Any] = Field(default_factory=dict, description="Model configuration")
    train_loss: float | None = Field(default=None, description="Final training loss")
    val_loss: float | None = Field(default=None, description="Final validation loss")
    test_metrics: dict[str, Any] = Field(default_factory=dict, description="Test metrics")
    training_time_seconds: float | None = Field(default=None, description="Training time in seconds")
    epochs_trained: int | None = Field(default=None, description="Number of epochs trained")
    notes: str | None = Field(default=None, description="Additional notes about the experiment")

    class Config:
        """Pydantic config."""

        json_encoders = {Path: str}


class ExperimentTracker:
    """Tracker for experiment metrics."""

    def __init__(self, experiments_dir: str | Path | None = None) -> None:
        """
        Initialize the experiment tracker.

        Args:
            experiments_dir: Directory to store experiments. Defaults to data/models/experiments/
        """
        if experiments_dir is None:
            # Default path relative to project root
            experiments_dir = Path(__file__).parent.parent.parent.parent / "data" / "models" / "experiments"
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)

    def _get_git_info(self) -> tuple[str | None, str | None]:
        """Get current git branch and commit hash."""
        try:
            repo = git.Repo(Path(__file__).parent.parent.parent.parent, search_parent_directories=True)
            branch = repo.active_branch.name
            commit = repo.head.commit.hexsha
            return branch, commit
        except Exception:
            return None, None

    def get_experiment_dir(self, experiment_name: str) -> Path:
        """Get the directory for a specific experiment."""
        return self.experiments_dir / experiment_name

    def save_experiment(
        self,
        experiment_name: str,
        metrics: ExperimentMetrics,
        weights_path: str | Path | None = None,
    ) -> Path:
        """
        Save experiment metrics and optionally copy weights.

        Args:
            experiment_name: Name of the experiment
            metrics: ExperimentMetrics object with all data
            weights_path: Optional path to weights file to copy

        Returns:
            Path to the created experiment directory
        """
        exp_dir = self.get_experiment_dir(experiment_name)
        exp_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics.json
        metrics_path = exp_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics.model_dump(), f, indent=2)

        # Save config.json separately for convenience
        config_path = exp_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(metrics.config, f, indent=2)

        # Copy weights if provided
        if weights_path:
            import shutil

            weights_src = Path(weights_path)
            if weights_src.exists():
                weights_dst = exp_dir / "cells.weights.h5"
                shutil.copy2(weights_src, weights_dst)

        return exp_dir

    def load_metrics(self, experiment_name: str) -> dict[str, Any]:
        """
        Load metrics for an experiment.

        Args:
            experiment_name: Name of the experiment

        Returns:
            Dictionary with experiment metrics
        """
        metrics_path = self.get_experiment_dir(experiment_name) / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Experiment metrics not found: {metrics_path}")

        with open(metrics_path) as f:
            return json.load(f)

    def list_experiments(self) -> list[str]:
        """List all available experiment names."""
        if not self.experiments_dir.exists():
            return []
        return [d.name for d in self.experiments_dir.iterdir() if d.is_dir()]

    def experiment_exists(self, experiment_name: str) -> bool:
        """Check if an experiment directory exists."""
        return (self.get_experiment_dir(experiment_name) / "metrics.json").exists()


def save_experiment_metrics(
    experiment_name: str,
    config: dict[str, Any],
    train_loss: float | None = None,
    val_loss: float | None = None,
    test_metrics: dict[str, Any] | None = None,
    training_time_seconds: float | None = None,
    epochs_trained: int | None = None,
    weights_path: str | Path | None = None,
    notes: str | None = None,
    experiments_dir: str | Path | None = None,
) -> Path:
    """
    Convenience function to save experiment metrics.

    Args:
        experiment_name: Name of the experiment
        config: Model configuration dictionary
        train_loss: Final training loss
        val_loss: Final validation loss
        test_metrics: Test metrics dictionary
        training_time_seconds: Training time in seconds
        epochs_trained: Number of epochs trained
        weights_path: Path to weights file
        notes: Additional notes
        experiments_dir: Custom experiments directory

    Returns:
        Path to the created experiment directory
    """
    from datetime import datetime

    tracker = ExperimentTracker(experiments_dir)
    branch, commit = tracker._get_git_info()

    metrics = ExperimentMetrics(
        experiment_name=experiment_name,
        date=datetime.now().strftime("%Y-%m-%d"),
        git_branch=branch,
        git_commit=commit,
        config=config,
        train_loss=train_loss,
        val_loss=val_loss,
        test_metrics=test_metrics or {},
        training_time_seconds=training_time_seconds,
        epochs_trained=epochs_trained,
        notes=notes,
    )

    return tracker.save_experiment(experiment_name, metrics, weights_path)
