"""
Experiment comparison utilities.

This module provides functionality to compare metrics between different experiments.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from pyparaglide.experiments.tracker import ExperimentTracker


class ExperimentComparator:
    """Comparator for experiment metrics."""

    def __init__(self, experiments_dir: str | Path | None = None) -> None:
        """
        Initialize the comparator.

        Args:
            experiments_dir: Directory containing experiments
        """
        self.tracker = ExperimentTracker(experiments_dir)
        self.console = Console()

    def _get_metric_value(self, metrics: dict[str, Any], metric_name: str) -> float:
        """Get a metric value from metrics dict, handling nested structures."""
        # Check in test_metrics first
        if "test_metrics" in metrics and metric_name in metrics["test_metrics"]:
            value = metrics["test_metrics"][metric_name]
            if value is not None:
                return float(value)

        # Check at top level
        if metric_name in metrics and metrics[metric_name] is not None:
            return float(metrics[metric_name])

        return 0.0

    def compare(
        self,
        baseline: str,
        experiment: str,
        metrics: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Compare two experiments.

        Args:
            baseline: Baseline experiment name
            experiment: Experiment name to compare against baseline
            metrics: List of metrics to compare (default: standard metrics)

        Returns:
            Dictionary with comparison results
        """
        baseline_metrics = self.tracker.load_metrics(baseline)
        experiment_metrics = self.tracker.load_metrics(experiment)

        if metrics is None:
            metrics = ["roc_auc", "precision", "recall", "f1", "val_loss"]

        comparison: dict[str, Any] = {
            "baseline": baseline,
            "experiment": experiment,
            "baseline_name": baseline_metrics.get("experiment_name", baseline),
            "experiment_name": experiment_metrics.get("experiment_name", experiment),
            "metrics": {},
        }

        for metric in metrics:
            b_val = self._get_metric_value(baseline_metrics, metric)
            e_val = self._get_metric_value(experiment_metrics, metric)
            delta = e_val - b_val

            # Determine sign (for loss metrics, lower is better)
            if metric in ["val_loss", "train_loss"]:
                sign = "↓" if delta < 0 else "↑" if delta > 0 else "="
                better = delta < 0
            else:
                sign = "↑" if delta > 0 else "↓" if delta < 0 else "="
                better = delta > 0

            comparison["metrics"][metric] = {
                "baseline": b_val,
                "experiment": e_val,
                "delta": delta,
                "sign": sign,
                "better": better,
            }

        # Determine winner (based on ROC AUC)
        baseline_auc = self._get_metric_value(baseline_metrics, "roc_auc")
        experiment_auc = self._get_metric_value(experiment_metrics, "roc_auc")

        if experiment_auc > baseline_auc:
            comparison["winner"] = "EXPERIMENT"
        elif baseline_auc > experiment_auc:
            comparison["winner"] = "BASELINE"
        else:
            comparison["winner"] = "TIE"

        return comparison

    def print_comparison(self, comparison: dict[str, Any]) -> None:
        """Print comparison results to console."""
        self.console.print()
        self.console.rule(
            f"[bold cyan]{comparison['baseline_name']} vs {comparison['experiment_name']}[/bold cyan]",
            style="cyan",
        )

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="dim")
        table.add_column("Baseline")
        table.add_column("Experiment")
        table.add_column("Delta")
        table.add_column("Status")

        for metric, values in comparison["metrics"].items():
            status = "[green]Better[/green]" if values["better"] else ("[red]Worse[/red]" if values["sign"] != "=" else "[dim]Equal[/dim]")
            table.add_row(
                metric,
                f"{values['baseline']:.4f}",
                f"{values['experiment']:.4f}",
                f"{values['sign']}{abs(values['delta']):.4f}",
                status,
            )

        self.console.print(table)

        winner_style = "green" if comparison["winner"] == "EXPERIMENT" else ("yellow" if comparison["winner"] == "BASELINE" else "dim")
        self.console.print(f"\n[bold {winner_style}]Winner: {comparison['winner']}[/bold {winner_style}]")
        self.console.print()

    def compare_all(
        self,
        baseline: str,
        metrics: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Compare baseline against all other experiments.

        Args:
            baseline: Baseline experiment name
            metrics: List of metrics to compare

        Returns:
            List of comparison results, sorted by ROC AUC improvement
        """
        experiments = self.tracker.list_experiments()
        comparisons = []

        for exp in experiments:
            if exp == baseline:
                continue
            try:
                comp = self.compare(baseline, exp, metrics)
                comparisons.append(comp)
            except FileNotFoundError:
                continue

        # Sort by ROC AUC delta
        comparisons.sort(key=lambda x: x["metrics"].get("roc_auc", {}).get("delta", 0), reverse=True)
        return comparisons


def compare_experiments(
    baseline: str,
    experiment: str,
    experiments_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Convenience function to compare two experiments.

    Args:
        baseline: Baseline experiment name
        experiment: Experiment name to compare
        experiments_dir: Custom experiments directory

    Returns:
        Dictionary with comparison results
    """
    comparator = ExperimentComparator(experiments_dir)
    comparison = comparator.compare(baseline, experiment)
    comparator.print_comparison(comparison)
    return comparison
