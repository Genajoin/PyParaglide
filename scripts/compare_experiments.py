#!/usr/bin/env python3
"""
Compare PyParaglide training experiments.

Usage:
    python scripts/compare_experiments.py baseline_v1 resnet_v2
    python scripts/compare_experiments.py baseline_v1 resnet_v2 --experiments-dir data/models/experiments
"""

import argparse
import json
import sys
from pathlib import Path


def load_metrics(experiment_path: str | Path) -> dict:
    """Load experiment metrics from JSON file."""
    exp_path = Path(experiment_path)
    metrics_file = exp_path / "metrics.json"

    if not metrics_file.exists():
        # Try as experiment name in default dir
        default_dir = Path(__file__).parent.parent / "data" / "models" / "experiments"
        metrics_file = default_dir / exp_path.name / "metrics.json"

    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found for: {experiment_path}")

    with open(metrics_file) as f:
        return json.load(f)


def get_metric_value(metrics: dict, metric_name: str) -> float:
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


def compare_experiments(
    baseline_path: str,
    experiment_path: str,
    metrics: list[str] | None = None,
) -> dict:
    """
    Compare two experiments.

    Args:
        baseline_path: Path or name of baseline experiment
        experiment_path: Path or name of experiment to compare
        metrics: List of metrics to compare

    Returns:
        Dictionary with comparison results
    """
    if metrics is None:
        metrics = ["roc_auc", "precision", "recall", "f1", "val_loss", "train_loss"]

    baseline = load_metrics(baseline_path)
    experiment = load_metrics(experiment_path)

    comparison = {
        "baseline": baseline.get("experiment_name", baseline_path),
        "experiment": experiment.get("experiment_name", experiment_path),
        "baseline_date": baseline.get("date", "N/A"),
        "experiment_date": experiment.get("date", "N/A"),
        "metrics": {},
    }

    for metric in metrics:
        b_val = get_metric_value(baseline, metric)
        e_val = get_metric_value(experiment, metric)
        delta = e_val - b_val

        # For loss metrics, lower is better
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

    # Determine winner based on ROC AUC
    baseline_auc = get_metric_value(baseline, "roc_auc")
    experiment_auc = get_metric_value(experiment, "roc_auc")

    if experiment_auc > baseline_auc:
        comparison["winner"] = "EXPERIMENT"
    elif baseline_auc > experiment_auc:
        comparison["winner"] = "BASELINE"
    else:
        comparison["winner"] = "TIE"

    return comparison


def print_comparison(comparison: dict) -> None:
    """Print comparison results."""
    print("\n" + "=" * 60)
    print(f"Comparison: {comparison['baseline']} vs {comparison['experiment']}")
    print("=" * 60)
    print(f"Baseline date: {comparison['baseline_date']}")
    print(f"Experiment date: {comparison['experiment_date']}")
    print()

    # Print metrics table
    print(f"{'Metric':<15} {'Baseline':>12} {'Experiment':>12} {'Delta':>12}")
    print("-" * 60)

    for metric, values in comparison["metrics"].items():
        print(f"{metric:<15} {values['baseline']:>12.4f} {values['experiment']:>12.4f} {values['sign']}{abs(values['delta']):>11.4f}")

    print("=" * 60)

    winner = comparison["winner"]
    if winner == "EXPERIMENT":
        print(f"Winner: [EXPERIMENT] - Model shows improvement")
    elif winner == "BASELINE":
        print(f"Winner: [BASELINE] - Experiment did not improve")
    else:
        print(f"Winner: [TIE] - Models perform similarly")
    print("=" * 60 + "\n")

    # Print config diff
    baseline_cfg = load_metrics(comparison["baseline"]).get("config", {})
    experiment_cfg = load_metrics(comparison["experiment"]).get("config", {})

    config_diff = []
    for key in set(list(baseline_cfg.keys()) + list(experiment_cfg.keys())):
        b_val = baseline_cfg.get(key, "-")
        e_val = experiment_cfg.get(key, "-")
        if b_val != e_val:
            config_diff.append((key, b_val, e_val))

    if config_diff:
        print("Configuration differences:")
        print("-" * 60)
        for key, b_val, e_val in config_diff:
            print(f"  {key:<20}: {b_val} -> {e_val}")
        print()


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Compare PyParaglide training experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s baseline_v1 resnet_v2
  %(prog)s baseline_v1 resnet_v2 --experiments-dir data/models/experiments
  %(prog)s baseline_v1 resnet_v2 --metrics roc_auc precision recall
        """,
    )
    parser.add_argument("baseline", help="Baseline experiment name or path")
    parser.add_argument("experiment", help="Experiment to compare against baseline")
    parser.add_argument(
        "--experiments-dir",
        default=None,
        help="Custom experiments directory (default: data/models/experiments)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Metrics to compare (default: roc_auc precision recall f1 val_loss train_loss)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of human-readable format",
    )

    args = parser.parse_args()

    try:
        comparison = compare_experiments(args.baseline, args.experiment, args.metrics)

        if args.json:
            import json

            print(json.dumps(comparison, indent=2))
        else:
            print_comparison(comparison)

        return 0

    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
