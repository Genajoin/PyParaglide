"""
Custom EarlyStopping callback for PyParaglide.

Monitors validation loss and stops training when no improvement is seen.
"""

import numpy as np
import tensorflow as tf


class EarlyStopping(tf.keras.callbacks.Callback):
    """
    Stop training when a monitored metric has stopped improving.

    Args:
        monitor: Quantity to be monitored (default: 'val_loss')
        patience: Number of epochs with no improvement after which training will be stopped (default: 10)
        min_delta: Minimum change to qualify as an improvement (default: 0.001)
        mode: One of {'auto', 'min', 'max'}. In 'min' mode, training will stop when the quantity monitored has stopped decreasing (default: 'auto')
        restore_best_weights: Whether to restore model weights from the epoch with the best value of the monitored quantity (default: True)
        verbose: Verbosity mode, 0 or 1 (default: 1)
    """

    def __init__(
        self,
        monitor: str = "val_loss",
        patience: int = 10,
        min_delta: float = 0.0,
        mode: str = "auto",
        restore_best_weights: bool = True,
        verbose: int = 1,
    ):
        super().__init__()

        self.monitor = monitor
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose

        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None

        # Determine mode FIRST
        if mode == "auto":
            if "loss" in monitor or "error" in monitor:
                self.mode = "min"
            else:
                self.mode = "max"

        # Initialize best AFTER mode is determined
        self.best = np.Inf if self.mode == "min" else -np.Inf

    def on_train_begin(self, logs=None):
        """Initialize best weights at start of training."""
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        """Check if we should stop training."""
        logs = logs or {}
        current = self.get_monitor_value(logs)

        if current is None:
            return

        # Check if improvement
        if self.is_improvement(current):
            if self.verbose > 0:
                print(
                    f"\033[92m[EarlyStopping]\033[0m Epoch {epoch}: "
                    f"{self.monitor} improved from {self.best:.6f} to {current:.6f}"
                )
            self.best = current
            self.wait = 0
            # Save best weights
            if self.restore_best_weights:
                self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.verbose > 0 and self.wait % 5 == 0:  # Log every 5 epochs of waiting
                print(
                    f"\033[93m[EarlyStopping]\033[0m Epoch {epoch}: "
                    f"No improvement for {self.wait} epochs (best: {self.best:.6f}, current: {current:.6f})"
                )

        # Check if we should stop
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            if self.verbose > 0:
                print(f"\n\033[93m{'='*60}\033[0m")
                print(f"\033[93m[EarlyStopping] Stopping at epoch {epoch}\033[0m")
                print(f"  Best {self.monitor}: {self.best:.6f} at epoch {epoch - self.patience}")
                print(f"  Current {self.monitor}: {current:.6f}")
                print(f"  Patience: {self.patience} epochs without improvement")
                print(f"\033[93m{'='*60}\033[0m\n")

            self.model.stop_training = True

    def on_train_end(self, logs=None):
        """Restore best weights if needed."""
        if self.stopped_epoch > 0 and self.restore_best_weights and self.best_weights is not None:
            if self.verbose > 0:
                print(f"\033[92m[EarlyStopping] Restoring best weights from epoch {self.stopped_epoch - self.patience}\033[0m")
            self.model.set_weights(self.best_weights)

    def get_monitor_value(self, logs):
        """Get the value of the monitored metric."""
        if self.monitor not in logs:
            return None
        return logs[self.monitor]

    def is_improvement(self, current):
        """Check if current value is an improvement."""
        if self.mode == "min":
            return current < self.best - self.min_delta
        else:
            return current > self.best + self.min_delta
