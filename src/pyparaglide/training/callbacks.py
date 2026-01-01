"""
Training callbacks for PyParaglide.
"""

import math
from pathlib import Path

import numpy as np
import tensorflow as tf

from pyparaglide.models.enums import ModelType


class TrainingLogger(tf.keras.callbacks.Callback):
    """
    Custom callback for logging training progress with colored output.

    Tracks losses, learning rate, and validation metrics.
    """

    def __init__(self, model_type: ModelType, log_file: Path | str | None = None):
        super().__init__()
        self.iteration = 0
        self.model_type = model_type
        self.log_file = Path(log_file) if log_file else None

    @staticmethod
    def _str_comp(val: float, val_ref: float) -> str:
        """Format value comparison with color."""
        if np.isnan(val) or np.isnan(val_ref) or val_ref == 0:
            return "\033[93m  NaN\033[0m"

        ratio = val / val_ref
        if ratio <= 1.0:
            color = "\033[92m"  # Green
            sign = "-"
        else:
            color = "\033[91m"  # Red
            sign = "+"

        return color + f"{sign}{abs(int(round((ratio - 1.0) * 100))):>4}%\033[0m"

    @staticmethod
    def _str_val(val: float, val_ref: float | None = None) -> str:
        """Format a value with optional comparison."""
        str_v = f"\033[93m{val:.4f}\033[0m"
        if val_ref is None:
            return str_v
        return str_v + " " + TrainingLogger._str_comp(val, val_ref)

    @staticmethod
    def _str_arr(arr: np.ndarray, arr_ref: np.ndarray | None = None) -> str:
        """Format array with optional comparison."""
        result = ""
        for i, v in enumerate(arr):
            if i > 0:
                result += ", "
            str_v = f"\033[94m{v:.3f}\033[0m"
            if arr_ref is None:
                result += str_v
            else:
                result += str_v + " " + TrainingLogger._str_comp(v, arr_ref[i])
        return result

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        """Called at the end of each epoch."""
        logs = logs or {}
        str_it = f"{self.iteration:4d} "

        # Get current learning rate
        lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        str_lr = f" lr: {lr:.2e}"

        if self.model_type == ModelType.CELLS:
            losses = ["population_block_loss", "population_block_1_loss", "population_block_2_loss", "population_block_3_loss"]

            str_training = "loss: " + self._str_val(logs["loss"]) + " (" + self._str_arr(np.array([logs[l] for l in losses])) + ")"

            if "val_loss" in logs:
                str_validation = (
                    "val_loss: "
                    + self._str_val(logs["val_loss"], logs["loss"])
                    + " ("
                    + self._str_arr(np.array([logs[f"val_{l}"] for l in losses]), np.array([logs[l] for l in losses]))
                    + ")"
                )
                print(str_it + str_training + " " + str_validation + str_lr)
            else:
                print(str_it + str_training + str_lr)

        else:  # SPOTS
            str_training = "loss: " + self._str_val(logs["loss"])
            print(str_it + str_training + str_lr)

        # Write to log file
        if self.log_file:
            val_loss_str = f" {logs['val_loss']:.8f}" if "val_loss" in logs else ""
            with open(self.log_file, "a") as f:
                f.write(f"{self.iteration} {logs['loss']:.8f}{val_loss_str} {lr:.4e}\n")

        self.iteration += 1


class LearningRateScheduler:
    """
    Learning rate scheduler with exponential decay.

    Creates a Keras callback that adjusts learning rate during training.
    """

    @staticmethod
    def create(lr_init: float = 0.01, lr_end: float = 1e-6, nb_epochs: int = 100) -> tf.keras.callbacks.LearningRateScheduler:
        """
        Create learning rate scheduler callback.

        Args:
            lr_init: Initial learning rate
            lr_end: Final learning rate
            nb_epochs: Total number of epochs

        Returns:
            Keras callback for learning rate scheduling
        """

        def schedule(epoch: int) -> float:
            s = (math.log(lr_init) - math.log(lr_end)) / math.log(10.0)
            lr = lr_init * pow(10.0, -float(epoch) / float(max(nb_epochs - 1, 1)) * s)
            return lr

        return tf.keras.callbacks.LearningRateScheduler(schedule)
