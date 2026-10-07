"""Metric recorders shared across phases.

Phase 1 uses :class:`GradientHistory` to log per-layer gradient norms each
step — the core evidence for the vanishing gradient experiment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class EpochHistory:
    """Train/validation loss and accuracy per epoch."""

    train_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)

    def record(self, train_loss: float, train_acc: float, val_loss: float, val_acc: float) -> None:
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)


@dataclass
class GradientHistory:
    """Per-step, per-layer gradient norms.

    grad_norms is a list of np.ndarray — one entry per recorded step, holding
    the L2 norm of dW for each weight layer in forward order.
    """

    grad_norms: list[np.ndarray] = field(default_factory=list)

    def record(self, grad_norms: np.ndarray) -> None:
        self.grad_norms.append(np.asarray(grad_norms, dtype=np.float64))

    def mean_per_layer(self) -> np.ndarray:
        """Mean gradient norm per layer across all recorded steps."""
        return np.mean(np.stack(self.grad_norms), axis=0)

    def to_dict(self) -> dict[str, list[list[float]]]:
        return {"grad_norms": [row.tolist() for row in self.grad_norms]}


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    """Classification accuracy from raw scores (batch, classes)."""
    predictions = np.argmax(logits, axis=1)
    return float(np.mean(predictions == labels))