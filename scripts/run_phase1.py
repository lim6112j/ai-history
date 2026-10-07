#!/usr/bin/env python3
"""Phase 1 experiments: vanishing gradient in deep sigmoid MLPs.

Run:  uv run python scripts/run_phase1.py

Experiment A — depth 3/5/10 × Sigmoid: early-layer gradient norms collapse.
Experiment B — same depth, Sigmoid vs ReLU(He): convergence speed.
Experiment C — 2-layer baseline accuracy (~88% expected on Fashion-MNIST).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_history.data import INPUT_DIM, NUM_CLASSES, load_fashion_mnist_numpy
from dl_history.metrics import EpochHistory, accuracy
from dl_history.models.mlp_numpy import MLP
from dl_history.viz import plot_accuracy_curves, plot_gradient_norms, plot_loss_curves

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.1
HIDDEN_WIDTH = 128
SEED = 0


def split_batches(n: int, batch_size: int):
    starts = range(0, n, batch_size)
    return [(start, min(start + batch_size, n)) for start in starts]


def train_mlp(
    model: MLP,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int,
    lr: float,
) -> EpochHistory:
    """Train with mini-batch SGD, recording per-epoch metrics."""
    history = EpochHistory()
    batches = split_batches(x_train.shape[0], BATCH_SIZE)
    for _ in range(epochs):
        epoch_loss = 0.0
        for start, end in batches:
            epoch_loss += model.train_batch(x_train[start:end], y_train[start:end], lr)
        train_preds = model.predict(x_train[:10_000])
        val_preds = model.predict(x_val)
        history.record(
            train_loss=epoch_loss / len(batches),
            train_acc=float(np.mean(train_preds == y_train[:10_000])),
            val_loss=float("nan"),  # tracked via val_acc; loss needs forward pass
            val_acc=float(np.mean(val_preds == y_val)),
        )
    return history


def experiment_a(x_train, y_train, x_test, y_test):
    """Deep sigmoid nets: gradient norms per layer, per depth."""
    histories: dict[str, EpochHistory] = {}
    grad_histories: dict = {}
    layer_labels: dict = {}
    for depth_name, hidden in [("3-layer", [HIDDEN_WIDTH, HIDDEN_WIDTH]), ("5-layer", [HIDDEN_WIDTH] * 4), ("10-layer", [HIDDEN_WIDTH] * 9)]:
        model = MLP.build(INPUT_DIM, hidden, NUM_CLASSES, activation="sigmoid", seed=SEED)
        print(f"  training {depth_name} sigmoid ...", flush=True)
        histories[depth_name] = train_mlp(model, x_train, y_train, x_test, y_test, EPOCHS, LEARNING_RATE)
        grad_histories[depth_name] = model.gradient_history
        layer_labels[depth_name] = model.layer_names()
    return histories, grad_histories, layer_labels


def experiment_b(x_train, y_train, x_test, y_test):
    """Same 10-layer shape: sigmoid/xavier vs relu/he."""
    histories: dict[str, EpochHistory] = {}
    for label, activation in [("sigmoid", "sigmoid"), ("relu", "relu")]:
        model = MLP.build(INPUT_DIM, [HIDDEN_WIDTH] * 9, NUM_CLASSES, activation=activation, seed=SEED)
        print(f"  training 10-layer {label} ...", flush=True)
        histories[label] = train_mlp(model, x_train, y_train, x_test, y_test, EPOCHS, LEARNING_RATE)
    return histories


def experiment_c(x_train, y_train, x_test, y_test):
    """2-layer baseline."""
    model = MLP.build(INPUT_DIM, [HIDDEN_WIDTH], NUM_CLASSES, activation="relu", seed=SEED)
    print("  training 2-layer baseline ...", flush=True)
    return train_mlp(model, x_train, y_train, x_test, y_test, EPOCHS, LEARNING_RATE)


def main() -> None:
    print("loading Fashion-MNIST ...", flush=True)
    x_train, y_train, x_test, y_test = load_fashion_mnist_numpy()
    # Standardize with train stats: zero-centered inputs keep deep ReLU nets
    # stable (raw [0,1] inputs diverge at practical learning rates).
    mean, std = x_train.mean(), x_train.std()
    x_train = (x_train - mean) / std
    x_test = (x_test - mean) / std
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Experiment A: depth × sigmoid", flush=True)
    hist_a, grad_histories, layer_labels = experiment_a(x_train, y_train, x_test, y_test)
    plot_loss_curves(hist_a, "Exp A — depth × Sigmoid (loss)", "phase1_expA_loss.png")
    plot_gradient_norms(
        grad_histories, layer_labels, "Exp A — mean per-layer gradient norms (Sigmoid)", "phase1_expA_gradnorm.png"
    )

    print("Experiment B: sigmoid vs relu (10 layers)", flush=True)
    hist_b = experiment_b(x_train, y_train, x_test, y_test)
    plot_loss_curves(hist_b, "Exp B — 10-layer Sigmoid vs ReLU (loss)", "phase1_expB_loss.png")
    plot_accuracy_curves(hist_b, "Exp B — 10-layer Sigmoid vs ReLU (val acc)", "phase1_expB_acc.png")

    print("Experiment C: 2-layer baseline", flush=True)
    hist_c = experiment_c(x_train, y_train, x_test, y_test)
    plot_accuracy_curves({"2-layer baseline": hist_c}, "Exp C — 2-layer baseline (val acc)", "phase1_expC_acc.png")

    summary = {
        "expA_final_val_acc": {name: hist.val_acc[-1] for name, hist in hist_a.items()},
        "expA_grad_norm_layer1_vs_last": {
            name: {
                "layer1": float(grad_histories[name].mean_per_layer()[0]),
                "last": float(grad_histories[name].mean_per_layer()[-1]),
            }
            for name in hist_a
        },
        "expB_final_val_acc": {name: hist.val_acc[-1] for name, hist in hist_b.items()},
        "expC_final_val_acc": hist_c.val_acc[-1],
    }
    (RESULTS_DIR / "phase1_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"figures written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()