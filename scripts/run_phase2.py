#!/usr/bin/env python3
"""Phase 2 experiments: LeNet-5 convolution vs flat MLP on Fashion-MNIST.

Run:  uv run python scripts/run_phase2.py

Experiment A — LeNet-5 with Sigmoid / Tanh / ReLU: what ReLU buys over the
    1998-era squashing functions (convergence speed + final accuracy).
Experiment B — LeNet-5 vs the flat Phase-1-shaped MLP: far fewer parameters,
    better accuracy — the payoff of preserving spatial structure.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_history.data import load_fashion_mnist_loaders
from dl_history.metrics import EpochHistory, GradientHistory
from dl_history.models.lenet5 import LeNet5
from dl_history.models.mlp_torch import FlatMLP
from dl_history.train import train_model
from dl_history.viz import plot_accuracy_curves, plot_gradient_norms, plot_loss_curves

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

BATCH_SIZE = 128
EPOCHS = 10
LEARNING_RATE = 0.05
MOMENTUM = 0.9
SEED = 0


def run(model, label: str, loaders, grad_history: GradientHistory | None = None):
    print(f"  training {label} ...", flush=True)
    torch.manual_seed(SEED)
    history, seconds_per_epoch = train_model(
        model, loaders, epochs=EPOCHS, lr=LEARNING_RATE, momentum=MOMENTUM,
        grad_history=grad_history,
    )
    print(
        f"    final val acc {history.val_acc[-1]:.4f}, "
        f"{seconds_per_epoch:.1f}s/epoch, "
        f"{sum(p.numel() for p in model.parameters()):,} params",
        flush=True,
    )
    return history, seconds_per_epoch


def main() -> None:
    print("loading Fashion-MNIST ...", flush=True)
    loaders = load_fashion_mnist_loaders(batch_size=BATCH_SIZE)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Experiment A: LeNet-5 activation comparison", flush=True)
    histories_a: dict[str, EpochHistory] = {}
    grad_histories: dict[str, GradientHistory] = {}
    layer_labels: dict[str, list[str]] = {}
    epoch_times: dict[str, float] = {}
    for activation in ["sigmoid", "tanh", "relu"]:
        label = f"LeNet-5 ({activation})"
        grad_histories[label] = GradientHistory()
        histories_a[label], epoch_times[label] = run(
            LeNet5(activation=activation), label, loaders, grad_histories[label]
        )
        layer_labels[label] = LeNet5(activation=activation).weight_layer_names()

    print("Experiment B: LeNet-5 vs flat MLP", flush=True)
    histories_b: dict[str, EpochHistory] = {}
    param_counts: dict[str, int] = {}
    for model, label in [
        (LeNet5(activation="relu"), "LeNet-5"),
        (FlatMLP(activation="relu"), "Flat MLP (784-128-10)"),
    ]:
        param_counts[label] = sum(p.numel() for p in model.parameters())
        histories_b[label], epoch_times[label] = run(model, label, loaders)

    plot_loss_curves(histories_a, "Exp A — LeNet-5 activation (loss)", "phase2_expA_loss.png")
    plot_accuracy_curves(histories_a, "Exp A — LeNet-5 activation (val acc)", "phase2_expA_acc.png")
    plot_gradient_norms(
        grad_histories,
        layer_labels,
        "Exp A — mean per-layer gradient norms (LeNet-5)",
        "phase2_expA_gradnorm.png",
    )
    plot_loss_curves(histories_b, "Exp B — LeNet-5 vs flat MLP (loss)", "phase2_expB_loss.png")
    plot_accuracy_curves(histories_b, "Exp B — LeNet-5 vs flat MLP (val acc)", "phase2_expB_acc.png")

    summary = {
        "expA_final_val_acc": {name: hist.val_acc[-1] for name, hist in histories_a.items()},
        "expA_grad_norm_first_vs_last": {
            name: {
                "first": float(grad_histories[name].mean_per_layer()[0]),
                "last": float(grad_histories[name].mean_per_layer()[-1]),
            }
            for name in histories_a
        },
        "expB_final_val_acc": {name: hist.val_acc[-1] for name, hist in histories_b.items()},
        "param_counts": {**param_counts, **{
            f"LeNet-5 ({activation})": sum(p.numel() for p in LeNet5(activation).parameters())
            for activation in ["sigmoid", "tanh", "relu"]
        }},
        "seconds_per_epoch": epoch_times,
    }
    (RESULTS_DIR / "phase2_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"figures written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
