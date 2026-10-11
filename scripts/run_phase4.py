#!/usr/bin/env python3
"""Phase 4 experiments: VGG — 3×3 kernel stacks and BatchNorm on Fashion-MNIST.

Run:  uv run python scripts/run_phase4.py

Experiment A — two 3×3 convs vs one 5×5: identical receptive field and spatial
    shape per stage, but the stack uses fewer weights (2·9 = 18 vs 25 per
    channel pair) and adds a nonlinearity in between. The VGG paper's argument,
    measured.
Experiment B — BatchNorm and initialization: a deep VGG without BatchNorm
    collapses when left on PyTorch's default init (signal dies across the conv
    stack), is rescued by He init, and needs neither with BatchNorm. Layer-wise
    gradient norms show the collapse directly.

The VGG-11 + BatchNorm run is the shared baseline for both experiments.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_history.data import load_fashion_mnist_loaders
from dl_history.metrics import EpochHistory, GradientHistory
from dl_history.models.vgg import VGG11, VGG11Wide
from dl_history.train import train_model
from dl_history.viz import (
    plot_accuracy_curves,
    plot_gradient_norms,
    plot_loss_curves,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

BATCH_SIZE = 128
EPOCHS = 8
LEARNING_RATE = 0.01
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
        f"    final train/val acc {history.train_acc[-1]:.4f}/{history.val_acc[-1]:.4f}, "
        f"{seconds_per_epoch:.1f}s/epoch, "
        f"{sum(p.numel() for p in model.parameters()):,} params",
        flush=True,
    )
    return history, seconds_per_epoch


def main() -> None:
    print("loading Fashion-MNIST ...", flush=True)
    loaders = load_fashion_mnist_loaders(batch_size=BATCH_SIZE)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    histories: dict[str, EpochHistory] = {}
    epoch_times: dict[str, float] = {}
    grad_histories: dict[str, GradientHistory] = {}
    layer_labels: dict[str, list[str]] = {}

    baseline_label = "VGG-11 + BatchNorm"
    print(f"shared baseline: {baseline_label}", flush=True)
    histories[baseline_label], epoch_times[baseline_label] = run(
        VGG11(batch_norm=True), baseline_label, loaders
    )

    print("Experiment A: two 3×3 convs vs one 5×5 (same receptive field)", flush=True)
    histories_a = dict(histories)
    wide_label = "VGG-11 wide (5×5, same RF)"
    histories_a[wide_label], epoch_times[wide_label] = run(
        VGG11Wide(batch_norm=True), wide_label, loaders
    )

    print("Experiment B: BatchNorm / initialization", flush=True)
    histories_b: dict[str, EpochHistory] = {}
    no_bn_default = "VGG-11 (no BN, default init)"
    no_bn_he = "VGG-11 (no BN, He init)"
    for model, label in [
        (VGG11(batch_norm=False), no_bn_default),
        (VGG11(batch_norm=False, init="he"), no_bn_he),
    ]:
        grad_histories[label] = GradientHistory()
        histories_b[label], epoch_times[label] = run(
            model, label, loaders, grad_histories[label]
        )
        layer_labels[label] = VGG11().weight_layer_names()
    histories_b[baseline_label] = histories[baseline_label]
    epoch_times[baseline_label] = epoch_times[baseline_label]

    plot_loss_curves(histories_a, "Exp A — 3×3 stack vs one 5×5 (loss)", "phase4_expA_loss.png")
    plot_accuracy_curves(
        histories_a, "Exp A — 3×3 stack vs one 5×5 (val acc)", "phase4_expA_acc.png"
    )
    plot_loss_curves(
        histories_b, "Exp B — BatchNorm / init (loss)", "phase4_expB_loss.png"
    )
    plot_accuracy_curves(
        histories_b, "Exp B — BatchNorm / init (val acc)", "phase4_expB_acc.png"
    )
    plot_gradient_norms(
        grad_histories,
        layer_labels,
        "Exp B — mean per-layer gradient norms (VGG-11, no BatchNorm)",
        "phase4_expB_gradnorm.png",
    )

    param_counts = {
        "VGG-11": sum(p.numel() for p in VGG11().parameters()),
        "VGG-11 + BatchNorm": sum(p.numel() for p in VGG11(batch_norm=True).parameters()),
        "VGG-11 wide (5×5)": sum(p.numel() for p in VGG11Wide(batch_norm=True).parameters()),
    }
    summary = {
        "expA_final_val_acc": {name: hist.val_acc[-1] for name, hist in histories_a.items()},
        "expB_final_val_acc": {name: hist.val_acc[-1] for name, hist in histories_b.items()},
        "expB_first_layer_grad_norm": {
            name: float(hist.mean_per_layer()[0]) for name, hist in grad_histories.items()
        },
        "param_counts": param_counts,
        "seconds_per_epoch": epoch_times,
    }
    (RESULTS_DIR / "phase4_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"figures written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
