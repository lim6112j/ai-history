#!/usr/bin/env python3
"""Phase 3 experiments: AlexNet — Dropout and Data Augmentation on Fashion-MNIST.

Run:  uv run python scripts/run_phase3.py

Experiment A — Dropout on/off: the same AlexNet with dropout=0.5 vs dropout=0.0.
    The no-dropout net drives train accuracy to ~1.0 while validation stalls —
    a widening train−val gap is the overfitting signature Dropout suppresses.
Experiment B — Data Augmentation (random crop + flip) on/off: geometric
    augmentation as a second regularizer, measured by validation accuracy.

The dropout=0.5 / no-augmentation run is the shared baseline for both.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dl_history.data import load_fashion_mnist_loaders
from dl_history.metrics import EpochHistory
from dl_history.models.alexnet import AlexNet
from dl_history.train import train_model
from dl_history.viz import (
    plot_accuracy_curves,
    plot_generalization_gap,
    plot_loss_curves,
)

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"

BATCH_SIZE = 128
EPOCHS = 15
LEARNING_RATE = 0.01
MOMENTUM = 0.9
SEED = 0


def run(model, label: str, loaders) -> tuple[EpochHistory, float]:
    print(f"  training {label} ...", flush=True)
    torch.manual_seed(SEED)
    history, seconds_per_epoch = train_model(
        model, loaders, epochs=EPOCHS, lr=LEARNING_RATE, momentum=MOMENTUM
    )
    gap = history.train_acc[-1] - history.val_acc[-1]
    print(
        f"    final train/val acc {history.train_acc[-1]:.4f}/{history.val_acc[-1]:.4f} "
        f"(gap {gap:+.4f}), {seconds_per_epoch:.1f}s/epoch, "
        f"{sum(p.numel() for p in model.parameters()):,} params",
        flush=True,
    )
    return history, seconds_per_epoch


def main() -> None:
    print("loading Fashion-MNIST ...", flush=True)
    plain_loaders = load_fashion_mnist_loaders(batch_size=BATCH_SIZE)
    augment_loaders = load_fashion_mnist_loaders(batch_size=BATCH_SIZE, augment=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("shared baseline: AlexNet (dropout=0.5, no augmentation)", flush=True)
    histories: dict[str, EpochHistory] = {}
    epoch_times: dict[str, float] = {}
    histories["AlexNet (dropout 0.5)"], epoch_times["AlexNet (dropout 0.5)"] = run(
        AlexNet(dropout=0.5), "AlexNet (dropout 0.5)", plain_loaders
    )

    print("Experiment A: Dropout on/off", flush=True)
    histories_a = dict(histories)
    histories_a["AlexNet (no dropout)"], epoch_times["AlexNet (no dropout)"] = run(
        AlexNet(dropout=0.0), "AlexNet (no dropout)", plain_loaders
    )

    print("Experiment B: Data Augmentation on/off", flush=True)
    histories_b = dict(histories)
    histories_b["AlexNet (dropout 0.5 + augment)"], epoch_times[
        "AlexNet (dropout 0.5 + augment)"
    ] = run(AlexNet(dropout=0.5), "AlexNet (dropout 0.5 + augment)", augment_loaders)

    plot_loss_curves(histories_a, "Exp A — Dropout on/off (loss)", "phase3_expA_loss.png")
    plot_accuracy_curves(histories_a, "Exp A — Dropout on/off (val acc)", "phase3_expA_acc.png")
    plot_generalization_gap(
        histories_a, "Exp A — train−val accuracy gap (overfitting)", "phase3_expA_gap.png"
    )
    plot_loss_curves(
        histories_b, "Exp B — Data Augmentation on/off (loss)", "phase3_expB_loss.png"
    )
    plot_accuracy_curves(
        histories_b, "Exp B — Data Augmentation on/off (val acc)", "phase3_expB_acc.png"
    )

    summary = {
        "expA_final_train_acc": {name: hist.train_acc[-1] for name, hist in histories_a.items()},
        "expA_final_val_acc": {name: hist.val_acc[-1] for name, hist in histories_a.items()},
        "expA_train_val_gap": {
            name: hist.train_acc[-1] - hist.val_acc[-1] for name, hist in histories_a.items()
        },
        "expB_final_val_acc": {name: hist.val_acc[-1] for name, hist in histories_b.items()},
        "param_counts": {"AlexNet": sum(p.numel() for p in AlexNet().parameters())},
        "seconds_per_epoch": epoch_times,
    }
    (RESULTS_DIR / "phase3_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"figures written to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
