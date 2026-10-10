"""Matplotlib visualizations shared across phases."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe: scripts write PNGs to results/
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

from .metrics import EpochHistory, GradientHistory

# Korean-capable fonts first (macOS/Windows). matplotlib 3.11 logs a
# "findfont: Font family ... not found" warning for every missing entry in the
# list even when an earlier one resolves, so keep only installed fonts.
_KOREAN_FONTS = ["Apple SD Gothic Neo", "Malgun Gothic", "DejaVu Sans"]
_INSTALLED_FONTS = {font.name for font in matplotlib.font_manager.fontManager.ttflist}
plt.rcParams["font.family"] = [name for name in _KOREAN_FONTS if name in _INSTALLED_FONTS]
plt.rcParams["axes.unicode_minus"] = False

# Plain-text log ticks ("1e-1"): mathtext exponents render as tofu when the
# active CJK font lacks superscript glyphs. Applied to minor ticks too —
# their default mathtext (U+2212 minus) has no glyphs either, and on a
# sub-decade range the minor ticks are the only labels below the top decade.
LOG_TICKS = FuncFormatter(lambda value, _: f"{value:g}")


def _apply_log_ticks(ax) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(LOG_TICKS)
    ax.yaxis.set_minor_formatter(LOG_TICKS)

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"

FIG_SIZE = (8, 5)
DPI = 150


def _ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def plot_loss_curves(histories: dict[str, EpochHistory], title: str, filename: str) -> Path:
    """Plot train/val loss for several runs on one figure."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for label, hist in histories.items():
        ax.plot(hist.train_loss, label=f"{label} (train)")
        ax.plot(hist.val_loss, "--", label=f"{label} (val)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title(title)
    _apply_log_ticks(ax)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = _ensure_results_dir() / filename
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def plot_accuracy_curves(histories: dict[str, EpochHistory], title: str, filename: str) -> Path:
    """Plot validation accuracy for several runs on one figure."""
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    for label, hist in histories.items():
        ax.plot(hist.val_acc, label=label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = _ensure_results_dir() / filename
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out


def plot_gradient_norms(
    gradient_histories: dict[str, GradientHistory],
    layer_labels: dict[str, list[str]],
    title: str,
    filename: str,
) -> Path:
    """Plot mean per-layer gradient norms (log scale) — vanishing gradient evidence.

    gradient_histories maps run label -> GradientHistory.
    layer_labels maps run label -> list of layer names in forward order.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    for label, history in gradient_histories.items():
        means = history.mean_per_layer()
        xs = np.arange(1, len(means) + 1)
        ax.plot(xs, np.maximum(means, 1e-20), marker="o", markersize=4, label=label)

    # Tick labels from the deepest run — shallow runs simply have fewer points.
    labels = max(layer_labels.values(), key=len) if layer_labels else []
    if labels:
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Layer (forward order)")
    ax.set_ylabel("Mean ||dW||₂ (log scale)")
    ax.set_title(title)
    _apply_log_ticks(ax)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = _ensure_results_dir() / filename
    fig.savefig(out, dpi=DPI)
    plt.close(fig)
    return out