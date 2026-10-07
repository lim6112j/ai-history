"""Shared PyTorch training loop used by Phase 2+.

Phase 1 trains its NumPy MLP with its own loop inside mlp_numpy.py; this
module exists so every later phase trains and evaluates identically.
"""

from __future__ import annotations

import time

import torch
from torch import nn

from .metrics import EpochHistory


def _device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    """Run one epoch. With optimizer=None this is evaluation-only."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            batch = labels.size(0)
            total_loss += loss.item() * batch
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += batch

    if total == 0:
        raise ValueError("Empty data loader")
    return total_loss / total, correct / total


def train_model(
    model: nn.Module,
    loaders: tuple,
    epochs: int,
    lr: float = 1e-3,
    device_name: str | None = None,
) -> tuple[EpochHistory, float]:
    """Train `model` and report history plus epoch time.

    Args:
        model: PyTorch model (e.g. LeNet-5, AlexNet, VGG, ResNet).
        loaders: (train_loader, test_loader) from dl_history.data.
        epochs: Number of training epochs.
        lr: SGD learning rate.
        device_name: Override device ("mps"/"cuda"/"cpu"); auto-detect if None.

    Returns:
        (EpochHistory with per-epoch train/val metrics, seconds per epoch)
    """
    device = torch.device(device_name) if device_name else _device()
    train_loader, test_loader = loaders
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    history = EpochHistory()
    seconds_per_epoch = 0.0
    for _ in range(epochs):
        start = time.perf_counter()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, test_loader, criterion, device, None)
        seconds_per_epoch = time.perf_counter() - start
        history.record(train_loss, train_acc, val_loss, val_acc)

    return history, seconds_per_epoch / max(epochs, 1)