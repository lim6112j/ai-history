"""Phase 2: LeNet-5 (LeCun et al., 1998).

Built from raw ``Conv2d`` / ``AvgPool2d`` / ``Linear`` modules so the
architecture is explicit rather than hidden behind ``nn.Sequential``. The
activation is selectable (sigmoid / tanh / relu) to reproduce the 1998-era
squashing functions and measure what ReLU buys.

Output-size bookkeeping for a 28×28 Fashion-MNIST image, W_out = (W − F + 2P)/S + 1:

    input  1×28×28
    conv1  F=5, P=2, S=1 → (28 − 5 + 4)/1 + 1 = 28   → 6×28×28
    pool1  2×2, S=2      → 28/2 = 14                 → 6×14×14
    conv2  F=5, P=0, S=1 → (14 − 5 + 0)/1 + 1 = 10   → 16×10×10
    pool2  2×2, S=2      → 10/2 = 5                  → 16×5×5
    flatten                           → 400
    fc1 400→120, fc2 120→84, fc3 84→10
"""

from __future__ import annotations

import torch
from torch import nn

_ACTIVATIONS = {"sigmoid": nn.Sigmoid, "tanh": nn.Tanh, "relu": nn.ReLU}

# (W − F + 2P) / S + 1 with the layer settings below.
_CONV1_OUT = (28 - 5 + 2 * 2) // 1 + 1  # 28
_CONV2_OUT = (_CONV1_OUT // 2 - 5 + 0) // 1 + 1  # 10
_FLATTEN_DIM = 16 * (_CONV2_OUT // 2) ** 2  # 400


def _activation(name: str) -> nn.Module:
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation {name!r}; expected one of {sorted(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]()


class LeNet5(nn.Module):
    """LeNet-5 for 28×28 grayscale images (the classic 6-16-120-84-10 trunk)."""

    def __init__(self, activation: str = "relu") -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(_FLATTEN_DIM, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.activate = _activation(activation)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.activate(self.conv1(x)))
        x = self.pool(self.activate(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.activate(self.fc1(x))
        x = self.activate(self.fc2(x))
        return self.fc3(x)

    def weight_layer_names(self) -> list[str]:
        """Names of weight tensors in forward order (for gradient-norm plots)."""
        return [name for name, param in self.named_parameters() if name.endswith("weight")]
