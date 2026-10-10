"""Phase 1 MLP ported to PyTorch — the flat baseline for Phase 2 comparisons.

Same shape as Phase 1's Experiment C (784 → 128 → 10) so the LeNet-5 vs
flat-MLP comparison isolates the effect of convolution, not of a different
training loop: both models run through the shared ``train_model``.
"""

from __future__ import annotations

import torch
from torch import nn

from .lenet5 import _activation


class FlatMLP(nn.Module):
    """Fully-connected baseline: flatten the image, forget spatial structure."""

    def __init__(self, activation: str = "relu", hidden_dim: int = 128) -> None:
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 10)
        self.activate = _activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = self.activate(self.fc1(x))
        return self.fc2(x)

    def weight_layer_names(self) -> list[str]:
        """Names of weight tensors in forward order (for gradient-norm plots)."""
        return [name for name, param in self.named_parameters() if name.endswith("weight")]
