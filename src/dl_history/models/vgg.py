"""Phase 4: VGG-11 / VGG-16 (Simonyan & Zisserman, 2014).

The idea under test is the **3×3 stack**: two stacked 3×3 convolutions cover the
same receptive field as a single 5×5 convolution, but use fewer weights
(2·3·3 = 18 vs 5·5 = 25 per input/output channel pair) and insert an extra
nonlinearity between them. :func:`make_layers` builds the trunk from a config
list — the block-factory style the VGG paper popularised — and is the one place
in this project where ``nn.Sequential`` is the point rather than a shortcut.

Two adaptations to Fashion-MNIST's 28×28 grayscale images:

1. **Input padded 28 → 32** (``INPUT_PAD``). The canonical config has five
   MaxPool stages; on a 28×28 input the fifth pool would leave an empty 0×0
   map, so the border is zero-padded to 32×32 → 16 → 8 → 4 → 2 → 1. Because
   inputs are standardized (mean 0), a zero border is a *mean-valued* border.
2. **Classifier scaled down** from the original 4096-4096 to ``hidden_dim``
   (default 512). The published 4096-wide head is sized for ImageNet's 1000
   classes and 1.2M images; on 60k Fashion-MNIST images it is pure overfitting
   and dominates the parameter count.

Configs (``M`` = 2×2 MaxPool stride 2):

    VGG-11: 64, M, 128, M, 256, 256, M, 512, 512, M, 512, 512, M   (8 conv)
    VGG-16: 64, 64, M, 128, 128, M, 256, 256, 256, M,
            512, 512, 512, M, 512, 512, 512, M                    (13 conv)
"""

from __future__ import annotations

import torch
from torch import nn

CFG_VGG11 = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
CFG_VGG16 = [
    64, 64, "M", 128, 128, "M", 256, 256, 256, "M",
    512, 512, 512, "M", 512, 512, 512, "M",
]

# One 5×5 conv replaces each *pair* of 3×3 convs in CFG_VGG11 — same receptive
# field per stage, the ablation for the 3×3-stack argument.
CFG_VGG11_WIDE = [64, "M", 128, "M", 256, "M", 512, "M", 512, "M"]

_POOL = "M"
INPUT_PAD = 2  # 28 + 2·2 = 32, so five pool stages end at 1×1
_CONV_KERNEL = 3
DEFAULT_DROPOUT = 0.5
DEFAULT_HIDDEN_DIM = 512
_FEATURES_CHANNELS = 512  # final trunk width, i.e. classifier input
_INIT_SCHEMES = ("default", "he")


def apply_he_init(module: nn.Module) -> None:
    """Kaiming/He init for Conv2d and Linear layers (BatchNorm untouched).

    PyTorch's default init lets the activation signal collapse across a deep
    conv stack when there is no BatchNorm to renormalize it. This is the
    variance-preserving alternative — the Phase 4 experiment uses it to show
    that BatchNorm removes the need for a hand-tuned initialization.
    """
    for child in module.modules():
        if isinstance(child, nn.Conv2d):
            nn.init.kaiming_normal_(child.weight, mode="fan_out", nonlinearity="relu")
            nn.init.zeros_(child.bias)
        elif isinstance(child, nn.Linear):
            nn.init.kaiming_normal_(child.weight, nonlinearity="relu")
            nn.init.zeros_(child.bias)


def make_layers(
    cfg: list, batch_norm: bool = False, in_channels: int = 1, kernel_size: int = _CONV_KERNEL
) -> nn.Sequential:
    """Build the convolutional trunk from a config list.

    Each integer entry becomes Conv2d(kernel_size, padding=kernel_size//2) →
    optional BatchNorm2d → ReLU; each ``"M"`` becomes a 2×2 MaxPool stride 2.
    ``padding = kernel_size // 2`` keeps the spatial size unchanged, which is
    what lets the 5×5 ablation match the 3×3 stack's shape exactly.
    """
    padding = kernel_size // 2
    layers: list[nn.Module] = []
    for entry in cfg:
        if entry == _POOL:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        layers.append(nn.Conv2d(in_channels, entry, kernel_size=kernel_size, padding=padding))
        if batch_norm:
            layers.append(nn.BatchNorm2d(entry))
        layers.append(nn.ReLU(inplace=True))
        in_channels = entry
    return nn.Sequential(*layers)


class VGG(nn.Module):
    """VGG trunk + a three-layer classifier with Dropout.

    Args:
        cfg: Layer config list consumed by :func:`make_layers`.
        batch_norm: Insert BatchNorm2d after every convolution.
        dropout: Dropout probability in the classifier (0.0 disables it).
        hidden_dim: Width of the two fully-connected classifier layers.
        kernel_size: Convolution kernel size; 5 builds the "one wide conv per
            3×3 pair" ablation.
        init: ``"default"`` (PyTorch's own init) or ``"he"`` (variance
            preserving — see :func:`apply_he_init`).
    """

    def __init__(
        self,
        cfg: list,
        batch_norm: bool = False,
        dropout: float = DEFAULT_DROPOUT,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        kernel_size: int = _CONV_KERNEL,
        init: str = "default",
    ) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim!r}")
        if init not in _INIT_SCHEMES:
            raise ValueError(f"Unknown init {init!r}; expected one of {list(_INIT_SCHEMES)}")

        self.pad = nn.ZeroPad2d(INPUT_PAD)
        self.features = make_layers(cfg, batch_norm=batch_norm, kernel_size=kernel_size)
        self.fc1 = nn.Linear(_FEATURES_CHANNELS, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 10)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

        if init == "he":
            apply_he_init(self)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(self.pad(x))
        x = torch.flatten(x, 1)
        x = self.dropout(self.act(self.fc1(x)))
        x = self.dropout(self.act(self.fc2(x)))
        return self.fc3(x)

    def weight_layer_names(self) -> list[str]:
        """Names of weight tensors in forward order (for gradient-norm plots)."""
        return [name for name, param in self.named_parameters() if name.endswith("weight")]


def VGG11(batch_norm: bool = False, **kwargs) -> VGG:
    """VGG-11: eight 3×3 convolutions in five stages."""
    return VGG(CFG_VGG11, batch_norm=batch_norm, **kwargs)


def VGG16(batch_norm: bool = False, **kwargs) -> VGG:
    """VGG-16: thirteen 3×3 convolutions in five stages."""
    return VGG(CFG_VGG16, batch_norm=batch_norm, **kwargs)


def VGG11Wide(batch_norm: bool = False, **kwargs) -> VGG:
    """VGG-11 ablation: one 5×5 conv in place of each 3×3 pair.

    Same receptive field and spatial shape per stage, more weights
    (25 vs 2·9 = 18 per channel pair) and one fewer nonlinearity.
    """
    return VGG(CFG_VGG11_WIDE, batch_norm=batch_norm, kernel_size=5, **kwargs)
