"""Phase 3: AlexNet (Krizhevsky, Sutskever & Hinton, 2012).

Faithful in structure to the original — five convolutions with Max Pooling
downsampling, then a three-layer classifier with Dropout — but adapted to
Fashion-MNIST's 28×28 grayscale images instead of 224×224 RGB. The 11×11/
stride-4 stem and the 4096-wide classifier are scaled down to fit the smaller
input; the ideas under test (ReLU, Max Pooling, Dropout, augmentation) are
unchanged. Layers are written out explicitly rather than wrapped in
``nn.Sequential``, matching Phase 2's style.

Dropout uses PyTorch's *inverted* formulation: during training each unit is
kept with probability ``1 − p`` and the survivors are scaled by ``1/(1 − p)``,
so the expected activation is unchanged; at eval time the layer is the
identity. No test-time rescaling is therefore required (see
``tests/test_alexnet.py``).

Output-size bookkeeping for a 28×28 image, W_out = (W − F + 2P)/S + 1 for
convolutions and floor((W + 2P − F)/S) + 1 for pooling:

    input   1×28×28
    conv1   F=5, P=2, S=1 → (28 − 5 + 4)/1 + 1 = 28  →  64×28×28
    pool1   F=3, P=1, S=2 → floor((28 + 2 − 3)/2) + 1 = 14  →  64×14×14
    conv2   F=5, P=2, S=1 → 14                        → 192×14×14
    pool2   F=3, P=1, S=2 → 7                         → 192×7×7
    conv3   F=3, P=1, S=1 → 7                         → 384×7×7
    conv4   F=3, P=1, S=1 → 7                         → 256×7×7
    conv5   F=3, P=1, S=1 → 7                         → 256×7×7
    pool3   F=3, P=1, S=2 → 4                         → 256×4×4
    flatten                                            → 4096
    fc6 4096→1024 (ReLU→Dropout), fc7 1024→1024 (ReLU→Dropout), fc8 1024→10
"""

from __future__ import annotations

import torch
from torch import nn

# (W − F + 2P)/S + 1 for convs, floor((W + 2P − F)/S) + 1 for pools.
_CONV1_OUT = (28 - 5 + 2 * 2) // 1 + 1  # 28
_POOL1_OUT = (_CONV1_OUT + 2 * 1 - 3) // 2 + 1  # 14
_CONV2_OUT = (_POOL1_OUT - 5 + 2 * 2) // 1 + 1  # 14
_POOL2_OUT = (_CONV2_OUT + 2 * 1 - 3) // 2 + 1  # 7
_CONV3_OUT = (_POOL2_OUT - 3 + 2 * 1) // 1 + 1  # 7
_CONV4_OUT = (_CONV3_OUT - 3 + 2 * 1) // 1 + 1  # 7
_CONV5_OUT = (_CONV4_OUT - 3 + 2 * 1) // 1 + 1  # 7
_POOL3_OUT = (_CONV5_OUT + 2 * 1 - 3) // 2 + 1  # 4
_FLATTEN_DIM = 256 * _POOL3_OUT**2  # 4096

DEFAULT_DROPOUT = 0.5
DEFAULT_HIDDEN_DIM = 1024


class AlexNet(nn.Module):
    """AlexNet adapted to 28×28 grayscale input.

    Args:
        dropout: Dropout probability in the classifier. ``0.0`` disables
            dropout entirely — the ablation for the overfitting experiment.
        hidden_dim: Width of the two fully-connected classifier layers. The
            original's 4096 is scaled down for the smaller input.
    """

    def __init__(self, dropout: float = DEFAULT_DROPOUT, hidden_dim: int = DEFAULT_HIDDEN_DIM) -> None:
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout!r}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim!r}")

        self.conv1 = nn.Conv2d(1, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(64, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(384, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.fc6 = nn.Linear(_FLATTEN_DIM, hidden_dim)
        self.fc7 = nn.Linear(hidden_dim, hidden_dim)
        self.fc8 = nn.Linear(hidden_dim, 10)

        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.act(self.conv1(x)))  # 64×14×14
        x = self.pool(self.act(self.conv2(x)))  # 192×7×7
        x = self.act(self.conv3(x))
        x = self.act(self.conv4(x))
        x = self.pool(self.act(self.conv5(x)))  # 256×4×4
        x = torch.flatten(x, 1)
        x = self.dropout(self.act(self.fc6(x)))
        x = self.dropout(self.act(self.fc7(x)))
        return self.fc8(x)

    def weight_layer_names(self) -> list[str]:
        """Names of weight tensors in forward order (for gradient-norm plots)."""
        return [name for name, param in self.named_parameters() if name.endswith("weight")]
