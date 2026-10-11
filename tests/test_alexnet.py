"""Phase 3 tests: AlexNet shapes, parameter counts, and Dropout semantics."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dl_history.data import train_transform
from dl_history.metrics import GradientHistory
from dl_history.models.alexnet import AlexNet
from dl_history.train import train_model

# conv1 1,664 + conv2 307,392 + conv3 663,936 + conv4 884,992 + conv5 590,080
# + fc6 4,195,328 + fc7 1,049,600 + fc8 10,250
ALEXNET_PARAM_COUNT = 7_703_242


@pytest.mark.parametrize("dropout", [0.0, 0.5])
def test_forward_output_shape(dropout: float) -> None:
    model = AlexNet(dropout=dropout)
    logits = model(torch.randn(4, 1, 28, 28))
    assert logits.shape == (4, 10)


@pytest.mark.parametrize("dropout", [-0.1, 1.0, 1.5])
def test_out_of_range_dropout_rejected(dropout: float) -> None:
    with pytest.raises(ValueError, match="dropout must be in"):
        AlexNet(dropout=dropout)


def test_non_positive_hidden_dim_rejected() -> None:
    with pytest.raises(ValueError, match="hidden_dim must be positive"):
        AlexNet(hidden_dim=0)


def test_alexnet_parameter_count() -> None:
    assert sum(p.numel() for p in AlexNet().parameters()) == ALEXNET_PARAM_COUNT


def test_flatten_dim_matches_output_size_formula() -> None:
    """The hand-computed (W−F+2P)/S+1 chain must match the real tensor sizes."""
    # Arrange: replay the conv/pool stages on a real 28×28 batch
    model = AlexNet()

    # Act
    x = torch.randn(2, 1, 28, 28)
    x = model.pool(model.act(model.conv1(x)))
    x = model.pool(model.act(model.conv2(x)))
    x = model.act(model.conv3(x))
    x = model.act(model.conv4(x))
    x = model.pool(model.act(model.conv5(x)))

    # Assert: 256 channels × 4×4 spatial, exactly the flatten dim fc6 expects
    assert x.shape == (2, 256, 4, 4)
    assert x.shape[1] * x.shape[2] * x.shape[3] == model.fc6.in_features == 4096


def test_weight_layer_names_in_forward_order() -> None:
    assert AlexNet().weight_layer_names() == [
        "conv1.weight", "conv2.weight", "conv3.weight", "conv4.weight",
        "conv5.weight", "fc6.weight", "fc7.weight", "fc8.weight",
    ]


def test_dropout_is_identity_at_eval() -> None:
    """Inverted dropout: eval mode must be deterministic, no rescaling needed."""
    model = AlexNet(dropout=0.5).eval()
    x = torch.randn(8, 1, 28, 28)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_zero_dropout_is_deterministic_in_train_mode() -> None:
    model = AlexNet(dropout=0.0).train()
    x = torch.randn(8, 1, 28, 28)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_dropout_scales_survivors_by_inverse_keep_prob() -> None:
    """Train mode: ~p of units zeroed, survivors scaled by 1/(1−p) = 2."""
    # Arrange: capture the input/output of the first dropout call
    model = AlexNet(dropout=0.5).train()
    captured: dict[str, torch.Tensor] = {}
    handle = model.dropout.register_forward_hook(
        lambda _module, inputs, output: captured.update(inp=inputs[0].detach(), out=output.detach())
    )
    torch.manual_seed(0)

    # Act
    with torch.no_grad():
        model(torch.randn(128, 1, 28, 28))
    handle.remove()

    # Assert
    inp, out = captured["inp"], captured["out"]
    alive = inp != 0  # ReLU has already zeroed the negative units
    zero_fraction = (out[alive] == 0).float().mean().item()
    assert 0.45 < zero_fraction < 0.55
    survivors = alive & (out != 0)
    assert torch.allclose(out[survivors], inp[survivors] * 2.0, atol=1e-5)


def test_train_model_records_gradient_norms() -> None:
    # Arrange: tiny loader (3 batches of 4) + one training epoch
    images = torch.randn(12, 1, 28, 28)
    labels = torch.randint(0, 10, (12,))
    loaders = (
        DataLoader(TensorDataset(images, labels), batch_size=4),
        DataLoader(TensorDataset(images, labels), batch_size=4),
    )
    grad_history = GradientHistory()

    # Act
    train_model(AlexNet(dropout=0.0), loaders, epochs=1, lr=0.01, grad_history=grad_history)

    # Assert: one record per training step, one norm per weight layer
    assert len(grad_history.grad_norms) == 3
    assert all(len(row) == 8 for row in grad_history.grad_norms)


def test_train_transform_adds_augmentation_steps() -> None:
    assert len(train_transform(augment=False).transforms) == 2
    assert len(train_transform(augment=True).transforms) == 4
    image = np.random.rand(28, 28).astype(np.float32)  # HWC, as ToTensor expects
    assert train_transform(augment=True)(image).shape == (1, 28, 28)
