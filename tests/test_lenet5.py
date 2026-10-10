"""Phase 2 tests: LeNet-5 shapes, the output-size formula, parameter counts."""

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from dl_history.metrics import GradientHistory
from dl_history.models.lenet5 import LeNet5
from dl_history.models.mlp_torch import FlatMLP
from dl_history.train import train_model

# conv1 156 + conv2 2416 + fc1 48120 + fc2 10164 + fc3 850
LENET_PARAM_COUNT = 61706
# fc1 100480 + fc2 1290
FLAT_MLP_PARAM_COUNT = 101770


@pytest.mark.parametrize("activation", ["sigmoid", "tanh", "relu"])
def test_forward_output_shape(activation: str) -> None:
    model = LeNet5(activation=activation)
    logits = model(torch.randn(4, 1, 28, 28))
    assert logits.shape == (4, 10)


def test_unknown_activation_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown activation"):
        LeNet5(activation="gelu")


def test_flatten_dim_matches_output_size_formula() -> None:
    """The hand-computed (W−F+2P)/S+1 chain must match the real tensor sizes."""
    # Arrange: model with a truncated forward pass — stop after pool2
    model = LeNet5()

    # Act: replay conv/pool stages on a real 28×28 batch
    x = torch.randn(2, 1, 28, 28)
    x = model.pool(model.activate(model.conv1(x)))
    x = model.pool(model.activate(model.conv2(x)))

    # Assert: 16 channels × 5×5 spatial, exactly the flatten dim fc1 expects
    assert x.shape == (2, 16, 5, 5)
    assert x.shape[1] * x.shape[2] * x.shape[3] == model.fc1.in_features == 400


def test_lenet5_parameter_count() -> None:
    """LeNet-5 (~62k) should be far smaller than the flat MLP (~102k)."""
    assert sum(p.numel() for p in LeNet5().parameters()) == LENET_PARAM_COUNT


def test_flat_mlp_parameter_count() -> None:
    assert sum(p.numel() for p in FlatMLP().parameters()) == FLAT_MLP_PARAM_COUNT


@pytest.mark.parametrize("activation", ["sigmoid", "tanh", "relu"])
def test_weight_layer_names_in_forward_order(activation: str) -> None:
    names = LeNet5(activation).weight_layer_names()
    assert names == ["conv1.weight", "conv2.weight", "fc1.weight", "fc2.weight", "fc3.weight"]


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
    train_model(LeNet5(), loaders, epochs=1, lr=0.01, grad_history=grad_history)

    # Assert: one record per training step, one norm per weight layer
    assert len(grad_history.grad_norms) == 3
    assert all(len(row) == 5 for row in grad_history.grad_norms)
    assert all(all(value >= 0 for value in row) for row in grad_history.grad_norms)
