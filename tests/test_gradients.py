"""Numerical gradient checks for the Phase 1 manual backprop."""

import numpy as np
import pytest

from dl_history.models.mlp_numpy import MLP, check_gradients

TOLERANCE = 1e-6


def make_model(activation: str, seed: int = 0) -> MLP:
    return MLP.build(
        in_dim=8, hidden_dims=[16, 12], num_classes=5, activation=activation, seed=seed
    )


@pytest.mark.parametrize("activation", ["sigmoid", "tanh", "relu"])
def test_backprop_matches_numerical_gradient(activation: str) -> None:
    # Arrange: small net + fixed random batch
    rng = np.random.default_rng(42)
    model = make_model(activation)
    x = rng.standard_normal((16, 8))
    labels = rng.integers(0, 5, size=16)

    # Act: compare backprop against central-difference numerical gradients
    max_rel = check_gradients(model, x, labels, eps=1e-5, num_checks=10)

    # Assert
    assert max_rel < TOLERANCE, f"{activation}: rel err {max_rel:.2e} ≥ {TOLERANCE:.0e}"


@pytest.mark.parametrize("activation", ["sigmoid", "tanh", "relu"])
def test_gradient_norm_recorded_per_weight_layer(activation: str) -> None:
    # Arrange: depth = 2 hidden + output = 3 Dense layers
    model = make_model(activation)
    rng = np.random.default_rng(1)
    x = rng.standard_normal((32, 8))
    labels = rng.integers(0, 5, size=32)

    # Act
    model.train_batch(x, labels, lr=0.1)
    model.train_batch(x, labels, lr=0.1)

    # Assert: two steps recorded, 3 weight-layer norms per step
    assert len(model.gradient_history.grad_norms) == 2
    assert all(len(row) == 3 for row in model.gradient_history.grad_norms)
    assert all(np.all(row >= 0) for row in model.gradient_history.grad_norms)


def test_deep_sigmoid_first_layer_gradient_much_smaller_than_last() -> None:
    """The vanishing-gradient signature: layer-1 norms ≪ layer-N norms.

    10 weight layers of sigmoid with Xavier init should push early-layer
    gradients several orders of magnitude below the final layer's.
    """
    # Arrange: 10 weight layers of sigmoid with Xavier init
    model = MLP.build(
        in_dim=32, hidden_dims=[64] * 9, num_classes=10, activation="sigmoid", seed=3
    )
    rng = np.random.default_rng(3)
    x = rng.standard_normal((64, 32)) * 0.5
    labels = rng.integers(0, 10, size=64)

    # Act
    model.train_batch(x, labels, lr=0.1)

    # Assert: first weight layer's grad norm far below the last layer's
    norms = model.gradient_history.grad_norms[0]
    assert norms[0] < norms[-1] * 1e-3


def test_unknown_activation_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown activation"):
        MLP.build(in_dim=8, hidden_dims=[16], num_classes=5, activation="gelu")


def test_unknown_initializer_rejected() -> None:
    rng = np.random.default_rng(0)
    from dl_history.models.mlp_numpy import Dense

    with pytest.raises(ValueError, match="Unknown initializer"):
        Dense(4, 4, rng, initializer="orthogonal")