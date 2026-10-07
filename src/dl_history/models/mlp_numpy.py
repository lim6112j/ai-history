"""Phase 1: pure-NumPy MLP with manual backpropagation (no autograd).

Every layer implements forward(x) and backward(grad) — the chain rule is
written out by hand so the mechanics of backprop are visible in code.
"""

from __future__ import annotations

import numpy as np

from ..metrics import GradientHistory


class Layer:
    """Base class for layers with manual forward/backward."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def params_and_grads(self):
        """Yield (param, grad) pairs for optimizer updates. Empty by default."""
        return
        yield  # pragma: no cover — makes this a generator

    def layer_name(self) -> str:
        return type(self).__name__


class Dense(Layer):
    """Fully-connected layer: y = x @ W + b.

    Shapes: x (N, in), W (in, out), b (out). Caches forward inputs for
    backward. Initializer: "xavier" for sigmoid/tanh nets, "he" for ReLU nets.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rng: np.random.Generator,
        initializer: str = "xavier",
    ) -> None:
        if initializer not in ("xavier", "he"):
            raise ValueError(f"Unknown initializer: {initializer!r}")
        if initializer == "he":
            std = np.sqrt(2.0 / in_features)
        else:
            std = np.sqrt(1.0 / in_features)
        self.W = rng.normal(0.0, std, size=(in_features, out_features)).astype(np.float64)
        self.b = np.zeros(out_features, dtype=np.float64)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return x @ self.W + self.b

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("backward() called before forward()")
        self.dW = self._x.T @ grad
        self.db = grad.sum(axis=0)
        return grad @ self.W.T

    def params_and_grads(self):
        yield self.W, self.dW
        yield self.b, self.db

    def layer_name(self) -> str:
        return f"Dense({self.W.shape[0]}→{self.W.shape[1]})"


class Sigmoid(Layer):
    """σ(z) = 1 / (1 + e^{-z}); σ'(z) = σ(z)(1 − σ(z)).

    The |σ'| ≤ 0.25 ceiling is exactly what vanishes gradients in deep
    sigmoid stacks — that is the phenomenon Experiment A must expose.
    """

    def __init__(self) -> None:
        self._out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._out = 1.0 / (1.0 + np.exp(-x))
        return self._out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self._out is None:
            raise RuntimeError("backward() called before forward()")
        return grad * self._out * (1.0 - self._out)


class Tanh(Layer):
    """tanh(z); tanh'(z) = 1 − tanh²(z). Range (−1, 1), zero-centered."""

    def __init__(self) -> None:
        self._out: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._out = np.tanh(x)
        return self._out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self._out is None:
            raise RuntimeError("backward() called before forward()")
        return grad * (1.0 - self._out**2)


class ReLU(Layer):
    """max(0, z); derivative is 1 where z > 0, else 0."""

    def __init__(self) -> None:
        self._x: np.ndarray | None = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        return np.maximum(0.0, x)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        if self._x is None:
            raise RuntimeError("backward() called before forward()")
        return grad * (self._x > 0)


class SoftmaxCrossEntropy(Layer):
    """Combined softmax + cross-entropy loss (numerically stable).

    forward returns the mean loss over the batch; backward must be seeded
    with 1.0 and produces dL/dz = (softmax(z) − onehot) / N.
    """

    def __init__(self) -> None:
        self._probs: np.ndarray | None = None
        self._labels: np.ndarray | None = None

    def forward(self, logits: np.ndarray, labels: np.ndarray) -> float:
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        self._probs = exp / exp.sum(axis=1, keepdims=True)
        self._labels = labels
        n = logits.shape[0]
        picked = self._probs[np.arange(n), labels]
        return float(-np.mean(np.log(np.clip(picked, 1e-12, None))))

    def backward(self, grad: float = 1.0) -> np.ndarray:
        if self._probs is None or self._labels is None:
            raise RuntimeError("backward() called before forward()")
        n = self._probs.shape[0]
        dz = self._probs.copy()
        dz[np.arange(n), self._labels] -= 1.0
        return grad * dz / n


class MLP:
    """Stack of Dense+activation layers with mini-batch SGD.

    Gradient norms are recorded on every update so deep sigmoid nets can be
    compared against deep ReLU nets layer by layer.
    """

    def __init__(self, layers: list[Layer], loss: SoftmaxCrossEntropy, rng: np.random.Generator) -> None:
        if not layers:
            raise ValueError("MLP needs at least one layer")
        self.layers = layers
        self.loss = loss
        self.rng = rng
        self.gradient_history = GradientHistory()

    @classmethod
    def build(
        cls,
        in_dim: int,
        hidden_dims: list[int],
        num_classes: int,
        activation: str = "sigmoid",
        seed: int = 0,
    ) -> "MLP":
        """Construct an MLP with one activation after every hidden Dense layer.

        hidden_dims=[256, 128] gives a 3-layer net (2 hidden + output);
        depth is len(hidden_dims) + 1 weight layers.
        """
        if activation not in ("sigmoid", "tanh", "relu"):
            raise ValueError(f"Unknown activation: {activation!r}")
        rng = np.random.default_rng(seed)
        init = "he" if activation == "relu" else "xavier"
        act_cls = {"sigmoid": Sigmoid, "tanh": Tanh, "relu": ReLU}[activation]

        dims = [in_dim, *hidden_dims]
        layers: list[Layer] = []
        for i, (fan_in, fan_out) in enumerate(zip(dims, dims[1:])):
            layers.append(Dense(fan_in, fan_out, rng, initializer=init))
            layers.append(act_cls())
        layers.append(Dense(dims[-1], num_classes, rng, initializer=init))
        return cls(layers, SoftmaxCrossEntropy(), rng)

    def layer_names(self) -> list[str]:
        return [layer.layer_name() for layer in self.layers if isinstance(layer, Dense)]

    def forward(self, x: np.ndarray) -> np.ndarray:
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def step(self, x: np.ndarray, labels: np.ndarray) -> float:
        """One forward+backward pass; returns the batch loss."""
        logits = self.forward(x)
        batch_loss = self.loss.forward(logits, labels)
        grad = self.loss.backward(1.0)
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return batch_loss

    def update(self, lr: float) -> None:
        """SGD step: param ← param − lr·grad. Records per-layer ||dW||₂."""
        norms = []
        for layer in self.layers:
            for param, grad in layer.params_and_grads():
                param -= lr * grad
            if isinstance(layer, Dense):
                norms.append(np.linalg.norm(layer.dW))
        self.gradient_history.record(np.array(norms))

    def train_batch(self, x: np.ndarray, labels: np.ndarray, lr: float) -> float:
        loss = self.step(x, labels)
        self.update(lr)
        return loss

    def predict(self, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
        """Argmax class predictions, batched to bound memory."""
        predictions = []
        for start in range(0, x.shape[0], batch_size):
            logits = self.forward(x[start : start + batch_size])
            predictions.append(np.argmax(logits, axis=1))
        return np.concatenate(predictions)


def check_gradients(
    model: MLP, x: np.ndarray, labels: np.ndarray, eps: float = 1e-5, num_checks: int = 20
) -> float:
    """Max relative error between backprop and central-difference gradients.

    Samples `num_checks` random weight entries per Dense layer. Uses
    rel_error = |a − n| / max(|a|, |n|), which stays honest near zero.
    """
    model.step(x, labels)
    analytic = [(layer.dW.copy(), layer.db.copy()) for layer in model.layers if isinstance(layer, Dense)]
    dense_layers = [layer for layer in model.layers if isinstance(layer, Dense)]

    max_rel = 0.0
    for layer_idx, layer in enumerate(dense_layers):
        for param_name, param in (("W", layer.W), ("b", layer.b)):
            flat = param.reshape(-1)
            chosen = model.rng.choice(flat.size, size=min(num_checks, flat.size), replace=False)
            for flat_idx in chosen:
                idx = np.unravel_index(flat_idx, param.shape)
                original = param[idx]
                param[idx] = original + eps
                plus = model.loss.forward(model.forward(x), labels)
                param[idx] = original - eps
                minus = model.loss.forward(model.forward(x), labels)
                param[idx] = original
                numerical = (plus - minus) / (2 * eps)

                if param_name == "W":
                    analytic_value = analytic[layer_idx][0][idx]
                else:
                    analytic_value = analytic[layer_idx][1][idx]
                denom = max(abs(analytic_value), abs(numerical), 1e-12)
                max_rel = max(max_rel, abs(analytic_value - numerical) / denom)
    return max_rel