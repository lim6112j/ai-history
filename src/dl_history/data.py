"""Fashion-MNIST loading utilities shared across all phases."""

from __future__ import annotations

from pathlib import Path

import numpy as np

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"
IMAGE_SIZE = 28
NUM_CLASSES = 10
INPUT_DIM = IMAGE_SIZE * IMAGE_SIZE  # 784

_FASHION_CLASSES = (
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
)


def load_fashion_mnist_numpy() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Fashion-MNIST as flat NumPy arrays for Phase 1.

    Returns:
        (x_train, y_train, x_test, y_test) where images are float32 in [0, 1]
        with shape (N, 784) and labels are int64 with shape (N,).
    """
    import torchvision
    from torchvision import transforms

    to_array = transforms.ToTensor()
    train_set = torchvision.datasets.FashionMNIST(
        root=DATA_ROOT, train=True, download=True, transform=to_array
    )
    test_set = torchvision.datasets.FashionMNIST(
        root=DATA_ROOT, train=False, download=True, transform=to_array
    )

    x_train = train_set.data.numpy().reshape(-1, INPUT_DIM).astype(np.float32) / 255.0
    y_train = train_set.targets.numpy().astype(np.int64)
    x_test = test_set.data.numpy().reshape(-1, INPUT_DIM).astype(np.float32) / 255.0
    y_test = test_set.targets.numpy().astype(np.int64)
    return x_train, y_train, x_test, y_test


def load_fashion_mnist_loaders(
    batch_size: int = 128, num_workers: int = 0
) -> tuple["torch.utils.data.DataLoader", "torch.utils.data.DataLoader"]:
    """Load Fashion-MNIST as train/test DataLoaders for Phase 2+.

    Returns:
        (train_loader, test_loader) yielding float32 images shaped
        (N, 1, 28, 28) in [0, 1] and int64 labels.
    """
    import torchvision
    from torchvision import transforms

    transform = transforms.ToTensor()
    train_set = torchvision.datasets.FashionMNIST(
        root=DATA_ROOT, train=True, download=True, transform=transform
    )
    test_set = torchvision.datasets.FashionMNIST(
        root=DATA_ROOT, train=False, download=True, transform=transform
    )

    train_loader = torch_data_loader(train_set, batch_size, num_workers, shuffle=True)
    test_loader = torch_data_loader(test_set, batch_size, num_workers, shuffle=False)
    return train_loader, test_loader


def torch_data_loader(dataset, batch_size: int, num_workers: int, shuffle: bool):
    from torch.utils.data import DataLoader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def class_name(label: int) -> str:
    """Return the human-readable class name for a Fashion-MNIST label."""
    return _FASHION_CLASSES[label]