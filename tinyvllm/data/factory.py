"""Dataset loaders for MNIST and CIFAR-10."""

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DatasetName = Literal["mnist", "cifar10"]


def _build_transform(dataset: DatasetName, image_size: int | None):
    """Scale to [0, 1] and optionally resize for ViT."""
    steps = []
    if image_size is not None:
        steps.append(transforms.Resize((image_size, image_size)))
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)


def get_dataloader(
    dataset: DatasetName,
    batch_size: int = 64,
    train: bool = True,
    num_workers: int = 0,
    data_root: str = "data",
    image_size: int | None = None,
) -> DataLoader:
    """Return a DataLoader for MNIST or CIFAR-10."""
    root = Path(data_root)
    transform = _build_transform(dataset, image_size)

    if dataset == "mnist":
        ds = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transform,
        )
    elif dataset == "cifar10":
        ds = datasets.CIFAR10(
            root=root,
            train=train,
            download=True,
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
