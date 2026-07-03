"""Dataset loaders for MNIST and CIFAR-10."""

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DatasetName = Literal["mnist", "cifar10"]


def _normalize(dataset: DatasetName):
    """Scale pixel values to [0, 1] — matches ViewCorruptor clamp range."""
    if dataset == "mnist":
        return transforms.Compose([
            transforms.ToTensor(),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
    ])


def get_dataloader(
    dataset: DatasetName,
    batch_size: int = 64,
    train: bool = True,
    num_workers: int = 0,
    data_root: str = "data",
) -> DataLoader:
    """Return a DataLoader for MNIST or CIFAR-10."""
    root = Path(data_root)
    transform = _normalize(dataset)

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
