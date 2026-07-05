"""Dataset loaders — MNIST, CIFAR-10/100, ImageNet (ImageFolder)."""

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

DatasetName = Literal["mnist", "cifar10", "cifar100", "imagenet"]

DATASET_CHOICES = ["mnist", "cifar10", "cifar100", "imagenet"]


def default_image_size(dataset: DatasetName) -> int:
    """Sensible defaults: 32 for small images, 224 for ImageNet."""
    if dataset == "imagenet":
        return 224
    if dataset == "mnist":
        return 32
    return 32


def default_vit_patch_size(dataset: DatasetName) -> int:
    if dataset == "imagenet":
        return 16
    return 4


def _build_transform(image_size: int | None):
    """Scale to [0, 1] and optionally resize for ViT."""
    steps = []
    if image_size is not None:
        steps.append(transforms.Resize((image_size, image_size)))
    steps.append(transforms.ToTensor())
    return transforms.Compose(steps)


def _imagenet_folder(root: Path, train: bool) -> Path:
    """Standard layout: {root}/train/ and {root}/val/ with class subfolders."""
    folder = root / ("train" if train else "val")
    if not folder.is_dir():
        raise FileNotFoundError(
            f"ImageNet folder not found: {folder}\n"
            "Expected layout:\n"
            f"  {root}/train/<class_name>/*.JPEG\n"
            f"  {root}/val/<class_name>/*.JPEG\n"
            "Download ImageNet and pass --data-root /path/to/imagenet"
        )
    return folder


def get_dataloader(
    dataset: DatasetName,
    batch_size: int = 64,
    train: bool = True,
    num_workers: int = 0,
    data_root: str = "data",
    image_size: int | None = None,
) -> DataLoader:
    """Return a DataLoader for the requested dataset."""
    root = Path(data_root)
    transform = _build_transform(image_size)

    if dataset == "mnist":
        ds = datasets.MNIST(root=root, train=train, download=True, transform=transform)
    elif dataset == "cifar10":
        ds = datasets.CIFAR10(root=root, train=train, download=True, transform=transform)
    elif dataset == "cifar100":
        ds = datasets.CIFAR100(root=root, train=train, download=True, transform=transform)
    elif dataset == "imagenet":
        folder = _imagenet_folder(root, train=train)
        ds = datasets.ImageFolder(folder, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
