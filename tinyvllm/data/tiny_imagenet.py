"""Tiny ImageNet-200 loader (~237 MB download, 64×64 RGB, 200 classes)."""

from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

from PIL import Image
from torch.utils.data import Dataset

TINY_IMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"


def tiny_imagenet_dir(root: Path) -> Path:
    return root / "tiny-imagenet-200"


def ensure_tiny_imagenet(root: Path) -> Path:
    """Download and unzip Tiny ImageNet if missing."""
    folder = tiny_imagenet_dir(root)
    if folder.is_dir() and (folder / "train").is_dir():
        return folder

    root.mkdir(parents=True, exist_ok=True)
    zip_path = root / "tiny-imagenet-200.zip"
    if not zip_path.is_file():
        print(f"Downloading Tiny ImageNet (~237 MB) to {zip_path} ...")
        urlretrieve(TINY_IMAGENET_URL, zip_path)

    print(f"Extracting {zip_path} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    return folder


class _TinyImageNetBase(Dataset):
    def __init__(self, image_paths: list[Path], transform=None):
        self.samples = image_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        img = Image.open(self.samples[idx]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


def _collect_train_images(root: Path) -> list[Path]:
    paths: list[Path] = []
    for class_dir in sorted((root / "train").iterdir()):
        img_dir = class_dir / "images"
        paths.extend(sorted(img_dir.glob("*.JPEG")))
    return paths


def _collect_val_images(root: Path) -> list[Path]:
    ann_file = root / "val" / "val_annotations.txt"
    img_dir = root / "val" / "images"
    paths: list[Path] = []
    with ann_file.open() as f:
        for line in f:
            name = line.split("\t")[0]
            paths.append(img_dir / name)
    return paths


def get_tiny_imagenet(root: Path, train: bool, transform):
    folder = ensure_tiny_imagenet(root)
    if train:
        return _TinyImageNetBase(_collect_train_images(folder), transform)
    return _TinyImageNetBase(_collect_val_images(folder), transform)
