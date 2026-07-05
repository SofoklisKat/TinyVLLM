"""Tests for dataset helpers."""

import pytest

from tinyvllm.config import apply_dataset_defaults
from tinyvllm.data.factory import default_image_size, default_vit_patch_size


def test_default_image_sizes():
    assert default_image_size("fashion_mnist") == 32
    assert default_image_size("cifar100") == 32
    assert default_image_size("tiny_imagenet") == 64
    assert default_image_size("imagenet") == 224


def test_default_vit_patch_sizes():
    assert default_vit_patch_size("cifar10") == 4
    assert default_vit_patch_size("tiny_imagenet") == 8
    assert default_vit_patch_size("imagenet") == 16


def test_apply_dataset_defaults_auto():
    image, patch = apply_dataset_defaults("imagenet", None, None)
    assert image == 224
    assert patch == 16


def test_apply_dataset_defaults_override():
    image, patch = apply_dataset_defaults("cifar100", 64, 8)
    assert image == 64
    assert patch == 8


def test_imagenet_missing_folder():
    from tinyvllm.data.factory import get_dataloader

    with pytest.raises(FileNotFoundError, match="ImageNet folder not found"):
        get_dataloader(
            "imagenet",
            batch_size=2,
            train=True,
            data_root="/nonexistent/imagenet/path",
            image_size=224,
        )
