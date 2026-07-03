"""Training and model configuration."""

from dataclasses import dataclass, field
from typing import Literal


DatasetName = Literal["mnist", "cifar10"]
CorruptionMode = Literal["noise", "blur", "mask", "mix"]


@dataclass
class Config:
    """All hyperparameters in one place for readability."""

    dataset: DatasetName = "mnist"
    embed_dim: int = 128
    predictor_hidden: int = 256
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 10
    corruption: CorruptionMode = "mix"
    checkpoint_dir: str = "checkpoints"
    num_workers: int = 0

    @property
    def in_channels(self) -> int:
        return 1 if self.dataset == "mnist" else 3

    @property
    def patch_size(self) -> int:
        """Patch size for random masking (dataset-dependent)."""
        return 7 if self.dataset == "mnist" else 8

    def checkpoint_path(self, epoch: int) -> str:
        return f"{self.checkpoint_dir}/{self.dataset}/epoch_{epoch}.pt"
