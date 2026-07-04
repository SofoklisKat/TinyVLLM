"""Training and model configuration."""

from dataclasses import dataclass
from typing import Literal


DatasetName = Literal["mnist", "cifar10"]
CorruptionMode = Literal["noise", "blur", "mask", "mix"]
EncoderType = Literal["cnn", "vit"]
JepaMode = Literal["global", "patch"]


@dataclass
class Config:
    """All hyperparameters in one place for readability."""

    dataset: DatasetName = "mnist"
    encoder: EncoderType = "cnn"
    jepa_mode: JepaMode = "global"
    embed_dim: int = 128
    predictor_hidden: int = 256
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 10
    corruption: CorruptionMode = "mix"
    checkpoint_dir: str = "checkpoints"
    num_workers: int = 0

    # ViT settings (used when encoder="vit")
    image_size: int = 32
    vit_patch_size: int = 4
    vit_depth: int = 4
    vit_heads: int = 4
    vit_mlp_ratio: int = 4

    @property
    def in_channels(self) -> int:
        return 1 if self.dataset == "mnist" else 3

    @property
    def mask_patch_size(self) -> int:
        """Patch size for ViewCorruptor spatial masking (pixels)."""
        return 7 if self.dataset == "mnist" else 8

    def checkpoint_path(self, epoch: int) -> str:
        tag = f"{self.encoder}_{self.jepa_mode}"
        return f"{self.checkpoint_dir}/{self.dataset}/{tag}/epoch_{epoch}.pt"
