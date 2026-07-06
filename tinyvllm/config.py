"""Training and model configuration."""

from dataclasses import dataclass
from typing import Literal

from tinyvllm.data.factory import DatasetName, default_image_size, default_vit_patch_size


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
    data_root: str = "data"
    num_workers: int = 0

    # ViT settings (used when encoder="vit")
    image_size: int = 32
    vit_patch_size: int = 4
    vit_depth: int = 4
    vit_heads: int = 4
    vit_mlp_ratio: int = 4

    @property
    def in_channels(self) -> int:
        if self.dataset in ("mnist", "fashion_mnist"):
            return 1
        return 3

    @property
    def mask_patch_size(self) -> int:
        """Patch size for ViewCorruptor spatial masking (pixels)."""
        if self.dataset in ("mnist", "fashion_mnist"):
            return 7
        if self.dataset == "imagenet":
            return 16
        if self.dataset == "tiny_imagenet":
            return 8
        return 8

    def checkpoint_path(self, epoch: int) -> str:
        tag = f"{self.encoder}_{self.jepa_mode}"
        return f"{self.checkpoint_dir}/{self.dataset}/{tag}/epoch_{epoch}.pt"


def apply_dataset_defaults(
    dataset: DatasetName,
    image_size: int | None,
    vit_patch_size: int | None,
) -> tuple[int, int]:
    """Fill image / ViT patch sizes when CLI leaves them unset."""
    resolved_image = image_size if image_size is not None else default_image_size(dataset)
    resolved_patch = vit_patch_size if vit_patch_size is not None else default_vit_patch_size(dataset)
    return resolved_image, resolved_patch


@dataclass
class ClipConfig:
    """CLIP stage config (image ↔ text alignment)."""

    dataset: str = "fashion_mnist"
    encoder: EncoderType = "vit"
    embed_dim: int = 128
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 10
    temperature: float = 0.07
    checkpoint_dir: str = "checkpoints"
    data_root: str = "data"
    num_workers: int = 0
    freeze_image_encoder: bool = False
    jepa_checkpoint: str | None = None
    image_size: int = 32
    vit_patch_size: int = 4
    vit_depth: int = 4
    vit_heads: int = 4
    vit_mlp_ratio: int = 4
    text_depth: int = 2
    text_heads: int = 4
    max_text_len: int = 32

    def to_jepa_config(self) -> Config:
        return Config(
            dataset=self.dataset,  # type: ignore[arg-type]
            encoder=self.encoder,
            jepa_mode="global",
            embed_dim=self.embed_dim,
            batch_size=self.batch_size,
            lr=self.lr,
            epochs=self.epochs,
            checkpoint_dir=self.checkpoint_dir,
            data_root=self.data_root,
            num_workers=self.num_workers,
            image_size=self.image_size,
            vit_patch_size=self.vit_patch_size,
            vit_depth=self.vit_depth,
            vit_heads=self.vit_heads,
            vit_mlp_ratio=self.vit_mlp_ratio,
        )

    def checkpoint_path(self, epoch: int) -> str:
        return f"{self.checkpoint_dir}/{self.dataset}/clip/epoch_{epoch}.pt"


def migrate_clip_config(raw) -> ClipConfig:
    """Rebuild ClipConfig from checkpoint (dict, dataclass, or legacy pickle)."""
    if isinstance(raw, ClipConfig):
        return raw
    fields = raw.__dict__ if hasattr(raw, "__dict__") else dict(raw)
    return ClipConfig(
        dataset=fields.get("dataset", "fashion_mnist"),
        encoder=fields.get("encoder", "vit"),
        embed_dim=fields.get("embed_dim", 128),
        batch_size=fields.get("batch_size", 64),
        lr=fields.get("lr", 1e-3),
        epochs=fields.get("epochs", 10),
        temperature=fields.get("temperature", 0.07),
        checkpoint_dir=fields.get("checkpoint_dir", "checkpoints"),
        data_root=fields.get("data_root", "data"),
        num_workers=fields.get("num_workers", 0),
        freeze_image_encoder=fields.get("freeze_image_encoder", False),
        jepa_checkpoint=fields.get("jepa_checkpoint"),
        image_size=fields.get("image_size", 32),
        vit_patch_size=fields.get("vit_patch_size", 4),
        vit_depth=fields.get("vit_depth", 4),
        vit_heads=fields.get("vit_heads", 4),
        vit_mlp_ratio=fields.get("vit_mlp_ratio", 4),
        text_depth=fields.get("text_depth", 2),
        text_heads=fields.get("text_heads", 4),
        max_text_len=fields.get("max_text_len", 32),
    )
