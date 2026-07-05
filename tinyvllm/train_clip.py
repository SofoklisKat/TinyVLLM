"""Phase 2 — CLIP training: align image encoder with text (class-name captions)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn

from tinyvllm.config import Config, apply_dataset_defaults
from tinyvllm.data.char_tokenizer import encode, vocab_size
from tinyvllm.data.factory import DATASET_CHOICES, get_dataloader
from tinyvllm.data.labels import get_label_names, label_to_caption
from tinyvllm.jepa.loss import contrastive_loss
from tinyvllm.models.factory import build_encoder
from tinyvllm.models.text_encoder import TextEncoder
from tinyvllm.train import _migrate_config, validate_config

CLIP_DATASETS = ["fashion_mnist", "cifar10", "cifar100", "mnist"]


@dataclass
class ClipConfig:
    """CLIP stage config (extends JEPA image settings)."""

    dataset: str = "fashion_mnist"
    encoder: Literal["cnn", "vit"] = "vit"
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


def labels_to_tokens(labels: torch.Tensor, dataset: str, max_len: int, device) -> torch.Tensor:
    """Convert batch of class indices → char token tensor."""
    rows = [encode(label_to_caption(dataset, int(y.item())), max_len=max_len) for y in labels]
    return torch.tensor(rows, dtype=torch.long, device=device)


def load_jepa_image_encoder(path: str, device: torch.device) -> tuple[nn.Module, Config]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = _migrate_config(ckpt["config"])
    validate_config(config)
    encoder = build_encoder(config).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    return encoder, config


def train_one_epoch(
    image_encoder: nn.Module,
    text_encoder: nn.Module,
    loader,
    optimizer,
    device: torch.device,
    dataset: str,
    max_text_len: int,
    temperature: float,
) -> float:
    image_encoder.train()
    text_encoder.train()
    total = 0.0
    n = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        tokens = labels_to_tokens(labels, dataset, max_text_len, device)

        img_emb = image_encoder(images)
        txt_emb = text_encoder(tokens)
        loss = contrastive_loss(img_emb, txt_emb, temperature=temperature)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total += loss.item()
        n += 1

    return total / max(n, 1)


def save_clip_checkpoint(path: str, image_encoder, text_encoder, config: ClipConfig, epoch: int):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "image_encoder": image_encoder.state_dict(),
            "text_encoder": text_encoder.state_dict(),
            "config": config,
        },
        path,
    )


def load_clip_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config: ClipConfig = ckpt["config"]
    jepa_cfg = config.to_jepa_config()
    image_encoder = build_encoder(jepa_cfg).to(device)
    text_encoder = TextEncoder(
        vocab_size=vocab_size(),
        embed_dim=config.embed_dim,
        max_len=config.max_text_len,
        depth=config.text_depth,
        num_heads=config.text_heads,
    ).to(device)
    image_encoder.load_state_dict(ckpt["image_encoder"])
    text_encoder.load_state_dict(ckpt["text_encoder"])
    return image_encoder, text_encoder, config, ckpt.get("epoch", 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CLIP alignment (image ↔ text)")
    parser.add_argument("--dataset", choices=CLIP_DATASETS, default="fashion_mnist")
    parser.add_argument("--encoder", choices=["cnn", "vit"], default="vit")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--vit-patch-size", type=int, default=None)
    parser.add_argument("--vit-depth", type=int, default=4)
    parser.add_argument("--vit-heads", type=int, default=4)
    parser.add_argument(
        "--jepa-checkpoint",
        default=None,
        help="Optional JEPA checkpoint to initialize (and optionally freeze) image encoder",
    )
    parser.add_argument(
        "--freeze-image-encoder",
        action="store_true",
        help="Keep image encoder weights fixed during CLIP training",
    )
    args = parser.parse_args()

    image_size, vit_patch_size = apply_dataset_defaults(
        args.dataset, args.image_size, args.vit_patch_size
    )

    config = ClipConfig(
        dataset=args.dataset,
        encoder=args.encoder,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        temperature=args.temperature,
        data_root=args.data_root,
        image_size=image_size,
        vit_patch_size=vit_patch_size,
        vit_depth=args.vit_depth,
        vit_heads=args.vit_heads,
        freeze_image_encoder=args.freeze_image_encoder,
        jepa_checkpoint=args.jepa_checkpoint,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    labels = get_label_names(args.dataset)  # noqa: validate captions exist
    print(f"Device: {device} | CLIP | Dataset: {args.dataset} | Classes: {len(labels)}")
    print(f"Example captions: {labels[:3]} ...")

    jepa_cfg = config.to_jepa_config()
    validate_config(jepa_cfg)

    if args.jepa_checkpoint:
        image_encoder, jepa_cfg = load_jepa_image_encoder(args.jepa_checkpoint, device)
        # Sync image settings from JEPA checkpoint.
        config.image_size = jepa_cfg.image_size
        config.vit_patch_size = jepa_cfg.vit_patch_size
        config.vit_depth = jepa_cfg.vit_depth
        config.vit_heads = jepa_cfg.vit_heads
        print(f"Loaded JEPA image encoder from {args.jepa_checkpoint}")
    else:
        image_encoder = build_encoder(jepa_cfg).to(device)

    if config.freeze_image_encoder:
        for p in image_encoder.parameters():
            p.requires_grad = False
        image_encoder.eval()
        print("Image encoder frozen")

    text_encoder = TextEncoder(
        vocab_size=vocab_size(),
        embed_dim=config.embed_dim,
        max_len=config.max_text_len,
        depth=config.text_depth,
        num_heads=config.text_heads,
    ).to(device)

    img_size = config.image_size if config.encoder == "vit" else None
    loader = get_dataloader(
        config.dataset,  # type: ignore[arg-type]
        batch_size=config.batch_size,
        train=True,
        num_workers=config.num_workers,
        data_root=config.data_root,
        image_size=img_size,
    )

    params = list(text_encoder.parameters())
    if not config.freeze_image_encoder:
        params += list(image_encoder.parameters())
    optimizer = torch.optim.Adam(params, lr=config.lr)

    for epoch in range(1, config.epochs + 1):
        avg = train_one_epoch(
            image_encoder,
            text_encoder,
            loader,
            optimizer,
            device,
            config.dataset,
            config.max_text_len,
            config.temperature,
        )
        path = config.checkpoint_path(epoch)
        save_clip_checkpoint(path, image_encoder, text_encoder, config, epoch)
        print(f"Epoch {epoch}/{config.epochs}  clip_loss={avg:.4f}  saved={path}")


if __name__ == "__main__":
    main()
