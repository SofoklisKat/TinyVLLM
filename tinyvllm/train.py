"""JEPA training loop — educational, not optimized."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from tinyvllm.config import Config
from tinyvllm.data.factory import get_dataloader
from tinyvllm.jepa.loss import jepa_loss, jepa_patch_loss
from tinyvllm.jepa.views import ViewCorruptor
from tinyvllm.models.factory import build_encoder, build_predictor


def compute_jepa_loss(
    encoder: nn.Module,
    predictor: nn.Module,
    clean: torch.Tensor,
    corrupt: torch.Tensor,
    jepa_mode: str,
) -> torch.Tensor:
    """Single JEPA forward — global (CNN/ViT) or patch-level (ViT only)."""
    if jepa_mode == "patch":
        patch_clean = encoder.forward_patches(clean).detach()
        patch_corrupt = encoder.forward_patches(corrupt)
        patch_pred = predictor(patch_corrupt)
        return jepa_patch_loss(patch_pred, patch_clean)

    z_pred = predictor(encoder(corrupt))
    z_clean = encoder(clean).detach()
    return jepa_loss(z_pred, z_clean)


def train_one_epoch(
    encoder: nn.Module,
    predictor: nn.Module,
    corruptor: ViewCorruptor,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    jepa_mode: str,
) -> float:
    encoder.train()
    predictor.train()
    total_loss = 0.0
    n_batches = 0

    for images, _labels in loader:
        clean = images.to(device)
        corrupt = corruptor(clean).to(device)

        loss = compute_jepa_loss(encoder, predictor, clean, corrupt, jepa_mode)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def save_checkpoint(
    path: str,
    encoder: nn.Module,
    predictor: nn.Module,
    config: Config,
    epoch: int,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "config": config,
        },
        path,
    )


def load_checkpoint(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = _migrate_config(ckpt["config"])
    encoder = build_encoder(config).to(device)
    predictor = build_predictor(config).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    predictor.load_state_dict(ckpt["predictor"])
    return encoder, predictor, config, ckpt.get("epoch", 0)


def _migrate_config(raw: Config) -> Config:
    """Fill defaults for checkpoints saved before ViT support."""
    fields = raw.__dict__ if isinstance(raw, Config) else dict(raw)
    return Config(
        dataset=fields.get("dataset", "mnist"),
        encoder=fields.get("encoder", "cnn"),
        jepa_mode=fields.get("jepa_mode", "global"),
        embed_dim=fields.get("embed_dim", 128),
        predictor_hidden=fields.get("predictor_hidden", 256),
        batch_size=fields.get("batch_size", 64),
        lr=fields.get("lr", 1e-3),
        epochs=fields.get("epochs", 10),
        corruption=fields.get("corruption", "mix"),
        checkpoint_dir=fields.get("checkpoint_dir", "checkpoints"),
        num_workers=fields.get("num_workers", 0),
        image_size=fields.get("image_size", 32),
        vit_patch_size=fields.get("vit_patch_size", 4),
        vit_depth=fields.get("vit_depth", 4),
        vit_heads=fields.get("vit_heads", 4),
        vit_mlp_ratio=fields.get("vit_mlp_ratio", 4),
    )


def build_models(config: Config, device: torch.device):
    encoder = build_encoder(config).to(device)
    predictor = build_predictor(config).to(device)
    return encoder, predictor


def validate_config(config: Config) -> None:
    if config.jepa_mode == "patch" and config.encoder != "vit":
        raise ValueError("Patch-level JEPA requires --encoder vit")
    if config.encoder == "vit" and config.image_size % config.vit_patch_size != 0:
        raise ValueError("image_size must be divisible by vit_patch_size")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinyVLLM image JEPA")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--encoder", choices=["cnn", "vit"], default="cnn")
    parser.add_argument("--jepa-mode", choices=["global", "patch"], default="global")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--vit-patch-size", type=int, default=4)
    parser.add_argument("--vit-depth", type=int, default=4)
    parser.add_argument("--vit-heads", type=int, default=4)
    parser.add_argument(
        "--corruption",
        choices=["noise", "blur", "mask", "mix"],
        default="mix",
    )
    args = parser.parse_args()

    config = Config(
        dataset=args.dataset,
        encoder=args.encoder,
        jepa_mode=args.jepa_mode,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        corruption=args.corruption,
        image_size=args.image_size,
        vit_patch_size=args.vit_patch_size,
        vit_depth=args.vit_depth,
        vit_heads=args.vit_heads,
    )
    validate_config(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_patches = (config.image_size // config.vit_patch_size) ** 2 if config.encoder == "vit" else "-"
    print(
        f"Device: {device} | Dataset: {config.dataset} | "
        f"Encoder: {config.encoder} | JEPA: {config.jepa_mode} | "
        f"Image: {config.image_size} | Patches: {n_patches}"
    )

    encoder, predictor = build_models(config, device)
    corruptor = ViewCorruptor(mode=config.corruption, patch_size=config.mask_patch_size)
    image_size = config.image_size if config.encoder == "vit" else None
    loader = get_dataloader(
        config.dataset,
        batch_size=config.batch_size,
        train=True,
        num_workers=config.num_workers,
        image_size=image_size,
    )
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=config.lr,
    )

    # Optional EMA target encoder (production JEPA pattern) — not enabled in v1:
    #
    # target_encoder = copy.deepcopy(encoder)
    # for p in target_encoder.parameters():
    #     p.requires_grad = False
    # ... update EMA after each step: τ * target + (1-τ) * encoder

    for epoch in range(1, config.epochs + 1):
        avg_loss = train_one_epoch(
            encoder, predictor, corruptor, loader, optimizer, device, config.jepa_mode
        )
        ckpt_path = config.checkpoint_path(epoch)
        save_checkpoint(ckpt_path, encoder, predictor, config, epoch)
        print(f"Epoch {epoch}/{config.epochs}  loss={avg_loss:.4f}  saved={ckpt_path}")


if __name__ == "__main__":
    main()
