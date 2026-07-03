"""JEPA training loop — educational, not optimized."""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from tinyvllm.config import Config
from tinyvllm.data.factory import get_dataloader
from tinyvllm.jepa.loss import jepa_loss
from tinyvllm.jepa.views import ViewCorruptor
from tinyvllm.models.encoder import ImageEncoder
from tinyvllm.models.predictor import Predictor


def train_one_epoch(
    encoder: nn.Module,
    predictor: nn.Module,
    corruptor: ViewCorruptor,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    encoder.train()
    predictor.train()
    total_loss = 0.0
    n_batches = 0

    for images, _labels in loader:
        clean = images.to(device)
        corrupt = corruptor(clean).to(device)

        # Corrupted path — gradients flow through encoder + predictor.
        z_corrupt = encoder(corrupt)
        z_pred = predictor(z_corrupt)

        # Clean path — stop-grad target (see spec: anti-collapse).
        z_clean = encoder(clean).detach()

        loss = jepa_loss(z_pred, z_clean)

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
    config: Config = ckpt["config"]
    encoder = ImageEncoder(config.in_channels, config.embed_dim).to(device)
    predictor = Predictor(config.embed_dim, config.predictor_hidden).to(device)
    encoder.load_state_dict(ckpt["encoder"])
    predictor.load_state_dict(ckpt["predictor"])
    return encoder, predictor, config, ckpt.get("epoch", 0)


def build_models(config: Config, device: torch.device):
    encoder = ImageEncoder(config.in_channels, config.embed_dim).to(device)
    predictor = Predictor(config.embed_dim, config.predictor_hidden).to(device)
    return encoder, predictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Train TinyVLLM image JEPA")
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--corruption",
        choices=["noise", "blur", "mask", "mix"],
        default="mix",
    )
    args = parser.parse_args()

    config = Config(
        dataset=args.dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        corruption=args.corruption,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Dataset: {config.dataset}")

    encoder, predictor = build_models(config, device)
    corruptor = ViewCorruptor(mode=config.corruption, patch_size=config.patch_size)
    loader = get_dataloader(
        config.dataset,
        batch_size=config.batch_size,
        train=True,
        num_workers=config.num_workers,
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
        avg_loss = train_one_epoch(encoder, predictor, corruptor, loader, optimizer, device)
        ckpt_path = config.checkpoint_path(epoch)
        save_checkpoint(ckpt_path, encoder, predictor, config, epoch)
        print(f"Epoch {epoch}/{config.epochs}  loss={avg_loss:.4f}  saved={ckpt_path}")


if __name__ == "__main__":
    main()
