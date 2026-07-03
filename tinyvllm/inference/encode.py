"""Encode a single image to a JEPA embedding vector."""

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from tinyvllm.train import load_checkpoint


def load_image(path: str, in_channels: int) -> torch.Tensor:
    img = Image.open(path)
    if in_channels == 1:
        img = img.convert("L")
        t = transforms.ToTensor()
    else:
        img = img.convert("RGB")
        t = transforms.ToTensor()
    return t(img).unsqueeze(0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Encode image to JEPA embedding")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, _predictor, config, epoch = load_checkpoint(args.checkpoint, device)
    encoder.eval()

    x = load_image(args.image, config.in_channels).to(device)

    # Educational inference loop — one image at a time.
    # A production vLLM-style engine would batch requests, reuse weights
    # on GPU, and pipeline pre/post-processing across workers.
    with torch.no_grad():
        emb = encoder(x)

    print(f"Checkpoint epoch: {epoch}")
    print(f"Embedding shape: {tuple(emb.shape)}")
    print(f"Embedding (first 8 dims): {emb[0, :8].tolist()}")
    print(f"L2 norm: {emb.norm(dim=1).item():.4f}  (should be ~1.0)")


if __name__ == "__main__":
    main()
