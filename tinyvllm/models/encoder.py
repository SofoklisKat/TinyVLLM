"""Image encoder (CNN) and TextEncoder stub for phase 2."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageEncoder(nn.Module):
    """Small CNN: conv blocks → global average pool → L2-normalized embedding.

    Not optimized — each block is explicit so you can trace the spatial
    downsampling and channel growth step by step.
    """

    def __init__(self, in_channels: int = 1, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim

        # Three stride-2 conv blocks halve spatial size each time.
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x: torch.Tensor)  -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x).flatten(1)  # (batch, embed_dim)
        return F.normalize(x, dim=1)


class TextEncoder(nn.Module):
    """Phase 2 stub — CLIP-style text tower (not implemented in v1)."""

    def __init__(self, vocab_size: int = 256, embed_dim: int = 128):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "TextEncoder is a phase 2 stub. See README roadmap for CLIP bridge."
        )
