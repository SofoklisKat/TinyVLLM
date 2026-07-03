"""Predictor MLP — maps corrupted embedding to predicted clean embedding."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Predictor(nn.Module):
    """Two-layer MLP in embedding space (not pixel space).

    This is the core JEPA idea: predict abstract representations, not raw inputs.
    """

    def __init__(self, embed_dim: int = 128, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(z), dim=1)
