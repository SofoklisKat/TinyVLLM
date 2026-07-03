"""Modality encoder protocol — extensibility hook for phase 2 (CLIP)."""

from typing import Protocol

import torch


class ModalityEncoder(Protocol):
    """Any encoder that maps raw input to a fixed-size embedding vector.

    Phase 2 will add TextEncoder implementing this protocol so image and text
    encoders can be swapped behind the same interface for contrastive training.
    """

    embed_dim: int

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized embeddings of shape (batch, embed_dim)."""
        ...
