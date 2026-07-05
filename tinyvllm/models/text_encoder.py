"""Text encoder — char tokens → shared embedding space (CLIP text tower)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """Char embedding + small transformer → L2-normalized 128-d vector.

    Educational CLIP text side: same output shape as ViTEncoder global embedding.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        max_len: int = 32,
        depth: int = 2,
        num_heads: int = 4,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.token_embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_len, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: (B, max_len) → embeddings (B, embed_dim)."""
        x = self.token_embed(token_ids) + self.pos_embed
        # Ignore padding positions in attention.
        pad_mask = token_ids == 0
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        x = self.norm(x)
        # Mean pool over non-pad tokens.
        mask = (~pad_mask).unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return F.normalize(x, dim=1)
