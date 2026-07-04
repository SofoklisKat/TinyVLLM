"""Small ViT encoder with self-attention — educational, consumer-GPU friendly."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    """Pre-norm transformer block (readable stack of attention + MLP)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = dim * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class ViTEncoder(nn.Module):
    """Patchify → positional embed → transformer blocks → patch tokens.

    forward() returns a global embedding (mean pool over patches).
    forward_patches() returns per-patch embeddings for patch-level JEPA.
    """

    def __init__(
        self,
        in_channels: int = 3,
        image_size: int = 32,
        patch_size: int = 4,
        embed_dim: int = 128,
        depth: int = 4,
        num_heads: int = 4,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.n_patches = self.grid_size**2

        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        self.blocks = nn.ModuleList(
            TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized patch tokens, shape (B, N, D)."""
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return F.normalize(x, dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Global embedding via mean pool over patch tokens."""
        patches = self.forward_patches(x)
        global_emb = patches.mean(dim=1)
        return F.normalize(global_emb, dim=-1)
