"""Multi-head attention implemented from scratch (educational).

This script shows how nn.MultiheadAttention works internally:
  1. Linear projections → Q, K, V
  2. Split into multiple heads
  3. Scaled dot-product attention per head
  4. Concatenate heads → output projection

Run:
    uv run python examples/multihead_attention_from_scratch.py
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    *,
    attn_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V.

    Args:
        query: (B, H, N, D_head)
        key:   (B, H, N, D_head)
        value: (B, H, N, D_head)
        attn_mask: optional (N, N) or broadcastable mask; -inf blocks attention

    Returns:
        output: (B, H, N, D_head)
        weights: (B, H, N, N) attention weights (sum to 1 over last dim)
    """
    d_head = query.size(-1)
    # Raw similarity scores between every query token and every key token.
    scores = query @ key.transpose(-2, -1) / math.sqrt(d_head)

    if attn_mask is not None:
        scores = scores + attn_mask

    weights = F.softmax(scores, dim=-1)
    output = weights @ value
    return output, weights


class MultiheadAttentionFromScratch(nn.Module):
    """Self-attention with multiple parallel heads — no nn.MultiheadAttention.

    Same interface spirit as PyTorch's module when batch_first=True:
        forward(x, x, x) for self-attention on (B, N, D).
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        *,
        batch_first: bool = True,
        bias: bool = True,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first

        # One big linear layer is equivalent to separate Q, K, V projections.
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _reshape_to_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, N, D) → (B, H, N, D_head)."""
        batch, n_tokens, _ = x.shape
        x = x.view(batch, n_tokens, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """(B, H, N, D_head) → (B, N, D)."""
        batch, _, n_tokens, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(batch, n_tokens, self.embed_dim)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        attn_mask: torch.Tensor | None = None,
        need_weights: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if not self.batch_first:
            # PyTorch legacy layout (N, B, D) — convert to batch-first internally.
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        # Step 1: project inputs
        q = self._reshape_to_heads(self.q_proj(query))
        k = self._reshape_to_heads(self.k_proj(key))
        v = self._reshape_to_heads(self.v_proj(value))

        # Step 2: attention per head
        attn_out, weights = scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

        # Step 3: merge heads + final linear
        merged = self._merge_heads(attn_out)
        output = self.out_proj(merged)

        if not self.batch_first:
            output = output.transpose(0, 1)

        if need_weights:
            return output, weights
        return output, None


def demo() -> None:
    """Compare scratch implementation vs PyTorch on random data."""
    torch.manual_seed(0)

    batch, n_patches, dim, heads = 2, 8, 128, 4
    x = torch.randn(batch, n_patches, dim)

    scratch = MultiheadAttentionFromScratch(dim, heads, batch_first=True)
    builtin = nn.MultiheadAttention(dim, heads, batch_first=True)

    # Copy weights so both modules do the same computation (fair comparison).
    with torch.no_grad():
        # PyTorch packs Q,K,V into one in_proj_weight; we use separate layers.
        # For demo, just show shapes — not weight alignment.
        pass

    out_scratch, w = scratch(x, x, x, need_weights=True)
    out_builtin, _ = builtin(x, x, x, need_weights=False)

    print("Input shape:          ", tuple(x.shape))
    print("Output shape:         ", tuple(out_scratch.shape))
    print("Attention weights:    ", tuple(w.shape), "  # (B, heads, N, N)")
    print()
    print("Example attention weights for batch 0, head 0 (patch → patch):")
    print(w[0, 0].round(decimals=3))
    print()
    print("Each row sums to 1.0 — a probability distribution over patches.")
    print("Scratch vs builtin max diff (random init):",
          (out_scratch - out_builtin).abs().max().item())


if __name__ == "__main__":
    demo()
