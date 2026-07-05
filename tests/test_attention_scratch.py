"""Tests for from-scratch multi-head attention."""

import torch

from examples.multihead_attention_from_scratch import (
    MultiheadAttentionFromScratch,
    scaled_dot_product_attention,
)


def test_output_shape():
    attn = MultiheadAttentionFromScratch(128, 4, batch_first=True)
    x = torch.randn(2, 16, 128)
    out, _ = attn(x, x, x)
    assert out.shape == (2, 16, 128)


def test_attention_weights_sum_to_one():
    q = k = v = torch.randn(1, 2, 4, 8)
    _, weights = scaled_dot_product_attention(q, k, v)
    row_sums = weights.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5)


def test_gradients_flow():
    attn = MultiheadAttentionFromScratch(64, 2, batch_first=True)
    x = torch.randn(1, 8, 64, requires_grad=True)
    out, _ = attn(x, x, x)
    out.sum().backward()
    assert x.grad is not None
