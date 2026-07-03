"""Tests for ViewCorruptor."""

import torch

from tinyvllm.jepa.views import ViewCorruptor


def test_corruptor_preserves_shape():
    clean = torch.rand(4, 1, 28, 28)
    corruptor = ViewCorruptor(mode="noise", patch_size=7)
    out = corruptor(clean)
    assert out.shape == clean.shape


def test_corruptor_clamps_to_unit_range():
    clean = torch.rand(2, 3, 32, 32)
    for mode in ("noise", "blur", "mask", "mix"):
        out = ViewCorruptor(mode=mode, patch_size=8)(clean)
        assert out.min() >= 0.0
        assert out.max() <= 1.0


def test_corruption_changes_input():
    clean = torch.ones(1, 1, 28, 28)
    out = ViewCorruptor(mode="noise", patch_size=7)(clean)
    assert not torch.allclose(out, clean)
