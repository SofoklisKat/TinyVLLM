"""Tests for ViT encoder and patch-level JEPA."""

import torch

from tinyvllm.config import Config
from tinyvllm.jepa.loss import jepa_patch_loss
from tinyvllm.jepa.views import ViewCorruptor
from tinyvllm.models.factory import build_encoder, build_predictor
from tinyvllm.models.vit_encoder import ViTEncoder
from tinyvllm.train import compute_jepa_loss


def test_vit_encoder_shapes():
    encoder = ViTEncoder(in_channels=3, image_size=32, patch_size=4, embed_dim=128, depth=2, num_heads=4)
    x = torch.rand(2, 3, 32, 32)
    patches = encoder.forward_patches(x)
    global_emb = encoder(x)
    assert patches.shape == (2, 64, 128)
    assert global_emb.shape == (2, 128)


def test_vit_patch_jepa_train_step():
    config = Config(dataset="cifar10", encoder="vit", jepa_mode="patch", image_size=32)
    encoder = build_encoder(config)
    predictor = build_predictor(config)
    corruptor = ViewCorruptor(mode="mix", patch_size=config.mask_patch_size)

    clean = torch.rand(4, 3, 32, 32)
    corrupt = corruptor(clean)
    loss = compute_jepa_loss(encoder, predictor, clean, corrupt, config.jepa_mode)

    assert torch.isfinite(loss)
    loss.backward()


def test_jepa_patch_loss_zero_when_identical():
    z = torch.randn(2, 16, 128)
    z = torch.nn.functional.normalize(z, dim=-1)
    loss = jepa_patch_loss(z, z.clone())
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)
