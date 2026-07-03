"""Tests for JEPA loss."""

import torch
import torch.nn.functional as F

from tinyvllm.jepa.loss import jepa_loss


def test_loss_non_negative():
    pred = F.normalize(torch.randn(8, 128), dim=1)
    target = F.normalize(torch.randn(8, 128), dim=1)
    loss = jepa_loss(pred, target)
    assert loss.item() >= 0.0


def test_loss_zero_when_identical():
    z = F.normalize(torch.randn(4, 128), dim=1)
    loss = jepa_loss(z, z.clone())
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)


def test_stop_grad_target_does_not_block_pred_grads():
    from tinyvllm.models.predictor import Predictor

    predictor = Predictor(128, 256)
    z_corrupt = torch.randn(4, 128)
    target = torch.randn(4, 128)

    z_pred = predictor(z_corrupt)
    loss = jepa_loss(z_pred, target, stop_grad_target=True)
    loss.backward()

    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in predictor.parameters())
