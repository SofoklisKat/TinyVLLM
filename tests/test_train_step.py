"""One training step smoke test (synthetic batch, no dataset download)."""

import torch

from tinyvllm.config import Config
from tinyvllm.jepa.loss import jepa_loss
from tinyvllm.jepa.views import ViewCorruptor
from tinyvllm.models.encoder import ImageEncoder
from tinyvllm.models.predictor import Predictor


def test_one_train_step_mnist_shapes():
    config = Config(dataset="mnist")
    device = torch.device("cpu")

    encoder = ImageEncoder(config.in_channels, config.embed_dim).to(device)
    predictor = Predictor(config.embed_dim, config.predictor_hidden).to(device)
    corruptor = ViewCorruptor(mode="mix", patch_size=config.mask_patch_size)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=config.lr,
    )

    clean = torch.rand(8, 1, 28, 28)
    corrupt = corruptor(clean)

    z_pred = predictor(encoder(corrupt))
    z_clean = encoder(clean).detach()
    loss = jepa_loss(z_pred, z_clean)

    assert torch.isfinite(loss)
    assert loss.ndim == 0

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def test_one_train_step_cifar_shapes():
    config = Config(dataset="cifar10")
    encoder = ImageEncoder(config.in_channels, config.embed_dim)
    predictor = Predictor(config.embed_dim, config.predictor_hidden)
    corruptor = ViewCorruptor(mode="mix", patch_size=config.mask_patch_size)

    clean = torch.rand(4, 3, 32, 32)
    corrupt = corruptor(clean)

    z_pred = predictor(encoder(corrupt))
    z_clean = encoder(clean).detach()
    loss = jepa_loss(z_pred, z_clean)

    assert torch.isfinite(loss)
