"""Build encoder and predictor from config."""

import torch.nn as nn

from tinyvllm.config import Config
from tinyvllm.models.encoder import ImageEncoder
from tinyvllm.models.predictor import PatchPredictor, Predictor
from tinyvllm.models.vit_encoder import ViTEncoder


def build_encoder(config: Config) -> nn.Module:
    if config.encoder == "cnn":
        return ImageEncoder(config.in_channels, config.embed_dim)
    return ViTEncoder(
        in_channels=config.in_channels,
        image_size=config.image_size,
        patch_size=config.vit_patch_size,
        embed_dim=config.embed_dim,
        depth=config.vit_depth,
        num_heads=config.vit_heads,
        mlp_ratio=config.vit_mlp_ratio,
    )


def build_predictor(config: Config) -> nn.Module:
    if config.jepa_mode == "patch":
        return PatchPredictor(config.embed_dim, config.predictor_hidden)
    return Predictor(config.embed_dim, config.predictor_hidden)
