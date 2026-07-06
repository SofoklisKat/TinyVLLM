"""Tests for Phase 2 CLIP components."""

import torch

from tinyvllm.data.char_tokenizer import decode, encode, vocab_size
from tinyvllm.data.labels import label_to_caption
from tinyvllm.jepa.loss import contrastive_loss
from tinyvllm.models.text_encoder import TextEncoder
from tinyvllm.config import ClipConfig


def test_char_tokenizer_roundtrip():
    text = "ankle boot"
    ids = encode(text, max_len=16)
    assert decode(ids).replace(" ", "") == text.replace(" ", "")


def test_fashion_mnist_caption():
    assert label_to_caption("fashion_mnist", 9) == "ankle boot"


def test_contrastive_loss_finite():
    img = torch.randn(8, 128)
    txt = torch.randn(8, 128)
    img = torch.nn.functional.normalize(img, dim=1)
    txt = torch.nn.functional.normalize(txt, dim=1)
    loss = contrastive_loss(img, txt)
    assert torch.isfinite(loss)


def test_text_encoder_shape():
    enc = TextEncoder(vocab_size=vocab_size(), embed_dim=128, max_len=32)
    tokens = torch.tensor([encode("sneaker", max_len=32), encode("coat", max_len=32)])
    out = enc(tokens)
    assert out.shape == (2, 128)
    assert torch.allclose(out.norm(dim=1), torch.ones(2), atol=1e-5)


def test_clip_train_step():
    from tinyvllm.config import Config
    from tinyvllm.models.factory import build_encoder

    cfg = Config(dataset="fashion_mnist", encoder="vit", image_size=32)
    img_enc = build_encoder(cfg)
    txt_enc = TextEncoder(vocab_size=vocab_size(), embed_dim=128)
    images = torch.rand(4, 1, 32, 32)
    tokens = torch.tensor([encode("sneaker", 32)] * 4)
    loss = contrastive_loss(img_enc(images), txt_enc(tokens))
    loss.backward()


def test_migrate_clip_config_from_dict():
    from tinyvllm.config import migrate_clip_config

    cfg = migrate_clip_config({"dataset": "fashion_mnist", "encoder": "vit", "embed_dim": 128})
    assert cfg.dataset == "fashion_mnist"
    assert cfg.embed_dim == 128


def test_legacy_clip_checkpoint_load(tmp_path):
    """Checkpoints pickled with ClipConfig under match_text still load."""
    import tinyvllm.inference.match_text as match_text_mod
    from tinyvllm.models.factory import build_encoder
    from tinyvllm.train_clip import load_clip_checkpoint

    config = ClipConfig(dataset="fashion_mnist", encoder="vit", image_size=32)

    jepa_cfg = config.to_jepa_config()
    image_encoder = build_encoder(jepa_cfg)
    text_encoder = TextEncoder(vocab_size=vocab_size(), embed_dim=config.embed_dim, max_len=config.max_text_len)

    path = tmp_path / "legacy_clip.pt"
    orig_module = ClipConfig.__module__
    ClipConfig.__module__ = "tinyvllm.inference.match_text"
    match_text_mod.ClipConfig = ClipConfig
    try:
        torch.save(
            {
                "epoch": 3,
                "image_encoder": image_encoder.state_dict(),
                "text_encoder": text_encoder.state_dict(),
                "config": config,
            },
            path,
        )
    finally:
        ClipConfig.__module__ = orig_module
        del match_text_mod.ClipConfig

    device = torch.device("cpu")
    loaded_img, loaded_txt, loaded_cfg, epoch = load_clip_checkpoint(str(path), device)
    assert epoch == 3
    assert loaded_cfg.dataset == "fashion_mnist"
    assert loaded_img.state_dict().keys() == image_encoder.state_dict().keys()
    assert loaded_txt.state_dict().keys() == text_encoder.state_dict().keys()
