"""Tests for Phase 2 CLIP components."""

import torch

from tinyvllm.data.char_tokenizer import decode, encode, vocab_size
from tinyvllm.data.labels import label_to_caption
from tinyvllm.jepa.loss import contrastive_loss
from tinyvllm.models.text_encoder import TextEncoder


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
