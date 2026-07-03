# TinyVLLM — Image JEPA Design Spec

**Date:** 2026-07-03  
**Status:** Approved  
**Goal:** Educational, unoptimized codebase for learning vision JEPA end-to-end, with a clear path to CLIP-style image→text in phase 2.

---

## Summary

TinyVLLM v1 trains a small CNN encoder with a JEPA objective on images: given a **corrupted** view of an image, a predictor network learns to match the **clean** image's embedding in latent space. No pixel reconstruction (MAE-style) in v1. Supports MNIST and CIFAR-10 via config. Inference v1 exposes embeddings and probe utilities, not text generation.

---

## Requirements

### In scope (v1)

- Image-only JEPA training on corrupted→clean view pairs
- MNIST (28×28 grayscale) and CIFAR-10 (32×32 RGB) via `--dataset mnist|cifar10`
- ~1M parameter small CNN encoder, CPU-friendly, trains in minutes
- Explicit modular code: encoder, predictor, corruptor, loss — readable over fast
- Minimal inference: encode image → embedding; probe script for cosine-similarity stats
- Stub interfaces for phase 2 (TextEncoder, ContrastiveLoss) with TODO comments
- Basic tests: corruption, loss, one training step

### Out of scope (v1)

- MAE / pixel reconstruction
- Text input, tokenization, or generation
- CLIP contrastive training (phase 2)
- GPU kernels, FlashAttention, production vLLM features
- Large datasets, ViT-scale models

### Phase 2 (designed in, not built)

- `TextEncoder` + InfoNCE contrastive loss (CLIP-style)
- `TokenDecoder`: shared embedding → caption tokens
- Trivial caption pairs: MNIST digit names, CIFAR-10 class labels

---

## Architecture

```
Corrupted image ──► ImageEncoder ──► Predictor ──► predicted embedding
                                                        │
                                              cosine loss
                                                        │
Clean image ──► ImageEncoder (stop-grad) ──► target embedding
```

### Modules

| Module | File | Role |
|--------|------|------|
| `ModalityEncoder` | `models/base.py` | Protocol: `forward(x) -> embedding`. Extensibility hook for phase 2. |
| `ImageEncoder` | `models/encoder.py` | Small CNN: conv blocks → global avg pool → 128-d L2-normalized embedding |
| `Predictor` | `models/predictor.py` | 2-layer MLP (128→256→128) with ReLU |
| `ViewCorruptor` | `jepa/views.py` | Applies one of: gaussian noise, gaussian blur, random patch mask, or random mix |
| `jepa_loss` | `jepa/loss.py` | `1 - cosine_sim(pred, target)` with stop-gradient on target |

### Model dimensions

- Embedding dim: **128**
- CNN: 3 conv blocks (channels 32→64→128), kernel 3, stride 2, padding 1
- Input: 1 channel (MNIST) or 3 channels (CIFAR-10) — handled by config
- Predictor hidden dim: 256

### Anti-collapse

- Stop-gradient on clean encoder output (required)
- Optional EMA target encoder: implemented as commented code path, not default

#### Why stop-grad on the clean encoder?

Both views pass through the **same** `ImageEncoder` weights, but the clean path is detached from the autograd graph:

```python
z_pred   = predictor(encoder(corrupt))          # gradients flow
z_target = encoder(clean).detach()              # stop-grad — fixed target
loss = 1 - cosine_sim(z_pred, z_target)
```

Without stop-grad, encoder and predictor can **collapse** — push all embeddings toward a constant vector so cosine loss ≈ 0 without learning useful structure. Stop-grad makes the clean embedding a fixed target each step; only the corrupted path + predictor receive gradients from `L_jepa`.

The encoder still learns because every batch also runs the corrupted image through it (with gradients). The clean forward pass is only used to produce targets, not to update weights via the JEPA loss.

**Optional upgrade (commented in code):** an EMA copy of the encoder as a dedicated target network (closer to production JEPA). v1 uses stop-grad on the shared encoder for simplicity.

---

## Data & corruption

### Datasets

- **MNIST:** 28×28 grayscale, 60k train / 10k test
- **CIFAR-10:** 32×32 RGB, 50k train / 10k test
- Loaded via torchvision; normalized to [0, 1]

### ViewCorruptor strategies

Applied to produce view A (corrupted) from view B (clean, original):

1. **Gaussian noise:** σ ∈ [0.1, 0.3], uniform random per sample
2. **Gaussian blur:** kernel 3×3 or 5×5, σ=1.0
3. **Random patch mask:** black out patches (7×7 MNIST, 8×8 CIFAR), ~30% of image area
4. **Mix (default):** uniform random choice among 1–3 per batch item

Corruption is applied **online** during training (not precomputed).

---

## Training

### Loss

```
L = L_jepa
L_jepa = mean(1 - cosine_sim(Predictor(E(corrupt)), stop_grad(E(clean))))
```

No auxiliary CE or reconstruction loss in v1.

### Hyperparameters (defaults)

| Param | Value |
|-------|-------|
| Batch size | 64 |
| Optimizer | Adam, lr=1e-3 |
| Epochs | 10 |
| Embedding dim | 128 |
| Device | CPU (CUDA if available, auto-detect) |

### CLI

```bash
uv run python -m tinyvllm.train --dataset mnist --epochs 10
uv run python -m tinyvllm.train --dataset cifar10 --epochs 10
```

Checkpoints saved to `checkpoints/{dataset}/epoch_{n}.pt`.

---

## Inference (v1)

Educational, not optimized. Comments note where a production system would batch/parallelize.

```bash
# Encode a single image
python -m tinyvllm.inference.encode --checkpoint checkpoints/mnist/epoch_10.pt --image path.png

# Probe: report mean cosine sim (corrupt→predicted vs clean) on test set
python -m tinyvllm.inference.probe --checkpoint checkpoints/mnist/epoch_10.pt --dataset mnist
```

---

## Project layout

```
tinyvllm/
  __init__.py
  config.py           # dataclass: dataset, dims, corruption, paths
  train.py            # training loop
  models/
    __init__.py
    base.py           # ModalityEncoder protocol
    encoder.py        # ImageEncoder
    predictor.py      # Predictor MLP
  jepa/
    __init__.py
    views.py          # ViewCorruptor
    loss.py           # jepa_loss
  data/
    __init__.py
    factory.py        # get_dataloader(dataset, batch_size)
  inference/
    __init__.py
    encode.py
    probe.py
tests/
  test_views.py
  test_loss.py
  test_train_step.py
requirements.txt
README.md             # updated with usage
```

---

## Phase 2 extension points

Stub files / interfaces in v1:

- `models/base.py`: `ModalityEncoder` protocol with docstring for future `TextEncoder`
- `jepa/loss.py`: commented `contrastive_loss` stub for InfoNCE
- `models/encoder.py`: `TextEncoder` class stub (raises NotImplementedError)
- README section: "Roadmap → Phase 2: CLIP bridge"

No phase 2 code runs in v1; stubs exist only to show where multimodal pieces attach.

---

## Testing

| Test | Validates |
|------|-----------|
| `test_views.py` | Corruptor output shape, value range [0,1], corruption differs from input |
| `test_loss.py` | Loss ≥ 0, loss = 0 when pred == target, stop-grad doesn't break backward on pred path |
| `test_train_step.py` | One forward-backward step on MNIST batch; loss is finite scalar |

---

## Success criteria

1. `python -m tinyvllm.train --dataset mnist --epochs 10` completes on CPU in < 15 min
2. Probe script shows JEPA cosine similarity improves over random baseline after training
3. Same codebase runs CIFAR-10 with `--dataset cifar10` without code changes
4. All tests pass: `pytest tests/`
5. Code is readable: each module has a single clear responsibility with docstrings

---

## Dependencies

Managed in `pyproject.toml`. Install with:

```bash
uv sync
```

Core: `torch`, `torchvision`, `Pillow`. Dev: `pytest`.

No CUDA requirement; CPU-only is supported.
