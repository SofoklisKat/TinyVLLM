# TinyVLLM

An educational, unoptimized implementation of **vision JEPA** (Joint Embedding Predictive Architecture) for learning — not for production inference speed.

## What this teaches

- **JEPA vs MAE:** predict embeddings in latent space, not pixels
- **Two views:** corrupted image → predict clean image embedding
- **Stop-gradient:** why the clean encoder path is detached (anti-collapse)
- **Path to CLIP:** shared embedding space designed for future image→text alignment

## Architecture

**v1 — CNN + global JEPA** (default):

```
Corrupted image ──► CNN encoder ──► Predictor ──► predicted global embedding
```

**v1.5 — ViT + patch JEPA** (paper direction, consumer GPU):

```
Corrupted image ──► ViT encoder ──► patch tokens ──► PatchPredictor ──► predicted patch embeddings
                              self-attention                    cosine loss vs clean patches (stop-grad)
```

## Setup (run on your server)

Install [uv](https://docs.astral.sh/uv/) on the machine where you train, then from the repo root:

```bash
uv sync
```

This creates `.venv`, installs the project in editable mode, and pulls in dev tools (pytest). Dependencies are defined in `pyproject.toml`.

Use `uv run` so you don't need to activate the venv:

```bash
uv run pytest tests/
uv run python -m tinyvllm.train --dataset mnist --epochs 10
```

Optional — activate the venv instead:

```bash
source .venv/bin/activate
python -m tinyvllm.train --dataset mnist --epochs 10
```

## Train

```bash
# v1: CNN + global JEPA (MNIST, fast on CPU)
uv run python -m tinyvllm.train --dataset mnist --epochs 10

# Fast download (~30 MB) — good if CIFAR-100 is slow on your network
uv run python -m tinyvllm.train \
  --encoder vit --jepa-mode patch \
  --dataset fashion_mnist --epochs 10

# Tiny ImageNet (~237 MB auto-download, 64×64 RGB, 200 classes — paper-friendly)
uv run python -m tinyvllm.train \
  --encoder vit --jepa-mode patch \
  --dataset tiny_imagenet --epochs 10

# CIFAR-100 (~169 MB — same download size as CIFAR-10)
uv run python -m tinyvllm.train \
  --encoder vit --jepa-mode patch \
  --dataset cifar100 --epochs 10

# ImageNet — manual download; expects {data_root}/train/ and {data_root}/val/
uv run python -m tinyvllm.train \
  --encoder vit --jepa-mode patch \
  --dataset imagenet --data-root /path/to/imagenet \
  --batch-size 32 --num-workers 4 --epochs 10

# Paper-scale CIFAR / small GPU
uv run python -m tinyvllm.train \
  --encoder vit --jepa-mode patch \
  --dataset cifar10 --image-size 32 --epochs 10

# ImageNet at 224×224 (12–16 GB GPU; reduce batch if OOM)
uv run python -m tinyvllm.train \
  --encoder vit --jepa-mode patch \
  --dataset imagenet --data-root /path/to/imagenet \
  --image-size 224 --vit-patch-size 16 --batch-size 16 --epochs 10
```

Checkpoints: `checkpoints/{dataset}/{encoder}_{jepa_mode}/epoch_{n}.pt`

### Datasets

| Dataset | Download | Default `--image-size` | Notes |
|---------|----------|------------------------|--------|
| `mnist` | ~11 MB | 32 | CPU-friendly |
| **`fashion_mnist`** | **~30 MB** | 32 | **Fast download**, grayscale, 10 classes |
| `cifar10` | ~170 MB | 32 | 10 classes |
| `cifar100` | ~169 MB | 32 | 100 classes |
| **`tiny_imagenet`** | **~237 MB** | 64 | **Auto-download**, 200 classes, 64×64 RGB |
| `imagenet` | **manual** | 224 | ImageFolder layout below |

**ImageNet folder layout** (or ImageNet-100 subset with same structure):

```
/path/to/imagenet/
  train/
    n01440764/
      *.JPEG
  val/
    n01440764/
      *.JPEG
```

| Flag | Options | Default |
|------|---------|---------|
| `--encoder` | `cnn`, `vit` | `cnn` |
| `--jepa-mode` | `global`, `patch` | `global` |
| `--image-size` | e.g. 32, 224 | 32 |
| `--vit-patch-size` | e.g. 4, 16 | 4 |
| `--vit-depth` | transformer blocks | 4 |
| `--vit-heads` | attention heads | 4 |

## Inference

```bash
# Embedding vector for one image
uv run python -m tinyvllm.inference.encode \
  --checkpoint checkpoints/mnist/epoch_10.pt \
  --image path/to/digit.png

# Probe cosine similarity on test set (trained vs random baseline)
uv run python -m tinyvllm.inference.probe \
  --checkpoint checkpoints/cifar10/vit_patch/epoch_10.pt \
  --dataset cifar10
```

## Tests

```bash
uv run pytest tests/
```

## Project layout

```
tinyvllm/
  models/     ImageEncoder, ViTEncoder, Predictor, PatchPredictor
  jepa/       ViewCorruptor, jepa_loss, jepa_patch_loss
  data/       MNIST / CIFAR-10 loaders
  inference/  encode, probe
  train.py
```

## Roadmap — Phase 2: CLIP bridge

- [ ] `TextEncoder` for captions (MNIST digit names, CIFAR class labels)
- [ ] InfoNCE contrastive loss in shared 128-d space
- [ ] `TokenDecoder`: embedding → text tokens

See `docs/superpowers/specs/2026-07-03-tinyvllm-jepa-design.md` for the full design.

## License

See [LICENSE](LICENSE).
