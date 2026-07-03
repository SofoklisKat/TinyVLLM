# TinyVLLM

An educational, unoptimized implementation of **vision JEPA** (Joint Embedding Predictive Architecture) for learning — not for production inference speed.

## What this teaches

- **JEPA vs MAE:** predict embeddings in latent space, not pixels
- **Two views:** corrupted image → predict clean image embedding
- **Stop-gradient:** why the clean encoder path is detached (anti-collapse)
- **Path to CLIP:** shared embedding space designed for future image→text alignment

## Architecture

```
Corrupted image ──► ImageEncoder ──► Predictor ──► predicted embedding
                                                        │
                                              cosine loss
                                                        │
Clean image ──► ImageEncoder (stop-grad) ──► target embedding
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
# MNIST (~minutes on CPU)
uv run python -m tinyvllm.train --dataset mnist --epochs 10

# CIFAR-10
uv run python -m tinyvllm.train --dataset cifar10 --epochs 10
```

Checkpoints: `checkpoints/{dataset}/epoch_{n}.pt`

## Inference

```bash
# Embedding vector for one image
uv run python -m tinyvllm.inference.encode \
  --checkpoint checkpoints/mnist/epoch_10.pt \
  --image path/to/digit.png

# Probe cosine similarity on test set (trained vs random baseline)
uv run python -m tinyvllm.inference.probe \
  --checkpoint checkpoints/mnist/epoch_10.pt \
  --dataset mnist
```

## Tests

```bash
uv run pytest tests/
```

## Project layout

```
tinyvllm/
  models/     ImageEncoder, Predictor, TextEncoder stub
  jepa/       ViewCorruptor, jepa_loss
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
