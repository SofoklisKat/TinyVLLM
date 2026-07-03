"""JEPA loss — cosine distance in embedding space."""

import torch
import torch.nn.functional as F


def jepa_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    *,
    stop_grad_target: bool = True,
) -> torch.Tensor:
    """Cosine JEPA loss: mean(1 - cosine_sim(predicted, target)).

    Args:
        predicted: Output of Predictor(encoder(corrupt)), shape (B, D).
        target: Output of encoder(clean). When stop_grad_target is True,
            callers should pass target.detach() — see train.py.
        stop_grad_target: Documented flag; detach must be applied before call.

    Returns:
        Scalar loss in [0, 2] (0 = perfect alignment).
    """
    if stop_grad_target:
        target = target.detach()

    # cosine_sim returns values in [-1, 1]; we want distance so 1 - sim.
    sim = F.cosine_similarity(predicted, target, dim=1)
    return (1.0 - sim).mean()


# ---------------------------------------------------------------------------
# Phase 2 stub — CLIP-style InfoNCE (not used in v1)
# ---------------------------------------------------------------------------
#
# def contrastive_loss(image_emb, text_emb, temperature=0.07):
#     """InfoNCE symmetric loss for image-text pairs (CLIP)."""
#     logits = image_emb @ text_emb.T / temperature
#     labels = torch.arange(len(logits), device=logits.device)
#     loss_i = F.cross_entropy(logits, labels)
#     loss_t = F.cross_entropy(logits.T, labels)
#     return (loss_i + loss_t) / 2
