"""Probe JEPA quality on a test set — cosine similarity stats."""

import argparse

import torch
import torch.nn.functional as F

from tinyvllm.data.factory import get_dataloader
from tinyvllm.jepa.views import ViewCorruptor
from tinyvllm.train import load_checkpoint


@torch.no_grad()
def probe(
    encoder,
    predictor,
    config,
    device: torch.device,
    max_batches: int = 50,
) -> dict[str, float]:
    """Compare trained JEPA vs random predictor baseline."""
    encoder.eval()
    predictor.eval()
    corruptor = ViewCorruptor(mode=config.corruption, patch_size=config.patch_size)
    loader = get_dataloader(
        config.dataset,
        batch_size=config.batch_size,
        train=False,
        num_workers=config.num_workers,
    )

    sim_pred_clean = []
    sim_random_clean = []

    for i, (images, _) in enumerate(loader):
        if i >= max_batches:
            break

        clean = images.to(device)
        corrupt = corruptor(clean).to(device)

        z_clean = encoder(clean)
        z_pred = predictor(encoder(corrupt))

        # Random unit vector baseline — untrained predictor equivalent.
        z_random = F.normalize(torch.randn_like(z_clean), dim=1)

        sim_pred_clean.append(F.cosine_similarity(z_pred, z_clean, dim=1).mean().item())
        sim_random_clean.append(F.cosine_similarity(z_random, z_clean, dim=1).mean().item())

    return {
        "mean_cos_pred_vs_clean": sum(sim_pred_clean) / len(sim_pred_clean),
        "mean_cos_random_vs_clean": sum(sim_random_clean) / len(sim_random_clean),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe JEPA embedding alignment")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=["mnist", "cifar10"], default=None)
    parser.add_argument("--max-batches", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, predictor, config, epoch = load_checkpoint(args.checkpoint, device)

    if args.dataset is not None:
        config.dataset = args.dataset

    stats = probe(encoder, predictor, config, device, max_batches=args.max_batches)

    print(f"Checkpoint epoch: {epoch} | Dataset: {config.dataset}")
    print(f"  cos(predicted, clean): {stats['mean_cos_pred_vs_clean']:.4f}")
    print(f"  cos(random,   clean): {stats['mean_cos_random_vs_clean']:.4f}")
    print(
        "  (Trained predictor should be higher than random baseline after training.)"
    )


if __name__ == "__main__":
    main()
