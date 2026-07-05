"""Match an image to the closest class-name caption (zero-shot CLIP-style)."""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from tinyvllm.data.char_tokenizer import encode, vocab_size
from tinyvllm.data.factory import get_dataloader
from tinyvllm.data.labels import get_label_names, label_to_caption
from tinyvllm.models.text_encoder import TextEncoder
from tinyvllm.train_clip import load_clip_checkpoint


@torch.no_grad()
def encode_all_class_names(text_encoder, dataset: str, max_len: int, device) -> tuple[torch.Tensor, list[str]]:
    names = get_label_names(dataset)
    tokens = torch.tensor(
        [encode(name, max_len=max_len) for name in names],
        dtype=torch.long,
        device=device,
    )
    return text_encoder(tokens), names


@torch.no_grad()
def evaluate(
    image_encoder,
    text_encoder,
    config,
    device: torch.device,
    max_batches: int = 50,
) -> dict[str, float]:
    image_encoder.eval()
    text_encoder.eval()

    class_emb, class_names = encode_all_class_names(
        text_encoder, config.dataset, config.max_text_len, device
    )

    img_size = config.image_size if config.encoder == "vit" else None
    loader = get_dataloader(
        config.dataset,
        batch_size=config.batch_size,
        train=False,
        num_workers=config.num_workers,
        data_root=config.data_root,
        image_size=img_size,
    )

    correct = 0
    total = 0
    sim_diag = []

    for i, (images, labels) in enumerate(loader):
        if i >= max_batches:
            break
        images = images.to(device)
        labels = labels.to(device)

        img_emb = image_encoder(images)
        logits = img_emb @ class_emb.T
        preds = logits.argmax(dim=1)

        for b in range(len(labels)):
            gt_name = label_to_caption(config.dataset, int(labels[b].item()))
            pred_name = class_names[int(preds[b].item())]
            if pred_name == gt_name:
                correct += 1
            total += 1

        # Similarity of image to its true caption vs random caption.
        true_tokens = torch.tensor(
            [encode(label_to_caption(config.dataset, int(y.item())), max_len=config.max_text_len)
             for y in labels],
            dtype=torch.long,
            device=device,
        )
        true_emb = text_encoder(true_tokens)
        sim_diag.append(F.cosine_similarity(img_emb, true_emb, dim=1).mean().item())

    return {
        "top1_accuracy": correct / max(total, 1),
        "mean_cos_image_true_text": sum(sim_diag) / len(sim_diag),
        "num_classes": len(class_names),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Match images to class-name captions")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-batches", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_encoder, text_encoder, config, epoch = load_clip_checkpoint(args.checkpoint, device)
    stats = evaluate(image_encoder, text_encoder, config, device, max_batches=args.max_batches)

    print(f"Checkpoint epoch: {epoch} | Dataset: {config.dataset}")
    print(f"  Top-1 caption accuracy: {stats['top1_accuracy']:.4f}")
    print(f"  cos(image, true caption): {stats['mean_cos_image_true_text']:.4f}")
    print(f"  ({stats['num_classes']} class names as text candidates)")


if __name__ == "__main__":
    main()
