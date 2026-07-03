"""View corruption — produces the 'context' view for JEPA training."""

import random
from typing import Literal

import torch
import torch.nn.functional as F

CorruptionKind = Literal["noise", "blur", "mask"]


class ViewCorruptor:
    """Turn a clean image batch into a corrupted view (view A).

    Corruption is applied online each batch — not precomputed — so the
    model sees diverse corruptions every epoch.
    """

    def __init__(
        self,
        mode: Literal["noise", "blur", "mask", "mix"] = "mix",
        patch_size: int = 7,
    ):
        self.mode = mode
        self.patch_size = patch_size

    def __call__(self, clean: torch.Tensor) -> torch.Tensor:
        """Return corrupted copy of clean, same shape, values clamped to [0, 1]."""
        if self.mode == "mix":
            return torch.stack(
                [self._corrupt_one(clean[i], self._pick_kind()) for i in range(clean.size(0))]
            )

        return torch.stack([self._corrupt_one(clean[i], self.mode) for i in range(clean.size(0))])

    def _pick_kind(self) -> CorruptionKind:
        return random.choice(["noise", "blur", "mask"])

    def _corrupt_one(self, img: torch.Tensor, kind: CorruptionKind) -> torch.Tensor:
        if kind == "noise":
            return self._add_noise(img)
        if kind == "blur":
            return self._blur(img)
        return self._patch_mask(img)

    def _add_noise(self, img: torch.Tensor) -> torch.Tensor:
        sigma = random.uniform(0.1, 0.3)
        noisy = img + torch.randn_like(img) * sigma
        return noisy.clamp(0.0, 1.0)

    def _blur(self, img: torch.Tensor) -> torch.Tensor:
        kernel_size = random.choice([3, 5])
        sigma = 1.0
        channels = img.shape[0]
        # Depthwise blur: one Gaussian kernel per channel.
        kernel = self._gaussian_kernel(kernel_size, sigma).to(img.device)
        kernel = kernel.expand(channels, 1, kernel_size, kernel_size)
        x = img.unsqueeze(0)
        pad = kernel_size // 2
        blurred = F.conv2d(x, kernel, padding=pad, groups=channels)
        return blurred.squeeze(0).clamp(0.0, 1.0)

    def _patch_mask(self, img: torch.Tensor) -> torch.Tensor:
        """Black out random patches covering ~30% of pixels."""
        corrupted = img.clone()
        _, height, width = img.shape
        ps = self.patch_size
        n_patches_h = max(1, height // ps)
        n_patches_w = max(1, width // ps)
        n_mask = max(1, int(0.3 * n_patches_h * n_patches_w))

        for _ in range(n_mask):
            ph = random.randint(0, n_patches_h - 1)
            pw = random.randint(0, n_patches_w - 1)
            y0, x0 = ph * ps, pw * ps
            y1 = min(y0 + ps, height)
            x1 = min(x0 + ps, width)
            corrupted[:, y0:y1, x0:x1] = 0.0

        return corrupted

    @staticmethod
    def _gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g = g / g.sum()
        kernel = g.outer(g)
        return kernel.view(1, 1, size, size)
