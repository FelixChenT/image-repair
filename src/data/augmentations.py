"""Synthetic augmentations to emulate old-photo degradations."""
from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch


def _random_line_mask(height: int, width: int, thickness: int = 1) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.float32)
    x0, y0 = random.randint(0, width - 1), random.randint(0, height - 1)
    angle = random.uniform(0, np.pi)
    length = random.randint(width // 4, int(width * 0.8))
    for i in range(length):
        x = int(x0 + i * np.cos(angle))
        y = int(y0 + i * np.sin(angle))
        for dx in range(-thickness, thickness + 1):
            for dy in range(-thickness, thickness + 1):
                xx, yy = x + dx, y + dy
                if 0 <= xx < width and 0 <= yy < height:
                    mask[yy, xx] = 1.0
    return mask


def apply_random_defects(
    image: torch.Tensor,
    scratch_count: int = 3,
    noise_std: float = 0.02,
    blotch_chance: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create a degraded copy of the image and a binary defect mask."""
    if image.ndim != 3:
        raise ValueError("apply_random_defects expects an image tensor with shape (C, H, W)")
    _, height, width = image.shape
    mask = np.zeros((height, width), dtype=np.float32)
    for _ in range(max(0, scratch_count)):
        mask = np.maximum(mask, _random_line_mask(height, width, thickness=1))
    if random.random() < blotch_chance:
        blotch = np.zeros_like(mask)
        cy, cx = random.randint(0, height - 1), random.randint(0, width - 1)
        radius = random.randint(min(height, width) // 10, min(height, width) // 4)
        y_grid, x_grid = np.ogrid[:height, :width]
        circle = (x_grid - cx) ** 2 + (y_grid - cy) ** 2 <= radius ** 2
        blotch[circle] = 1.0
        mask = np.maximum(mask, blotch)
    defect_mask = torch.from_numpy(mask).unsqueeze(0)
    noisy = image + noise_std * torch.randn_like(image)
    degraded = torch.where(defect_mask > 0, torch.rand_like(image) * defect_mask, noisy)
    return degraded.clamp(0.0, 1.0), defect_mask.clamp(0.0, 1.0)
