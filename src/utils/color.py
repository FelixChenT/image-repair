"""Color restoration and denoising utilities."""
from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F


def restore_colors(image: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Apply a gray-world inspired gain to rebalance channel intensities."""
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("restore_colors expects an image tensor with shape (3, H, W)")
    channel_means = image.view(3, -1).mean(dim=1)
    global_mean = channel_means.mean()
    gains = (global_mean / (channel_means + epsilon)).view(3, 1, 1)
    balanced = image * gains
    return balanced.clamp(0.0, 1.0)


def _gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    return kernel


def denoise_image(
    image: torch.Tensor,
    strength: float = 0.2,
    kernel_size: int = 5,
    sigma: float | None = None,
) -> torch.Tensor:
    """Apply a lightweight Gaussian blur and blend it with the original image."""
    if image.ndim != 3:
        raise ValueError("denoise_image expects an image tensor with shape (C, H, W)")
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("kernel_size must be a positive odd integer")
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength must be in [0, 1]")

    sigma = sigma or max(kernel_size / 3.0, 1.0)
    kernel = _gaussian_kernel(kernel_size, sigma)
    kernel = kernel.to(image.device, image.dtype)
    kernel = kernel.expand(image.shape[0], 1, kernel_size, kernel_size)

    padded = image.unsqueeze(0)
    smoothed = F.conv2d(padded, kernel, padding=kernel_size // 2, groups=image.shape[0])
    blended = (1.0 - strength) * padded + strength * smoothed
    return blended.squeeze(0)
