"""Evaluation metrics for image restoration quality."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _validate_pair(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.shape != target.shape:
        raise ValueError("Input tensors for metrics must share the same shape")
    if pred.ndim not in (3, 4):
        raise ValueError("Metrics expect 3D (C, H, W) or 4D (N, C, H, W) tensors")
    if pred.ndim == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
    return pred, target


def compute_psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    """Peak signal-to-noise ratio."""
    pred, target = _validate_pair(pred, target)
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float("inf")
    value = 20 * torch.log10(torch.tensor(data_range, device=pred.device)) - 10 * torch.log10(mse)
    return float(value.item())


def _gaussian_window(window_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32) - (window_size - 1) / 2
    grid_x, grid_y = torch.meshgrid(coords, coords, indexing="ij")
    window = torch.exp(-(grid_x ** 2 + grid_y ** 2) / (2 * sigma ** 2))
    window /= window.sum()
    return window


def compute_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> float:
    """Structural similarity index (single image variant)."""
    pred, target = _validate_pair(pred, target)
    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    window = _gaussian_window(window_size, sigma).to(pred.device, pred.dtype)
    window = window.expand(pred.shape[1], 1, window_size, window_size)

    mu_pred = F.conv2d(pred, window, padding=window_size // 2, groups=pred.shape[1])
    mu_target = F.conv2d(target, window, padding=window_size // 2, groups=pred.shape[1])

    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=pred.shape[1]) - mu_pred_sq
    sigma_target_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=pred.shape[1]) - mu_target_sq
    sigma_pred_target = F.conv2d(pred * target, window, padding=window_size // 2, groups=pred.shape[1]) - mu_pred_target

    numerator = (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
    denominator = (mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2)

    ssim_map = numerator / (denominator + 1e-12)
    return float(ssim_map.mean().item())
