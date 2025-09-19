"""I/O helpers for reading and writing image tensors."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image


def load_image(path: str | Path, device: Optional[torch.device | str] = None) -> torch.Tensor:
    """Load an image file into a normalized float tensor in CHW format."""
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    if device is not None:
        tensor = tensor.to(device)
    return tensor


def save_image(tensor: torch.Tensor, path: str | Path) -> None:
    """Persist a normalized image tensor (CHW) to disk as an RGB PNG."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    clamped = tensor.detach().cpu().clamp(0.0, 1.0)
    array = (clamped.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)
    Image.fromarray(array).save(path)


def ensure_rgb(image: torch.Tensor) -> torch.Tensor:
    """Guarantee that the tensor has three channels by duplicating or trimming."""
    if image.ndim != 3:
        raise ValueError("Expected image tensor with shape (C, H, W)")
    channels = image.shape[0]
    if channels == 3:
        return image
    if channels == 1:
        return image.repeat(3, 1, 1)
    if channels > 3:
        return image[:3]
    raise ValueError(f"Unsupported channel count: {channels}")
