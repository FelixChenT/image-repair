"""Dataset utilities for loading vintage photo samples."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Optional

import torch
from torch.utils.data import Dataset

from .augmentations import apply_random_defects
from ..utils.image_io import load_image


class PhotoDataset(Dataset):
    """Lightweight dataset that reads images from a directory."""

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        simulate_defects: bool = False,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.root = Path(root)
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root}")
        self.paths = sorted(
            [p for p in self.root.rglob("*") if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}]
        )
        if not self.paths:
            raise ValueError(f"No image files discovered under {self.root}")
        self.transform = transform
        self.simulate_defects = simulate_defects
        self.device = torch.device(device) if device is not None else None

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        path = self.paths[index]
        image = load_image(path, device=self.device)
        original = image.clone()
        mask = torch.zeros(1, image.shape[1], image.shape[2], device=image.device)
        if self.simulate_defects:
            image, mask = apply_random_defects(image)
            if self.device is not None:
                image = image.to(self.device)
                mask = mask.to(self.device)
        if self.transform is not None:
            image = self.transform(image)
        return {"image": image, "original": original, "defect_mask": mask, "path": path}
