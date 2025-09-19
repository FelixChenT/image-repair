from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.data import PhotoDataset


def _create_sample_image(path: Path) -> None:
    array = (np.random.rand(32, 32, 3) * 255).astype(np.uint8)
    Image.fromarray(array).save(path)


def test_photo_dataset_loads_images(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _create_sample_image(image_dir / "sample.png")

    dataset = PhotoDataset(image_dir)
    sample = dataset[0]
    assert set(sample.keys()) == {"image", "original", "defect_mask", "path"}
    assert sample["image"].shape == (3, 32, 32)
    assert sample["original"].shape == (3, 32, 32)
    assert sample["defect_mask"].shape == (1, 32, 32)


def test_photo_dataset_with_defect_simulation(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    image_dir.mkdir()
    _create_sample_image(image_dir / "sample.png")

    dataset = PhotoDataset(image_dir, simulate_defects=True)
    sample = dataset[0]
    assert torch.any(sample["defect_mask"] >= 0)
    assert sample["image"].shape == (3, 32, 32)
