"""Training entry point for the image repair project."""
from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.dataset import PhotoDataset
from src.data.augmentations import apply_random_defects
from src.models.multi_scale_srnet import MultiScaleSRNet
from src.models.defect_repair import DefectRepairModule
from src.utils.logging import setup_logger
from src.utils.metrics import compute_psnr, compute_ssim


@dataclass
class BatchMetrics:
    sr_loss: float
    repair_loss: float
    psnr: float
    ssim: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the image repair and super-resolution models")
    parser.add_argument("--data-root", type=Path, required=True, help="Directory containing training images")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for training")
    parser.add_argument("--patch-size", type=int, default=128, help="Square crop size in pixels (must be divisible by scale)")
    parser.add_argument("--scale", type=int, default=4, choices=[2, 3, 4], help="Target super-resolution scale")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate for both optimizers")
    parser.add_argument("--num-workers", type=int, default=2, help="Number of DataLoader worker processes")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Training device")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/checkpoints"), help="Directory to store checkpoints")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoints every N epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_crop(image: torch.Tensor, size: int) -> torch.Tensor:
    """Randomly crop a square patch; resize if the image is too small."""
    if image.ndim != 3:
        raise ValueError("Expected image tensor with shape (C, H, W)")
    _, height, width = image.shape
    if height < size or width < size:
        # Upsample to at least the desired size before cropping
        new_height = max(size, height)
        new_width = max(size, width)
        image = F.interpolate(
            image.unsqueeze(0),
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        height, width = image.shape[1], image.shape[2]
    top = 0 if height == size else torch.randint(0, height - size + 1, (1,)).item()
    left = 0 if width == size else torch.randint(0, width - size + 1, (1,)).item()
    return image[:, top : top + size, left : left + size]


def build_dataloader(data_root: Path, patch_size: int, batch_size: int, num_workers: int) -> DataLoader:
    def transform(image: torch.Tensor) -> torch.Tensor:
        return random_crop(image, patch_size)

    dataset = PhotoDataset(data_root, transform=transform, simulate_defects=False)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def simulate_defects(batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    degraded_images = []
    defect_masks = []
    for image in batch:
        degraded, mask = apply_random_defects(image)
        degraded_images.append(degraded)
        defect_masks.append(mask)
    degraded_batch = torch.stack(degraded_images)
    masks_batch = torch.stack(defect_masks)
    return degraded_batch, masks_batch


def compute_sr_loss(
    outputs: Dict[int, torch.Tensor],
    targets: torch.Tensor,
    scale: int,
) -> torch.Tensor:
    total_loss = torch.zeros(1, device=targets.device)
    for factor, prediction in outputs.items():
        if factor == scale:
            target = targets
        else:
            factor_scale = factor / scale
            target = F.interpolate(
                targets,
                scale_factor=factor_scale,
                mode="bilinear",
                align_corners=False,
            )
        total_loss = total_loss + F.l1_loss(prediction, target)
    return total_loss / len(outputs)


def train_one_batch(
    sr_model: MultiScaleSRNet,
    repair_module: DefectRepairModule,
    sr_optimizer: torch.optim.Optimizer,
    repair_optimizer: torch.optim.Optimizer,
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    scale: int,
) -> BatchMetrics:
    sr_model.train()
    repair_module.train()

    high_res_cpu = batch["image"]
    degraded_cpu, masks_cpu = simulate_defects(high_res_cpu)

    high_res = high_res_cpu.to(device)
    degraded = degraded_cpu.to(device)
    masks = masks_cpu.to(device)

    low_res = F.interpolate(
        high_res,
        scale_factor=1.0 / scale,
        mode="bicubic",
        align_corners=False,
        recompute_scale_factor=False,
    )

    sr_optimizer.zero_grad(set_to_none=True)
    repair_optimizer.zero_grad(set_to_none=True)

    sr_outputs = sr_model(low_res, return_all=True)
    sr_loss = compute_sr_loss(sr_outputs, high_res, scale)

    repaired, predicted_masks = repair_module(degraded)
    reconstruction_loss = F.l1_loss(repaired, high_res)
    mask_loss = F.binary_cross_entropy(predicted_masks, masks)
    repair_loss = reconstruction_loss + 0.1 * mask_loss

    total_loss = sr_loss + repair_loss
    total_loss.backward()

    torch.nn.utils.clip_grad_norm_(sr_model.parameters(), max_norm=1.0)
    torch.nn.utils.clip_grad_norm_(repair_module.parameters(), max_norm=1.0)

    sr_optimizer.step()
    repair_optimizer.step()

    with torch.no_grad():
        prediction = sr_outputs[scale].clamp(0.0, 1.0)
        psnr = compute_psnr(prediction, high_res)
        ssim = compute_ssim(prediction, high_res)

    return BatchMetrics(
        sr_loss=float(sr_loss.item()),
        repair_loss=float(repair_loss.item()),
        psnr=psnr,
        ssim=ssim,
    )


def save_checkpoint(
    output_dir: Path,
    epoch: int,
    sr_model: MultiScaleSRNet,
    repair_module: DefectRepairModule,
    sr_optimizer: torch.optim.Optimizer,
    repair_optimizer: torch.optim.Optimizer,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"checkpoint_epoch_{epoch:03d}.pth"
    torch.save(
        {
            "epoch": epoch,
            "sr_model": sr_model.state_dict(),
            "repair_module": repair_module.state_dict(),
            "sr_optimizer": sr_optimizer.state_dict(),
            "repair_optimizer": repair_optimizer.state_dict(),
        },
        checkpoint_path,
    )


def main() -> None:
    args = parse_args()
    logger = setup_logger("scripts.train")
    set_seed(args.seed)

    if args.patch_size % args.scale != 0:
        raise ValueError("patch_size must be divisible by the selected scale")

    device = torch.device(args.device)
    logger.info("Using device: %s", device)

    dataloader = build_dataloader(args.data_root, args.patch_size, args.batch_size, args.num_workers)
    logger.info("Loaded dataset with %d samples", len(dataloader.dataset))

    sr_model = MultiScaleSRNet().to(device)
    repair_module = DefectRepairModule().to(device)

    sr_optimizer = torch.optim.Adam(sr_model.parameters(), lr=args.learning_rate)
    repair_optimizer = torch.optim.Adam(repair_module.parameters(), lr=args.learning_rate)

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        running_sr = 0.0
        running_repair = 0.0
        running_psnr = 0.0
        running_ssim = 0.0
        for batch in dataloader:
            metrics = train_one_batch(
                sr_model,
                repair_module,
                sr_optimizer,
                repair_optimizer,
                batch,
                device,
                args.scale,
            )
            running_sr += metrics.sr_loss
            running_repair += metrics.repair_loss
            running_psnr += metrics.psnr
            running_ssim += metrics.ssim
            global_step += 1

            if global_step % 10 == 0:
                logger.info(
                    "Step %d | SR Loss: %.4f | Repair Loss: %.4f | PSNR: %.2f | SSIM: %.4f",
                    global_step,
                    metrics.sr_loss,
                    metrics.repair_loss,
                    metrics.psnr,
                    metrics.ssim,
                )

        num_batches = len(dataloader)
        logger.info(
            "Epoch %d/%d | Avg SR Loss: %.4f | Avg Repair Loss: %.4f | Avg PSNR: %.2f | Avg SSIM: %.4f",
            epoch,
            args.epochs,
            running_sr / num_batches,
            running_repair / num_batches,
            running_psnr / num_batches,
            running_ssim / num_batches,
        )

        if epoch % args.save_every == 0:
            save_checkpoint(
                args.output_dir,
                epoch,
                sr_model,
                repair_module,
                sr_optimizer,
                repair_optimizer,
            )
            logger.info("Saved checkpoint for epoch %d", epoch)

    logger.info("Training complete")


if __name__ == "__main__":
    main()
