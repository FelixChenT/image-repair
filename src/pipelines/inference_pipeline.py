"""End-to-end inference pipeline for image repair."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F

from ..models.defect_repair import DefectRepairModule
from ..models.multi_scale_srnet import MultiScaleSRNet
from ..utils import color
from ..utils.image_io import ensure_rgb
from ..utils.metrics import compute_psnr, compute_ssim


@dataclass
class PipelineOutput:
    repaired_low_res: torch.Tensor
    defect_mask: torch.Tensor
    super_resolved: torch.Tensor
    color_restored: torch.Tensor
    denoised: torch.Tensor
    metrics: Dict[str, float]


class ImageRepairPipeline:
    """Pipeline that combines defect repair, super-resolution, and enhancement."""

    def __init__(
        self,
        sr_model: Optional[MultiScaleSRNet] = None,
        repair_module: Optional[DefectRepairModule] = None,
        device: torch.device | str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.sr_model = (sr_model or MultiScaleSRNet()).to(self.device)
        self.repair_module = (repair_module or DefectRepairModule()).to(self.device)
        self.sr_model.eval()
        self.repair_module.eval()

    @torch.no_grad()
    def __call__(
        self,
        image: torch.Tensor,
        scale: int = 4,
        reference: Optional[torch.Tensor] = None,
    ) -> PipelineOutput:
        image = ensure_rgb(image).to(self.device).unsqueeze(0)
        repaired_low_res, defect_mask = self.repair_module(image)
        super_resolved = self.sr_model(repaired_low_res, scale=scale)
        enhanced = color.restore_colors(super_resolved.squeeze(0))
        denoised = color.denoise_image(enhanced)

        metrics: Dict[str, float] = {}
        if reference is not None:
            reference = ensure_rgb(reference).to(self.device)
            if reference.shape[-2:] != denoised.shape[-2:]:
                reference = F.interpolate(
                    reference.unsqueeze(0),
                    size=denoised.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0)
            metrics = {
                "psnr": compute_psnr(denoised, reference),
                "ssim": compute_ssim(denoised, reference),
            }

        return PipelineOutput(
            repaired_low_res=repaired_low_res.squeeze(0).cpu(),
            defect_mask=defect_mask.squeeze(0).cpu(),
            super_resolved=super_resolved.squeeze(0).cpu(),
            color_restored=enhanced.cpu(),
            denoised=denoised.cpu(),
            metrics=metrics,
        )
