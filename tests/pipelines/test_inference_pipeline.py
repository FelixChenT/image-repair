import torch

from src.pipelines import ImageRepairPipeline


def test_pipeline_executes_and_returns_outputs():
    pipeline = ImageRepairPipeline()
    image = torch.rand(3, 16, 16)
    result = pipeline(image, scale=2)
    assert result.super_resolved.shape == (3, 32, 32)
    assert result.denoised.shape == (3, 32, 32)
    assert result.defect_mask.shape == (1, 16, 16)
    assert isinstance(result.metrics, dict) and not result.metrics


def test_pipeline_accepts_reference_for_metrics():
    pipeline = ImageRepairPipeline()
    image = torch.rand(3, 16, 16)
    reference = image.clone()
    result = pipeline(image, scale=2, reference=reference)
    assert "psnr" in result.metrics
    assert "ssim" in result.metrics
