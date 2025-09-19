import torch

from src.utils.metrics import compute_psnr, compute_ssim


def test_psnr_identical_images_is_infinite():
    image = torch.rand(3, 16, 16)
    assert compute_psnr(image, image) == float("inf")


def test_psnr_decreases_with_noise():
    image = torch.zeros(3, 16, 16)
    noisy = image + 0.1 * torch.randn_like(image)
    psnr = compute_psnr(image, noisy)
    assert psnr < 40
    assert psnr > 0


def test_ssim_bounds():
    image = torch.rand(3, 32, 32)
    ssim_same = compute_ssim(image, image)
    noisy = (image + 0.3 * torch.randn_like(image)).clamp(0, 1)
    ssim_noisy = compute_ssim(image, noisy)
    assert ssim_same <= 1.0
    assert ssim_same > ssim_noisy
