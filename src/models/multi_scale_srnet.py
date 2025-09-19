"""Multi-scale super-resolution backbone."""
from __future__ import annotations

from typing import Dict

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.activation = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.activation(self.conv1(x))
        out = self.conv2(out)
        return residual + out


class MultiScaleSRNet(nn.Module):
    """A lightweight residual network that produces x2/x3/x4 outputs."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_blocks: int = 8,
    ) -> None:
        super().__init__()
        self.head = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.body = nn.Sequential(*[ResidualBlock(base_channels) for _ in range(num_blocks)])
        self.fusion = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)

        self.upsamplers = nn.ModuleDict(
            {
                "x2": nn.Sequential(
                    nn.Conv2d(base_channels, base_channels * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
                ),
                "x3": nn.Sequential(
                    nn.Conv2d(base_channels, base_channels * 9, kernel_size=3, padding=1),
                    nn.PixelShuffle(3),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
                ),
                "x4": nn.Sequential(
                    nn.Conv2d(base_channels, base_channels * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(base_channels, base_channels * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),
                    nn.LeakyReLU(0.1, inplace=True),
                    nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1),
                ),
            }
        )
        self.output_activation = nn.Sigmoid()

    def forward(
        self,
        x: torch.Tensor,
        scale: int = 4,
        return_all: bool = False,
    ) -> torch.Tensor | Dict[int, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError("Expected input of shape (N, C, H, W)")
        features = self.head(x)
        body_out = self.body(features)
        fused = self.fusion(body_out + features)

        outputs: Dict[int, torch.Tensor] = {}
        for key, module in self.upsamplers.items():
            factor = int(key[1:])
            outputs[factor] = self.output_activation(module(fused))
        if return_all:
            return outputs
        if scale not in outputs:
            raise ValueError(f"Unsupported scale factor: {scale}")
        return outputs[scale]
