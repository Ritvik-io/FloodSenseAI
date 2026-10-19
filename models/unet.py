"""
FloodSense AI - Lightweight Configurable PyTorch U-Net
Architecture:
  - Input: 2 channels (Sentinel-1 VV and VH backscatter in dB)
  - Output: 1 logit per pixel (unnormalized probability of flood inundation)
  - Configurable base channels (default: 32)
  - Robust spatial dimension handling with automatic padding
"""

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """
    [Conv2d -> BatchNorm -> ReLU] x 2
    """
    def __init__(self, in_channels: int, out_channels: int, mid_channels: Optional[int] = None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.double_conv(x)


class DownBlock(nn.Module):
    """
    Downscaling with MaxPool2d then DoubleConv
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool_conv(x)


class UpBlock(nn.Module):
    """
    Upscaling then DoubleConv. Handles odd spatial dimensions via center padding.
    """
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = False):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # Handle spatial padding if dimensions differ
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        if diff_x != 0 or diff_y != 0:
            x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])

        # Concatenate along channel dimension
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    Lightweight, configurable U-Net for Sentinel-1 SAR flood segmentation.
    Args:
        in_channels: Number of input channels (default: 2 for VV and VH)
        out_channels: Number of output channels (default: 1 for binary flood logit)
        base_channels: Starting feature channel depth (default: 32)
        dropout: Dropout probability in the bottleneck (default: 0.2)
        bilinear: If True, uses bilinear interpolation instead of ConvTranspose2d
    """
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.2,
        bilinear: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.bilinear = bilinear

        b = base_channels
        factor = 2 if bilinear else 1

        # Encoder
        self.inc = DoubleConv(in_channels, b)
        self.down1 = DownBlock(b, b * 2)
        self.down2 = DownBlock(b * 2, b * 4)
        self.down3 = DownBlock(b * 4, b * 8)
        self.down4 = DownBlock(b * 8, (b * 16) // factor)

        # Bottleneck dropout
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0.0 else nn.Identity()

        # Decoder
        self.up1 = UpBlock(b * 16, (b * 8) // factor, bilinear)
        self.up2 = UpBlock(b * 8, (b * 4) // factor, bilinear)
        self.up3 = UpBlock(b * 4, (b * 2) // factor, bilinear)
        self.up4 = UpBlock(b * 2, b, bilinear)

        # Segmentation Head (1x1 Conv returning unnormalized logits)
        self.outc = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.dropout(x5)

        # Decoder
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        logits = self.outc(x)
        return logits


def get_model(
    in_channels: int = 2,
    out_channels: int = 1,
    base_channels: int = 32,
    dropout: float = 0.2,
) -> UNet:
    """Helper factory for instantiating the baseline U-Net."""
    return UNet(
        in_channels=in_channels,
        out_channels=out_channels,
        base_channels=base_channels,
        dropout=dropout,
    )
