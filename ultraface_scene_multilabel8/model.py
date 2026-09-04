"""UltraFace slim (Mb_Tiny) backbone adapted for 8-label scene classification.

Backbone topology follows Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
vision/nn/mb_tiny.py (commit dffdddda9794a50607cba8f318507a28c1c27cab):
base_channel=16, one conv-bn-relu stem and 12 depthwise-separable blocks.
Detection-specific SSD extras/box/classification heads are intentionally omitted.
"""
from __future__ import annotations
import torch
from torch import nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, inp: int, oup: int, stride: int):
        super().__init__(
            nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )


class ConvDW(nn.Sequential):
    def __init__(self, inp: int, oup: int, stride: int):
        super().__init__(
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )


class UltraFaceSlimScene8(nn.Module):
    def __init__(self, num_labels: int = 8, dropout: float = 0.1):
        super().__init__()
        c = 16  # original Mb_Tiny: base_channel = 8 * 2
        self.backbone = nn.Sequential(
            ConvBNReLU(3, c, 2),
            ConvDW(c, c * 2, 1),
            ConvDW(c * 2, c * 2, 2),
            ConvDW(c * 2, c * 2, 1),
            ConvDW(c * 2, c * 4, 2),
            ConvDW(c * 4, c * 4, 1),
            ConvDW(c * 4, c * 4, 1),
            ConvDW(c * 4, c * 4, 1),
            ConvDW(c * 4, c * 8, 2),
            ConvDW(c * 8, c * 8, 1),
            ConvDW(c * 8, c * 8, 1),
            ConvDW(c * 8, c * 16, 2),
            ConvDW(c * 16, c * 16, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(c * 16, num_labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.pool(x).flatten(1)
        x = self.dropout(x)
        return self.head(x)


def create_ultraface_slim_scene8(dropout: float = 0.1) -> UltraFaceSlimScene8:
    return UltraFaceSlimScene8(num_labels=8, dropout=dropout)


if __name__ == "__main__":
    m = create_ultraface_slim_scene8().eval()
    x = torch.zeros(1, 3, 360, 640)
    y = m(x)
    print("input", tuple(x.shape), "output", tuple(y.shape))
    print("params", sum(p.numel() for p in m.parameters()))
    assert tuple(y.shape) == (1, 8)
