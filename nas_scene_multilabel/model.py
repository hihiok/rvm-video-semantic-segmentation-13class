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


class DepthwiseSeparable(nn.Sequential):
    def __init__(self, inp: int, oup: int, stride: int):
        super().__init__(
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            nn.ReLU(inplace=True),
            nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
            nn.ReLU(inplace=True),
        )


class UltraFaceSlimMultiLabel(nn.Module):
    """UltraFace Mb_Tiny/slim backbone with RFB/detection heads removed.

    The convolutional stack follows the public UltraFace `Mb_Tiny.model` topology.
    For scene tagging, SSD extras/RFB/box heads are removed and replaced by
    AdaptiveAvgPool2d + a 9-logit linear head.
    """

    def __init__(self, num_labels: int = 9, base_channel: int = 16, dropout: float = 0.1):
        super().__init__()
        c = base_channel
        self.base_channel = c
        self.backbone = nn.Sequential(
            ConvBNReLU(3, c, 2),
            DepthwiseSeparable(c, c * 2, 1),
            DepthwiseSeparable(c * 2, c * 2, 2),
            DepthwiseSeparable(c * 2, c * 2, 1),
            DepthwiseSeparable(c * 2, c * 4, 2),
            DepthwiseSeparable(c * 4, c * 4, 1),
            DepthwiseSeparable(c * 4, c * 4, 1),
            DepthwiseSeparable(c * 4, c * 4, 1),
            DepthwiseSeparable(c * 4, c * 8, 2),
            DepthwiseSeparable(c * 8, c * 8, 1),
            DepthwiseSeparable(c * 8, c * 8, 1),
            DepthwiseSeparable(c * 8, c * 16, 2),
            DepthwiseSeparable(c * 16, c * 16, 1),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(c * 16, num_labels)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        return self.pool(x).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.forward_features(x)))


def create_model(num_labels: int = 9, base_channel: int = 16, dropout: float = 0.1):
    return UltraFaceSlimMultiLabel(num_labels=num_labels, base_channel=base_channel, dropout=dropout)
