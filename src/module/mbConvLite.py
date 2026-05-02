import torch
from torch import nn

from classifier_spectrogram.src.module.ecaBlock import ECABlock


class MBConvLite(nn.Module):
    """A lightweight MBConv-style block with optional ECA attention."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expand_ratio: int = 2,
        use_attention: bool = True
    ):
        super().__init__()

        hidden_channels = in_channels * expand_ratio
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

        layers = []

        # Expansion
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_channels),
                nn.SiLU(inplace=True)
            ])
        else:
            hidden_channels = in_channels

        # Depthwise
        layers.extend([
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=hidden_channels,
                bias=False
            ),
            nn.BatchNorm2d(hidden_channels),
            nn.SiLU(inplace=True)
        ])

        # Attention
        if use_attention:
            layers.append(ECABlock())

        # Projection
        layers.extend([
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        out += self.shortcut(x)
        return out
