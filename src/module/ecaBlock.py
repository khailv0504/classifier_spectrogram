import torch
from torch import nn


class ECABlock(nn.Module):
    """Efficient Channel Attention block for lightweight channel reweighting."""

    def __init__(self, kernel_size: int = 3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel attention to the input tensor."""
        y = self.avg_pool(x)                              # (B, C, 1, 1)
        y = y.squeeze(-1).transpose(-1, -2)              # (B, 1, C)
        y = self.conv(y)                                 # (B, 1, C)
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)  # (B, C, 1, 1)
        return x * y.expand_as(x)