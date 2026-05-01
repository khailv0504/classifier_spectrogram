import torch
from torch import nn

from classifier_spectrogram.src.classifier_spectrogram.module.mbConvLite import MBConvLite

def init_weights_he(m):
    """Initialize model layers with He-style defaults.

    Args:
        m: A module instance passed by ``model.apply``.
    """
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

class WaveCNN(nn.Module):
    """Compact CNN backbone/head stack for 12-class radar image classification."""

    def __init__(self, num_classes: int):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.SiLU(inplace=True)
        )

        self.features = nn.Sequential(
            MBConvLite(16, 24, stride=1, expand_ratio=2, use_attention=True),
            MBConvLite(24, 32, stride=2, expand_ratio=2, use_attention=True),
            MBConvLite(32, 48, stride=1, expand_ratio=2, use_attention=True),
            MBConvLite(48, 64, stride=2, expand_ratio=2, use_attention=True),
            MBConvLite(64, 80, stride=1, expand_ratio=2, use_attention=True),
            MBConvLite(80, 96, stride=2, expand_ratio=2, use_attention=True),
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(96, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        return x