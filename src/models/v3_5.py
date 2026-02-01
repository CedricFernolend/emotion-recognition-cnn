"""
V3.5: V4 Architecture without Attention Mechanisms

Architecture:
- 4 ResidualBlocks (no SE or Spatial attention)
- Filter progression: 64 -> 128 -> 256 -> 512
- Graduated dropout in classifier
- Same structure as V4, but without attention mechanisms
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """Residual Block without attention mechanisms."""

    def __init__(self, in_channels, out_channels, num_convs, skip=None):
        super().__init__()
        convs = []
        for i in range(num_convs):
            convs.append(nn.Conv2d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3,
                padding=1
            ))
            convs.append(nn.BatchNorm2d(out_channels))
            if i < num_convs - 1:
                convs.append(nn.ReLU(inplace=False))

        self.body = nn.Sequential(*convs)
        self.skip = skip if skip else nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.body(x)
        out = out + identity
        return F.relu(out)


class EmotionCNN(nn.Module):
    """Emotion recognition CNN without attention mechanisms (V4 architecture base)."""

    def __init__(self, num_classes=6, dropout_rate=0.5):
        super(EmotionCNN, self).__init__()

        # Input: 3 x 64 x 64
        self.layer1 = ResidualBlock(3, 64, 2, nn.Conv2d(3, 64, 1))      # -> 64 x 64
        self.layer2 = ResidualBlock(64, 128, 2, nn.Conv2d(64, 128, 1))  # -> 32 x 32 (after pool)
        self.layer3 = ResidualBlock(128, 256, 3, nn.Conv2d(128, 256, 1))  # -> 16 x 16 (after pool)
        self.layer4 = ResidualBlock(256, 512, 2, nn.Conv2d(256, 512, 1))  # -> 8 x 8 (after pool)

        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # Final feature: 512 x 1 x 1

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 128),
            nn.ReLU(inplace=False),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout_rate * 0.6),  # Slightly lower dropout on second layer
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.pool(self.layer1(x))  # 64 -> 32
        x = self.pool(self.layer2(x))  # 32 -> 16
        x = self.pool(self.layer3(x))  # 16 -> 8
        x = self.pool(self.layer4(x))  # 8 -> 4

        x = self.global_pool(x)
        return self.classifier(x)


def create_model():
    """Create and return a new V3.5 model instance."""
    return EmotionCNN(num_classes=6)
