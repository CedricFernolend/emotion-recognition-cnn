"""
V1: Baseline 3-Block CNN (from Preliminary Report)

Architecture:
- 3 convolutional blocks
- Block 1: 2 convs (3 -> 64)
- Block 2: 2 convs (64 -> 128)
- Block 3: 3 convs (128 -> 256)
- Simple 1x1 skip connections for channel matching
- NO SE attention, NO spatial attention
- Classifier: GlobalAvgPool -> 128 -> Dropout -> 6
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleBlock(nn.Module):
    """Simple convolutional block with skip connection (no attention)."""

    def __init__(self, in_channels, out_channels, num_convs):
        super().__init__()

        # Build convolutional layers
        layers = []
        for i in range(num_convs):
            layers.append(
                nn.Conv2d(
                    in_channels if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1
                )
            )
            layers.append(nn.BatchNorm2d(out_channels))
            if i < num_convs - 1:
                layers.append(nn.ReLU(inplace=False))

        self.body = nn.Sequential(*layers)

        # Skip connection: 1x1 conv if channels don't match
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.body(x)
        out = out + identity
        out = F.relu(out)
        return out


class EmotionCNN_V1(nn.Module):
    """
    Baseline emotion recognition CNN from preliminary report.

    3-block architecture without attention mechanisms.
    """

    def __init__(self, num_classes=6):
        super().__init__()

        # Input: 3 x 64 x 64
        self.block1 = SimpleBlock(3, 64, num_convs=2)      # -> 64 x 64 x 64
        self.block2 = SimpleBlock(64, 128, num_convs=2)    # -> 128 x 32 x 32 (after pool)
        self.block3 = SimpleBlock(128, 256, num_convs=3)   # -> 256 x 16 x 16 (after pool)

        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # -> 256 x 1 x 1

        # Classifier: simpler than v4
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.pool(self.block1(x))  # 64 -> 32
        x = self.pool(self.block2(x))  # 32 -> 16
        x = self.pool(self.block3(x))  # 16 -> 8

        x = self.global_pool(x)  # 8 -> 1
        return self.classifier(x)


def create_model():
    """Create and return a new V1 model instance."""
    return EmotionCNN_V1(num_classes=6)
