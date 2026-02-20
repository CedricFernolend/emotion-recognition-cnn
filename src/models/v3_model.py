"""
V4: Final Model with Full Enhancements

Architecture:
- 4 ImprovedBlocks with SE + Spatial attention
- Filter progression: 64 -> 128 -> 256 -> 512
- Graduated dropout in classifier
- Full feature set from current production model
- Adaptive spatial attention kernel sizes (7->7->7->3) to match feature map sizes
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Channel Attention (Squeezeq-and-Excitation)"""

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=False),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SpatialAttention(nn.Module):
    """Spatial Attention to focus on key facial landmarks."""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.sigmoid(self.conv(y))
        return x * y


class ImprovedBlock(nn.Module):
    """Residual Block with SE and Spatial Attention."""

    def __init__(self, in_channels, out_channels, num_convs, spatial_kernel_size=7):
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

        # Skip connection: 1x1 conv + BatchNorm if channels don't match (like V1/V2)
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

        self.se = SEBlock(out_channels)
        self.spatial = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x):
        identity = self.skip(x)
        out = self.body(x)
        out = self.se(out)
        out = out + identity
        out = F.relu(out)
        out = self.spatial(out)
        return out


class EmotionCNN(nn.Module):
    """Final emotion recognition CNN with full enhancements."""

    def __init__(self, num_classes=6, dropout_rate=0.5):
        super(EmotionCNN, self).__init__()

        # Input: 3 x 64 x 64
        # Spatial attention kernel sizes adapted to feature map sizes:
        # layer1: 64x64 -> kernel=7, layer2: 32x32 -> kernel=7
        # layer3: 16x16 -> kernel=7, layer4: 8x8 -> kernel=3 (smaller map needs smaller kernel)
        self.layer1 = ImprovedBlock(3, 64, 2, spatial_kernel_size=7)      # 64x64
        self.layer2 = ImprovedBlock(64, 128, 2, spatial_kernel_size=7)    # 32x32
        self.layer3 = ImprovedBlock(128, 256, 3, spatial_kernel_size=7)   # 16x16
        self.layer4 = ImprovedBlock(256, 512, 2, spatial_kernel_size=3)   # 8x8

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
    """Create and return a new V4 model instance."""
    return EmotionCNN(num_classes=6)
