"""
V2 Model: 4-Block CNN with SE Attention

Changes from V1:
- Added 4th block (256 -> 512 channels)
- Added SE (Squeeze-and-Excitation) attention to each block
- Lower learning rate (0.0001) for more stable training
- Keeping class weights from V1

Architecture:
    Input: 64x64x3
        |
    Block 1: 3->64 (2 convs) + SE -> Pool -> 32x32
    Block 2: 64->128 (2 convs) + SE -> Pool -> 16x16
    Block 3: 128->256 (3 convs) + SE -> Pool -> 8x8
    Block 4: 256->512 (2 convs) + SE -> Pool -> 4x4   <- NEW
        |
    Global Average Pool -> 512
        |
    FC(512->128) -> ReLU -> Dropout -> FC(128->6)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation block for channel attention.

    Learns to weight channels based on their importance.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze: global average pooling
        y = self.squeeze(x).view(b, c)
        # Excitation: FC layers to learn channel weights
        y = self.excitation(y).view(b, c, 1, 1)
        # Scale: multiply input by learned weights
        return x * y.expand_as(x)


class SEBlock_V2(nn.Module):
    """
    Convolutional block with SE attention and skip connection.
    """

    def __init__(self, in_channels, out_channels, num_convs, se_reduction=16):
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
                layers.append(nn.ReLU(inplace=True))

        self.body = nn.Sequential(*layers)

        # SE attention
        self.se = SEBlock(out_channels, reduction=se_reduction)

        # Skip connection: 1x1 conv + BatchNorm if channels don't match
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        identity = self.skip(x)
        out = self.body(x)
        out = self.se(out)  # Apply SE attention
        out = out + identity
        out = F.relu(out)
        return out


class EmotionCNN_V2(nn.Module):
    """
    V2 Emotion recognition CNN with SE attention.

    4-block architecture with Squeeze-and-Excitation attention.
    """

    def __init__(self, num_classes=6, dropout_rate=0.5):
        super().__init__()

        # Input: 3 x 64 x 64
        self.block1 = SEBlock_V2(3, 64, num_convs=2)      # -> 64 x 64 x 64
        self.block2 = SEBlock_V2(64, 128, num_convs=2)    # -> 128 x 32 x 32 (after pool)
        self.block3 = SEBlock_V2(128, 256, num_convs=3)   # -> 256 x 16 x 16 (after pool)
        self.block4 = SEBlock_V2(256, 512, num_convs=2)   # -> 512 x 8 x 8 (after pool) <- NEW

        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d(1)  # -> 512 x 1 x 1

        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.pool(self.block1(x))  # 64 -> 32
        x = self.pool(self.block2(x))  # 32 -> 16
        x = self.pool(self.block3(x))  # 16 -> 8
        x = self.pool(self.block4(x))  # 8 -> 4

        x = self.global_pool(x)  # 4 -> 1
        return self.classifier(x)


def create_model():
    """Create and return a new V2 model instance."""
    return EmotionCNN_V2(num_classes=6)
