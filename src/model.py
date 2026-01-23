import torch
import torch.nn as nn
from config import IMAGE_SIZE, NUM_CLASSES, DROPOUT_RATE


class SpatialAttention(nn.Module):
    """Spatial attention mechanism to focus on important facial regions."""

    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        combined = torch.cat([max_pool, avg_pool], dim=1)
        attention_map = self.sigmoid(self.conv(combined))
        return x * attention_map


class EmotionCNN(nn.Module):
    """
    CNN for classifying facial expressions into 6 emotion categories.

    Architecture: 4 conv blocks (64->128->256->512 filters) with spatial attention,
    skip connections, batch normalization, and dropout regularization.
    """

    def __init__(self):
        super(EmotionCNN, self).__init__()

        self.conv1_1, self.bn1_1 = self._conv_bn(3, 64)
        self.conv1_2, self.bn1_2 = self._conv_bn(64, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.attention1 = SpatialAttention()

        self.conv2_1, self.bn2_1 = self._conv_bn(64, 128)
        self.conv2_2, self.bn2_2 = self._conv_bn(128, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.attention2 = SpatialAttention()

        self.conv3_1, self.bn3_1 = self._conv_bn(128, 256)
        self.conv3_2, self.bn3_2 = self._conv_bn(256, 256)
        self.conv3_3, self.bn3_3 = self._conv_bn(256, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.attention3 = SpatialAttention()

        self.conv4_1, self.bn4_1 = self._conv_bn(256, 512)
        self.conv4_2, self.bn4_2 = self._conv_bn(512, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.attention4 = SpatialAttention()

        self.skip1 = nn.Conv2d(3, 64, kernel_size=1)
        self.skip2 = nn.Conv2d(64, 128, kernel_size=1)
        self.skip3 = nn.Conv2d(128, 256, kernel_size=1)
        self.skip4 = nn.Conv2d(256, 512, kernel_size=1)

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, NUM_CLASSES)
        )

    def _conv_bn(self, in_channels, out_channels):
        """Create a conv layer with batch normalization."""
        conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        bn = nn.BatchNorm2d(out_channels)
        return conv, bn

    def _forward_block(self, x, convs, bns, pool, attention, skip, dropout_p=0.3):
        """Forward pass through a single conv block."""
        identity = skip(x)
        for conv, bn in zip(convs, bns):
            x = torch.relu(bn(conv(x)))
        x = x + identity
        x = pool(x)
        x = attention(x)
        x = nn.functional.dropout2d(x, p=dropout_p, training=self.training)
        return x

    def forward(self, x):
        x = self._forward_block(
            x,
            [self.conv1_1, self.conv1_2],
            [self.bn1_1, self.bn1_2],
            self.pool1, self.attention1, self.skip1
        )

        x = self._forward_block(
            x,
            [self.conv2_1, self.conv2_2],
            [self.bn2_1, self.bn2_2],
            self.pool2, self.attention2, self.skip2
        )

        x = self._forward_block(
            x,
            [self.conv3_1, self.conv3_2, self.conv3_3],
            [self.bn3_1, self.bn3_2, self.bn3_3],
            self.pool3, self.attention3, self.skip3
        )

        x = self._forward_block(
            x,
            [self.conv4_1, self.conv4_2],
            [self.bn4_1, self.bn4_2],
            self.pool4, self.attention4, self.skip4
        )

        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def create_model():
    """Create and return a new model instance."""
    return EmotionCNN()


def load_model(path):
    """Load a saved model from file."""
    model = create_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(path, map_location=device))
    return model
