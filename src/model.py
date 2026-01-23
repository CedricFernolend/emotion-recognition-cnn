import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES, DROPOUT_RATE

class SEBlock(nn.Module):
    """Channel Attention (Squeeze-and-Excitation)"""
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
    def __init__(self, in_channels, out_channels, num_convs, skip=None):
        super().__init__()
        convs = []
        for i in range(num_convs):
            convs.append(nn.Conv2d(in_channels if i==0 else out_channels, out_channels, kernel_size=3, padding=1))
            convs.append(nn.BatchNorm2d(out_channels))
            if i < num_convs - 1:
                convs.append(nn.ReLU(inplace=False))
        
        self.body = nn.Sequential(*convs)
        self.skip = skip if skip else nn.Identity()
        self.se = SEBlock(out_channels)
        self.spatial = SpatialAttention()

    def forward(self, x):
        identity = self.skip(x)
        out = self.body(x)
        out = self.se(out)
        out = out + identity
        out = F.relu(out)
        out = self.spatial(out)
        return out

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(EmotionCNN, self).__init__()
        
        # input: 3 x 64 x 64
        self.layer1 = ImprovedBlock(3, 64, 2, nn.Conv2d(3, 64, 1))   # -> 64 x 64
        self.layer2 = ImprovedBlock(64, 128, 2, nn.Conv2d(64, 128, 1)) # -> 32 x 32 (after pool)
        self.layer3 = ImprovedBlock(128, 256, 3, nn.Conv2d(128, 256, 1)) # -> 16 x 16 (after pool)
        self.layer4 = ImprovedBlock(256, 512, 2, nn.Conv2d(256, 512, 1)) # -> 8 x 8 (after pool)

        self.pool = nn.MaxPool2d(2)
        self.global_pool = nn.AdaptiveAvgPool2d(1) # Final feature: 512 x 1 x 1
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(512, 128),
            nn.ReLU(inplace=False),
            nn.BatchNorm1d(128),
            nn.Dropout(DROPOUT_RATE * 0.6), # Slightly lower dropout on second layer
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.pool(self.layer1(x)) # 64 -> 32
        x = self.pool(self.layer2(x)) # 32 -> 16
        x = self.pool(self.layer3(x)) # 16 -> 8
        x = self.pool(self.layer4(x)) # 8 -> 4
        
        x = self.global_pool(x)
        return self.classifier(x)

def create_model():
    """Create and return a new model instance."""
    from config import NUM_CLASSES
    return EmotionCNN(num_classes=NUM_CLASSES)

def load_model(path):
    """Load a saved model from file."""
    model = create_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(path, map_location=device))
    return model
