# CNN model for emotion recognition

import torch
import torch.nn as nn
from config import IMAGE_SIZE, NUM_CLASSES, DROPOUT_RATE


class SpatialAttention(nn.Module):
    """
    Spatial attention mechanism to focus on important facial regions.
    Helps model learn WHERE to look in the image.
    """
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Generate attention map from max and avg pooling
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        combined = torch.cat([max_pool, avg_pool], dim=1)
        attention_map = self.sigmoid(self.conv(combined))
        return x * attention_map  # Apply attention weights


class EmotionCNN(nn.Module):
    """
    Improved CNN for classifying facial expressions into 6 emotion categories.
    
    Architecture improvements:
    - 4 convolutional blocks (increased depth)
    - More filters: 64 → 128 → 256 → 512 (increased capacity)
    - Spatial attention after each block (improved interpretability)
    - Skip connections for better gradient flow
    - Dropout2d after conv blocks + dropout in FC layers
    - BatchNorm for training stability
    
    To modify the architecture:
    - Add/remove Conv blocks in __init__
    - Change number of filters
    - Adjust fully connected layer sizes
    """

    def __init__(self):
        super(EmotionCNN, self).__init__()

        # First convolutional block (2 conv layers, 64 filters)
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2)
        self.attention1 = SpatialAttention()

        # Second convolutional block (2 conv layers, 128 filters)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2)
        self.attention2 = SpatialAttention()

        # Third convolutional block (3 conv layers, 256 filters)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2)
        self.attention3 = SpatialAttention()

        # Fourth convolutional block (2 conv layers, 512 filters) - NEW!
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2)
        self.attention4 = SpatialAttention()

        # Skip connection projections (for when dimensions change)
        self.skip1 = nn.Conv2d(3, 64, kernel_size=1)
        self.skip2 = nn.Conv2d(64, 128, kernel_size=1)
        self.skip3 = nn.Conv2d(128, 256, kernel_size=1)
        self.skip4 = nn.Conv2d(256, 512, kernel_size=1)

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(512, 128),  # 512 from fourth conv block
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, NUM_CLASSES)
        )

    def forward(self, x):
        """
        Forward pass with skip connections, spatial attention, and regularization.
        
        Architecture flow:
        Input (3×64×64) 
        → Conv Block 1 + Attention → 64×32×32
        → Conv Block 2 + Attention → 128×16×16
        → Conv Block 3 + Attention → 256×8×8
        → Conv Block 4 + Attention → 512×4×4
        → Global Pool → 512×1×1
        → FC Layers → NUM_CLASSES
        """
        
        # First block with skip connection and attention
        identity = self.skip1(x)
        x = torch.relu(self.bn1_1(self.conv1_1(x)))
        x = torch.relu(self.bn1_2(self.conv1_2(x)))
        x = x + identity  # Skip connection
        x = self.pool1(x)
        x = self.attention1(x)  # Spatial attention
        x = nn.functional.dropout2d(x, p=0.3, training=self.training)  # Regularization
        
        # Second block with skip connection and attention
        identity = self.skip2(x)
        x = torch.relu(self.bn2_1(self.conv2_1(x)))
        x = torch.relu(self.bn2_2(self.conv2_2(x)))
        x = x + identity  # Skip connection
        x = self.pool2(x)
        x = self.attention2(x)  # Spatial attention
        x = nn.functional.dropout2d(x, p=0.3, training=self.training)  # Regularization
        
        # Third block with skip connection and attention (3 conv layers)
        identity = self.skip3(x)
        x = torch.relu(self.bn3_1(self.conv3_1(x)))
        x = torch.relu(self.bn3_2(self.conv3_2(x)))
        x = torch.relu(self.bn3_3(self.conv3_3(x)))
        x = x + identity  # Skip connection
        x = self.pool3(x)
        x = self.attention3(x)  # Spatial attention
        x = nn.functional.dropout2d(x, p=0.3, training=self.training)  # Regularization
        
        # Fourth block with skip connection and attention (NEW!)
        identity = self.skip4(x)
        x = torch.relu(self.bn4_1(self.conv4_1(x)))
        x = torch.relu(self.bn4_2(self.conv4_2(x)))
        x = x + identity  # Skip connection
        x = self.pool4(x)
        x = self.attention4(x)  # Spatial attention
        x = nn.functional.dropout2d(x, p=0.3, training=self.training)  # Regularization
        
        # Global pooling and classification
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x


def create_model():
    """Create and return a new model instance"""
    return EmotionCNN()


def load_model(path):
    """Load a saved model from file"""
    model = create_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(path, map_location=device))
    return model
