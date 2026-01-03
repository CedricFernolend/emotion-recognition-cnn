# CNN model for emotion recognition

import torch
import torch.nn as nn
from config import IMAGE_SIZE, NUM_CLASSES, DROPOUT_RATE


class EmotionCNN(nn.Module):
    """
    CNN for classifying facial expressions into 6 emotion categories.

    To modify the architecture:
    - Add/remove Conv blocks in __init__
    - Change number of filters (currently 32, 64, 128)
    - Adjust fully connected layer sizes
    """

    def __init__(self):
        super(EmotionCNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Second convolutional block
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Third convolutional block
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
            nn.Linear(64, NUM_CLASSES)
        )

    def forward(self, x):
        """Forward pass through the network"""
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
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
    model.load_state_dict(torch.load(path))
    return model
