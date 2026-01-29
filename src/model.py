"""
Model file - backward compatible wrapper.

This file maintains backward compatibility with existing code.
For version-specific models, use: from models import create_model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import NUM_CLASSES, DROPOUT_RATE

# Import the actual implementation from v4_final
from models.v4_final import SEBlock, SpatialAttention, ImprovedBlock, EmotionCNN


def create_model():
    """Create and return a new model instance (v4 by default)."""
    return EmotionCNN(num_classes=NUM_CLASSES)


def load_model(path):
    """Load a saved model from file."""
    model = create_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(path, map_location=device))
    return model
