import sys
import os
import pytest
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config import IMAGE_SIZE, NUM_CLASSES, EMOTION_LABELS
from model import create_model, EmotionCNN
from data import get_transforms


@pytest.fixture
def model():
    """Create a fresh model instance for testing."""
    return create_model()


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@pytest.fixture
def sample_input():
    """Create a sample input tensor matching expected input shape."""
    return torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)


@pytest.fixture
def batch_input():
    """Create a batch of sample inputs."""
    return torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)


@pytest.fixture
def sample_image():
    """Create a sample PIL Image for testing transforms."""
    return Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))


@pytest.fixture
def transform_augmented():
    """Get augmented transforms."""
    return get_transforms(augment=True)


@pytest.fixture
def transform_no_augment():
    """Get non-augmented transforms."""
    return get_transforms(augment=False)
