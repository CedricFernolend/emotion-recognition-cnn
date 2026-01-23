import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch
import tempfile
import numpy as np
from PIL import Image

from config import IMAGE_SIZE, NUM_CLASSES, EMOTION_LABELS
from data import EmotionDataset, get_transforms


class TestEmotionDataset:
    @pytest.fixture
    def temp_dataset_dir(self):
        """Create a temporary dataset directory with sample images."""
        with tempfile.TemporaryDirectory() as tmpdir:
            train_dir = os.path.join(tmpdir, 'train')
            os.makedirs(train_dir)

            for emotion in EMOTION_LABELS[:3]:  # Create 3 emotions
                emotion_dir = os.path.join(train_dir, emotion)
                os.makedirs(emotion_dir)

                for i in range(2):  # 2 images per emotion
                    img = Image.fromarray(np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8))
                    img.save(os.path.join(emotion_dir, f'img_{i}.jpg'))

            yield tmpdir

    def test_dataset_initialization(self, temp_dataset_dir):
        """Dataset should initialize without errors."""
        transform = get_transforms(augment=False)
        dataset = EmotionDataset(temp_dataset_dir, 'train', transform=transform)
        assert len(dataset) == 6  # 3 emotions * 2 images

    def test_dataset_returns_tuple(self, temp_dataset_dir):
        """Dataset should return (image, label) tuple."""
        transform = get_transforms(augment=False)
        dataset = EmotionDataset(temp_dataset_dir, 'train', transform=transform)
        image, label = dataset[0]
        assert isinstance(image, torch.Tensor)
        assert isinstance(label, int)

    def test_image_tensor_shape(self, temp_dataset_dir):
        """Image tensor should have correct shape."""
        transform = get_transforms(augment=False)
        dataset = EmotionDataset(temp_dataset_dir, 'train', transform=transform)
        image, _ = dataset[0]
        assert image.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_label_valid_range(self, temp_dataset_dir):
        """Labels should be valid indices."""
        transform = get_transforms(augment=False)
        dataset = EmotionDataset(temp_dataset_dir, 'train', transform=transform)
        for i in range(len(dataset)):
            _, label = dataset[i]
            assert 0 <= label < NUM_CLASSES

    def test_empty_split_handling(self, temp_dataset_dir):
        """Dataset should handle empty splits gracefully."""
        transform = get_transforms(augment=False)
        dataset = EmotionDataset(temp_dataset_dir, 'val', transform=transform)
        assert len(dataset) == 0

    def test_without_transform(self, temp_dataset_dir):
        """Dataset should work without transform (returns PIL Image)."""
        dataset = EmotionDataset(temp_dataset_dir, 'train', transform=None)
        image, label = dataset[0]
        assert isinstance(image, Image.Image)
        assert isinstance(label, int)
