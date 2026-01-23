import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch
import numpy as np
from PIL import Image

from config import IMAGE_SIZE
from data import get_transforms


class TestTransforms:
    def test_augmented_output_shape(self, sample_image, transform_augmented):
        """Augmented transforms should produce correct output shape."""
        output = transform_augmented(sample_image)
        assert output.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_non_augmented_output_shape(self, sample_image, transform_no_augment):
        """Non-augmented transforms should produce correct output shape."""
        output = transform_no_augment(sample_image)
        assert output.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_output_is_tensor(self, sample_image, transform_augmented, transform_no_augment):
        """Transforms should output PyTorch tensors."""
        assert isinstance(transform_augmented(sample_image), torch.Tensor)
        assert isinstance(transform_no_augment(sample_image), torch.Tensor)

    def test_normalization_range(self, sample_image, transform_no_augment):
        """Normalized output should be in approximately [-1, 1] range."""
        output = transform_no_augment(sample_image)
        assert output.min() >= -1.5  # Allow some margin
        assert output.max() <= 1.5

    def test_non_augmented_deterministic(self, sample_image, transform_no_augment):
        """Non-augmented transforms should be deterministic."""
        output1 = transform_no_augment(sample_image)
        output2 = transform_no_augment(sample_image)
        assert torch.allclose(output1, output2)

    def test_handles_different_image_sizes(self, transform_no_augment):
        """Transforms should handle various input sizes."""
        for size in [(50, 50), (100, 100), (200, 150), (64, 64)]:
            img = Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8))
            output = transform_no_augment(img)
            assert output.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_handles_grayscale_converted_to_rgb(self, transform_no_augment):
        """Transforms should handle grayscale images converted to RGB."""
        gray_img = Image.fromarray(np.random.randint(0, 255, (100, 100), dtype=np.uint8))
        rgb_img = gray_img.convert('RGB')
        output = transform_no_augment(rgb_img)
        assert output.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
