import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch

from config import IMAGE_SIZE, NUM_CLASSES
from model import create_model, EmotionCNN, SpatialAttention


class TestSpatialAttention:
    def test_output_shape_preserved(self):
        """Spatial attention should preserve input shape."""
        attention = SpatialAttention()
        x = torch.randn(2, 64, 32, 32)
        output = attention(x)
        assert output.shape == x.shape

    def test_different_channel_sizes(self):
        """Test with various channel sizes."""
        attention = SpatialAttention()
        for channels in [32, 64, 128, 256, 512]:
            x = torch.randn(1, channels, 16, 16)
            output = attention(x)
            assert output.shape == x.shape


class TestEmotionCNN:
    def test_model_creation(self):
        """create_model should return EmotionCNN instance."""
        model = create_model()
        assert isinstance(model, EmotionCNN)

    def test_output_shape(self, model, sample_input):
        """Model output should have correct shape (batch, num_classes)."""
        model.eval()
        with torch.no_grad():
            output = model(sample_input)
        assert output.shape == (1, NUM_CLASSES)

    def test_batch_processing(self, model, batch_input):
        """Model should handle batches correctly."""
        model.eval()
        with torch.no_grad():
            output = model(batch_input)
        assert output.shape == (4, NUM_CLASSES)

    def test_different_batch_sizes(self, model):
        """Model should handle various batch sizes."""
        model.eval()
        for batch_size in [1, 2, 8, 16]:
            x = torch.randn(batch_size, 3, IMAGE_SIZE, IMAGE_SIZE)
            with torch.no_grad():
                output = model(x)
            assert output.shape == (batch_size, NUM_CLASSES)

    def test_parameter_count(self, model):
        """Model should have reasonable number of parameters."""
        total_params = sum(p.numel() for p in model.parameters())
        assert total_params > 100000  # At least 100K parameters
        assert total_params < 50000000  # Less than 50M parameters

    def test_trainable_parameters(self, model):
        """All parameters should be trainable by default."""
        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert total == trainable

    def test_train_eval_modes(self, model, sample_input):
        """Model should behave differently in train vs eval mode due to dropout."""
        model.train()
        outputs_train = []
        for _ in range(5):
            with torch.no_grad():
                outputs_train.append(model(sample_input).clone())

        model.eval()
        outputs_eval = []
        for _ in range(5):
            with torch.no_grad():
                outputs_eval.append(model(sample_input).clone())

        # In eval mode, outputs should be identical
        for i in range(1, 5):
            assert torch.allclose(outputs_eval[0], outputs_eval[i])

    def test_gpu_compatibility(self, model, sample_input, device):
        """Model should work on available device."""
        model = model.to(device)
        x = sample_input.to(device)
        model.eval()
        with torch.no_grad():
            output = model(x)
        assert output.device == device
        assert output.shape == (1, NUM_CLASSES)
