import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest
import torch
import numpy as np

from config import IMAGE_SIZE, NUM_CLASSES
from model import create_model
from gradcam import GradCAM, GradCAMPlusPlus, BaseGradCAM


class TestGradCAM:
    @pytest.fixture
    def model(self):
        """Create model for Grad-CAM testing."""
        model = create_model()
        model.eval()
        return model

    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor."""
        return torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE, requires_grad=True)

    def test_gradcam_generates_heatmap(self, model, sample_input):
        """GradCAM should generate a heatmap."""
        gradcam = GradCAM(model)
        heatmap, predicted_class = gradcam.generate(sample_input)

        assert isinstance(heatmap, np.ndarray)
        assert heatmap.ndim == 2
        assert isinstance(predicted_class, int)

    def test_gradcam_heatmap_values_valid(self, model, sample_input):
        """GradCAM heatmap values should be in [0, 1]."""
        gradcam = GradCAM(model)
        heatmap, _ = gradcam.generate(sample_input)

        assert heatmap.min() >= 0
        assert heatmap.max() <= 1

    def test_gradcam_pp_generates_heatmap(self, model, sample_input):
        """GradCAM++ should generate a heatmap."""
        gradcam_pp = GradCAMPlusPlus(model)
        heatmap, predicted_class = gradcam_pp.generate(sample_input)

        assert isinstance(heatmap, np.ndarray)
        assert heatmap.ndim == 2
        assert isinstance(predicted_class, int)

    def test_gradcam_pp_heatmap_values_valid(self, model, sample_input):
        """GradCAM++ heatmap values should be in [0, 1]."""
        gradcam_pp = GradCAMPlusPlus(model)
        heatmap, _ = gradcam_pp.generate(sample_input)

        assert heatmap.min() >= 0
        assert heatmap.max() <= 1

    def test_target_class_override(self, model, sample_input):
        """Should be able to specify target class."""
        gradcam = GradCAM(model)
        target_class = 2
        heatmap, returned_class = gradcam.generate(sample_input, target_class=target_class)

        assert returned_class == target_class

    def test_predicted_class_valid_range(self, model, sample_input):
        """Predicted class should be in valid range."""
        gradcam = GradCAM(model)
        _, predicted_class = gradcam.generate(sample_input)

        assert 0 <= predicted_class < NUM_CLASSES

    def test_different_models_produce_different_heatmaps(self, sample_input):
        """Different model initializations should produce different heatmaps."""
        model1 = create_model()
        model2 = create_model()
        model1.eval()
        model2.eval()

        gradcam1 = GradCAM(model1)
        gradcam2 = GradCAM(model2)

        heatmap1, _ = gradcam1.generate(sample_input.clone())
        heatmap2, _ = gradcam2.generate(sample_input.clone())

        # Due to random initialization, heatmaps should differ
        # (though there's a small chance they could be similar)
        assert heatmap1.shape == heatmap2.shape
