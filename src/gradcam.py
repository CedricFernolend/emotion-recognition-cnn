import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Ensure these imports match your file structure
from config import EMOTION_LABELS, MODEL_SAVE_PATH, RESULTS_PATH
from model import load_model 

class BaseGradCAM:
    """Base class for Grad-CAM implementations."""

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        
        # TARGET LAYER LOGIC FOR NEW ARCHITECTURE
        # layer4 is the final block, body is the Sequential list
        # body[-3] is the last Conv2d layer in that block
        if target_layer is None:
            self.target_layer = model.layer4.body[-3]
        else:
            self.target_layer = target_layer
            
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            # FIX: Add .clone() and .detach() to avoid the "view modification" error
            if grad_output[0] is not None:
                self.gradients = grad_output[0].clone().detach()

        def forward_hook(module, input, output):
            # FIX: Add .detach() here as well for safety
            self.activations = output.detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def _compute_weights(self, gradients, activations):
        """Compute weights for activation maps. Override in subclasses."""
        raise NotImplementedError

    def generate(self, image, target_class=None):
        """Generate heatmap for an image."""
        output = self.model(image)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = self._compute_weights(gradients, activations)

        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)

        return cam, target_class


class GradCAM(BaseGradCAM):
    """Standard Grad-CAM implementation."""
    def _compute_weights(self, gradients, activations):
        return np.mean(gradients, axis=(1, 2))


class GradCAMPlusPlus(BaseGradCAM):
    """Grad-CAM++ for improved localization."""
    def _compute_weights(self, gradients, activations):
        gradients_power_2 = gradients ** 2
        gradients_power_3 = gradients ** 3
        sum_activations = np.sum(activations, axis=(1, 2))
        eps = 1e-8
        alpha = gradients_power_2 / (
            2 * gradients_power_2 + sum_activations[:, None, None] * gradients_power_3 + eps
        )
        positive_gradients = np.maximum(gradients, 0)
        return np.sum(alpha * positive_gradients, axis=(1, 2))
