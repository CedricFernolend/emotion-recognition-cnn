# Grad-CAM for visualizing what the model looks at

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

from config import EMOTION_LABELS, MODEL_SAVE_PATH, RESULTS_PATH
from model import load_model
from data import get_transforms


class GradCAM:
    """
    Grad-CAM implementation to visualize where the model is looking.

    Usage:
        gradcam = GradCAM(model)
        heatmap = gradcam.generate(image, target_class)
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None

        # Hook into the last convolutional layer
        self.hook_layers()

    def hook_layers(self):
        """Set up hooks to capture gradients and activations"""

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        # Hook into conv3 (last conv layer)
        self.model.conv3.register_forward_hook(forward_hook)
        self.model.conv3.register_full_backward_hook(backward_hook)

    def generate(self, image, target_class=None):
        """
        Generate Grad-CAM heatmap for an image.

        Args:
            image: Input image tensor (1, 3, H, W)
            target_class: Class to visualize. If None, uses predicted class.

        Returns:
            heatmap: Numpy array with heatmap values
        """
        # Forward pass
        output = self.model(image)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        # Generate heatmap
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()

        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)

        return cam, target_class


def visualize_gradcam(image_path, model_path=MODEL_SAVE_PATH, save_path=None):
    """
    Create and save a Grad-CAM visualization for an image.

    Args:
        image_path: Path to input image
        model_path: Path to trained model
        save_path: Where to save visualization (optional)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model(model_path).to(device)

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = get_transforms(augment=False)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Generate Grad-CAM
    gradcam = GradCAM(model)
    heatmap, predicted_class = gradcam.generate(image_tensor)

    # Resize heatmap to match image
    heatmap_resized = np.array(Image.fromarray(heatmap).resize(image.size, Image.BILINEAR))

    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # Heatmap
    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')

    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(heatmap_resized, cmap='jet', alpha=0.5)
    axes[2].set_title(f'Prediction: {EMOTION_LABELS[predicted_class]}')
    axes[2].axis('off')

    plt.tight_layout()

    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"Saved Grad-CAM visualization to {save_path}")
    else:
        plt.show()

    return predicted_class


if __name__ == "__main__":
    # Example usage
    image_path = "data/test/happiness/example.jpg"  # Change this to your image
    save_path = f"{RESULTS_PATH}/visualizations/gradcam_example.png"
    visualize_gradcam(image_path, save_path=save_path)
