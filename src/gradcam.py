import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from config import EMOTION_LABELS, MODEL_SAVE_PATH, RESULTS_PATH
from model import load_model
from data import get_transforms


class BaseGradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        
        # FIX: Point to the last conv layer in the new architecture
        # layer4 is the last block, body is the Sequential, -2 is the last BatchNorm, -3 is the last Conv
        if target_layer is None:
            self.target_layer = model.layer4.body[-3] 
        else:
            self.target_layer = target_layer
            
        self._register_hooks()
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

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


def visualize_gradcam(image_path, model_path=MODEL_SAVE_PATH, save_path=None, use_gradcam_pp=False):
    """Create and save a Grad-CAM visualization for an image."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(model_path).to(device)

    image = Image.open(image_path).convert('RGB')
    transform = get_transforms(augment=False)
    image_tensor = transform(image).unsqueeze(0).to(device)

    gradcam_class = GradCAMPlusPlus if use_gradcam_pp else GradCAM
    method_name = "Grad-CAM++" if use_gradcam_pp else "Grad-CAM"

    gradcam = gradcam_class(model)
    heatmap, predicted_class = gradcam.generate(image_tensor)

    heatmap_resized = np.array(Image.fromarray(heatmap).resize(image.size, Image.BILINEAR))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title(f'{method_name} Heatmap')
    axes[1].axis('off')

    axes[2].imshow(image)
    axes[2].imshow(heatmap_resized, cmap='jet', alpha=0.5)
    axes[2].set_title(f'Prediction: {EMOTION_LABELS[predicted_class]}')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved {method_name} visualization to {save_path}")
    else:
        plt.show()

    return predicted_class


def compare_gradcam_methods(image_path, model_path=MODEL_SAVE_PATH, save_path=None):
    """Compare standard Grad-CAM and Grad-CAM++ side by side."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(model_path).to(device)

    image = Image.open(image_path).convert('RGB')
    transform = get_transforms(augment=False)
    image_tensor = transform(image).unsqueeze(0).to(device)

    gradcam = GradCAM(model)
    heatmap_standard, pred_class = gradcam.generate(image_tensor)

    gradcam_pp = GradCAMPlusPlus(model)
    heatmap_pp, _ = gradcam_pp.generate(image_tensor, target_class=pred_class)

    heatmap_standard_resized = np.array(
        Image.fromarray(heatmap_standard).resize(image.size, Image.BILINEAR)
    )
    heatmap_pp_resized = np.array(
        Image.fromarray(heatmap_pp).resize(image.size, Image.BILINEAR)
    )

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original Image')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(heatmap_standard_resized, cmap='jet')
    axes[0, 1].set_title('Grad-CAM Heatmap')
    axes[0, 1].axis('off')

    axes[0, 2].imshow(image)
    axes[0, 2].imshow(heatmap_standard_resized, cmap='jet', alpha=0.5)
    axes[0, 2].set_title('Grad-CAM Overlay')
    axes[0, 2].axis('off')

    axes[1, 0].imshow(image)
    axes[1, 0].set_title('Original Image')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(heatmap_pp_resized, cmap='jet')
    axes[1, 1].set_title('Grad-CAM++ Heatmap')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(image)
    axes[1, 2].imshow(heatmap_pp_resized, cmap='jet', alpha=0.5)
    axes[1, 2].set_title(f'Grad-CAM++ Overlay\nPrediction: {EMOTION_LABELS[pred_class]}')
    axes[1, 2].axis('off')

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()

    return pred_class


if __name__ == "__main__":
    image_path = "my_face_2.jpg"

    save_path = f"{RESULTS_PATH}/visualizations/gradcam_example.png"
    visualize_gradcam(image_path, save_path=save_path)

    save_path_pp = f"{RESULTS_PATH}/visualizations/gradcam_pp_example.png"
    visualize_gradcam(image_path, save_path=save_path_pp, use_gradcam_pp=True)

    save_path_compare = f"{RESULTS_PATH}/visualizations/gradcam_comparison.png"
    compare_gradcam_methods(image_path, save_path=save_path_compare)
