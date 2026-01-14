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

        # Hook into the last conv layer of the third block (conv3_3)
        # This is the final convolutional layer before pooling
        self.model.conv4_2.register_forward_hook(forward_hook)
        self.model.conv4_2.register_full_backward_hook(backward_hook)

    def generate(self, image, target_class=None):
        """
        Generate Grad-CAM heatmap for an image.

        Args:
            image: Input image tensor (1, 3, H, W)
            target_class: Class to visualize. If None, uses predicted class.

        Returns:
            heatmap: Numpy array with heatmap values
            target_class: The class that was visualized
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

        # Global average pooling on gradients (Grad-CAM weighting)
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # Apply ReLU to focus on positive influences
        cam = np.maximum(cam, 0)
        
        # Normalize to [0, 1]
        cam = cam / (cam.max() + 1e-8)

        return cam, target_class


class GradCAMPlusPlus:
    """
    Grad-CAM++ implementation for improved localization.
    
    As mentioned in the preliminary report, this provides sharper
    localization for subtle facial features.
    
    Usage:
        gradcam_pp = GradCAMPlusPlus(model)
        heatmap = gradcam_pp.generate(image, target_class)
    """

    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        """Set up hooks to capture gradients and activations"""

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        def forward_hook(module, input, output):
            self.activations = output

        self.model.conv4_2.register_forward_hook(forward_hook)
        self.model.conv4_2.register_full_backward_hook(backward_hook)

    def generate(self, image, target_class=None):
        """
        Generate Grad-CAM++ heatmap for an image.

        Args:
            image: Input image tensor (1, 3, H, W)
            target_class: Class to visualize. If None, uses predicted class.

        Returns:
            heatmap: Numpy array with heatmap values
            target_class: The class that was visualized
        """
        # Forward pass
        output = self.model(image)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        class_score = output[0, target_class]
        class_score.backward()

        # Get gradients and activations
        gradients = self.gradients[0].cpu().data.numpy()  # [C, H, W]
        activations = self.activations[0].cpu().data.numpy()  # [C, H, W]

        # Grad-CAM++ weighting
        # Calculate alpha (importance weights for each spatial location)
        gradients_power_2 = gradients ** 2
        gradients_power_3 = gradients ** 3
        
        sum_activations = np.sum(activations, axis=(1, 2))
        eps = 1e-8
        
        alpha = gradients_power_2 / (2 * gradients_power_2 + 
                                      sum_activations[:, None, None] * gradients_power_3 + eps)
        
        # Apply ReLU to gradients
        positive_gradients = np.maximum(gradients, 0)
        
        # Calculate weights
        weights = np.sum(alpha * positive_gradients, axis=(1, 2))
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize to [0, 1]
        cam = cam / (cam.max() + 1e-8)

        return cam, target_class


def visualize_gradcam(image_path, model_path=MODEL_SAVE_PATH, save_path=None, use_gradcam_pp=False):
    """
    Create and save a Grad-CAM visualization for an image.

    Args:
        image_path: Path to input image
        model_path: Path to trained model
        save_path: Where to save visualization (optional)
        use_gradcam_pp: If True, use Grad-CAM++ instead of standard Grad-CAM
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model(model_path).to(device)

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = get_transforms(augment=False)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Generate Grad-CAM
    if use_gradcam_pp:
        gradcam = GradCAMPlusPlus(model)
        method_name = "Grad-CAM++"
    else:
        gradcam = GradCAM(model)
        method_name = "Grad-CAM"
    
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
    axes[1].set_title(f'{method_name} Heatmap')
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
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved {method_name} visualization to {save_path}")
    else:
        plt.show()

    return predicted_class


def compare_gradcam_methods(image_path, model_path=MODEL_SAVE_PATH, save_path=None):
    """
    Compare standard Grad-CAM and Grad-CAM++ side by side.
    
    Args:
        image_path: Path to input image
        model_path: Path to trained model
        save_path: Where to save comparison (optional)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model = load_model(model_path).to(device)

    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    transform = get_transforms(augment=False)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Generate both heatmaps
    gradcam = GradCAM(model)
    heatmap_standard, pred_class = gradcam.generate(image_tensor)
    
    gradcam_pp = GradCAMPlusPlus(model)
    heatmap_pp, _ = gradcam_pp.generate(image_tensor, target_class=pred_class)

    # Resize heatmaps
    heatmap_standard_resized = np.array(
        Image.fromarray(heatmap_standard).resize(image.size, Image.BILINEAR)
    )
    heatmap_pp_resized = np.array(
        Image.fromarray(heatmap_pp).resize(image.size, Image.BILINEAR)
    )

    # Create comparison visualization
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Grad-CAM
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

    # Row 2: Grad-CAM++
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

    # Save or show
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved comparison to {save_path}")
    else:
        plt.show()

    return pred_class


if __name__ == "__main__":
    # Example usage
    image_path = "sad_man.jpg"  # Change this to your image
    
    # Standard Grad-CAM
    save_path = f"{RESULTS_PATH}/visualizations/gradcam_example.png"
    visualize_gradcam(image_path, save_path=save_path)
    
    # Grad-CAM++
    save_path_pp = f"{RESULTS_PATH}/visualizations/gradcam_pp_example.png"
    visualize_gradcam(image_path, save_path=save_path_pp, use_gradcam_pp=True)
    
    # Comparison
    save_path_compare = f"{RESULTS_PATH}/visualizations/gradcam_comparison.png"
    compare_gradcam_methods(image_path, save_path=save_path_compare)
