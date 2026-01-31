"""
Generate GradCAM visualizations for each emotion and create comparison grids.

This script:
1. Generates grad_{emotion}.png for each model version (v1, v2, v4)
2. Creates a comparison grid showing all models side-by-side

Usage:
    python generate_gradcam_comparison.py
    python generate_gradcam_comparison.py --generate-only    # Only generate individual GradCAMs
    python generate_gradcam_comparison.py --compare-only     # Only create comparison (uses existing files)
"""
import os
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Setup
VERSIONS = ['v1', 'v2', 'v4']
VERSION_NAMES = {'v1': 'V1', 'v2': 'V2', 'v4': 'V3'}  # v4 displayed as V3
COLORS = {'v1': '#3498db', 'v2': '#2ecc71', 'v4': '#e74c3c'}
EMOTION_LABELS = ['happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
EMOTION_DISPLAY = ['Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear']

RESULTS_PATH = '../results'
DATA_PATH = '../data/raw/test'
OUTPUT_DIR = '../results/presentation'


class GradCAM:
    """Grad-CAM implementation for visualization."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []

        # Register hooks
        self.handles.append(target_layer.register_forward_hook(self._save_activation))
        self.handles.append(target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        """Generate GradCAM heatmap."""
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        # Resize to input size
        cam = F.interpolate(cam, size=(64, 64), mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy(), output

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()


def get_gradcam_target_layer(model, version):
    """Get target layer for GradCAM based on model version."""
    if version == 'v1':
        # V1: 3-block model, target last conv in block3
        return model.block3.body[-2]  # Conv2d before final BN
    elif version == 'v2':
        # V2: 4-block model with SE, target last conv in block4
        return model.block4.body[-2]  # Conv2d before final BN
    else:  # v4
        # V4: 4-block model with SE + Spatial attention
        # body[-2] is the last Conv2d (body[-3] was ReLU - wrong!)
        return model.layer4.body[-2]  # Conv2d before final BN


def get_sample_image_for_emotion(emotion):
    """Get the grad_{emotion}.jpg image for the given emotion."""
    # Use the pre-selected grad_{emotion}.jpg file
    grad_image_path = os.path.join(DATA_PATH, emotion, f'grad_{emotion}.jpg')
    if os.path.exists(grad_image_path):
        return grad_image_path

    # Fallback: try to find any image
    emotion_dir = os.path.join(DATA_PATH, emotion)
    if not os.path.exists(emotion_dir):
        return None

    images = [f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
    if not images:
        return None

    # Use consistent random seed for reproducibility
    random.seed(42 + EMOTION_LABELS.index(emotion))
    return os.path.join(emotion_dir, random.choice(images))


def generate_gradcam_for_version(version, device):
    """Generate GradCAM images for all emotions for a specific model version."""
    from models import load_model

    print(f"\n  Loading model {version}...")
    model = load_model(version).to(device)
    model.eval()

    target_layer = get_gradcam_target_layer(model, version)
    gradcam = GradCAM(model, target_layer)

    # Transform for model input
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    viz_dir = os.path.join(RESULTS_PATH, version, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)

    for emotion_idx, emotion in enumerate(EMOTION_LABELS):
        img_path = get_sample_image_for_emotion(emotion)
        if img_path is None:
            print(f"    Skipping {emotion}: no images found")
            continue

        # Load and process image
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((64, 64))
        img_array = np.array(img_resized) / 255.0
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Generate GradCAM
        heatmap, output = gradcam.generate(img_tensor.clone(), emotion_idx)

        # Get prediction
        probs = F.softmax(output, dim=1)
        pred_idx = output.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item() * 100
        pred_emotion = EMOTION_DISPLAY[pred_idx]

        # Create overlay
        heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
        overlay = 0.5 * img_array + 0.5 * heatmap_colored
        overlay = np.clip(overlay, 0, 1)

        # Create figure with original and overlay
        fig, axes = plt.subplots(1, 2, figsize=(8, 4))

        axes[0].imshow(img_resized)
        axes[0].set_title(f'Original\nTrue: {EMOTION_DISPLAY[emotion_idx]}', fontsize=11)
        axes[0].axis('off')

        # Color based on correct/incorrect
        color = 'green' if pred_idx == emotion_idx else 'red'
        axes[1].imshow(overlay)
        axes[1].set_title(f'GradCAM\nPred: {pred_emotion} ({confidence:.0f}%)',
                         fontsize=11, color=color)
        axes[1].axis('off')

        plt.suptitle(f'{VERSION_NAMES[version]} - {EMOTION_DISPLAY[emotion_idx]}',
                    fontsize=12, fontweight='bold')
        plt.tight_layout()

        # Save
        save_path = os.path.join(viz_dir, f'grad_{emotion}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Created: grad_{emotion}.png")

    gradcam.remove_hooks()
    del model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None


def create_gradcam_comparison_grid():
    """Create a comparison grid using existing grad_{emotion}.png files from each version."""
    print("\nCreating comparison grid...")

    # Create figure: rows = emotions, cols = versions
    n_emotions = len(EMOTION_LABELS)
    n_versions = len(VERSIONS)

    fig, axes = plt.subplots(n_emotions, n_versions + 1, figsize=(4 * (n_versions + 1), 3.5 * n_emotions))

    for row, emotion in enumerate(EMOTION_LABELS):
        # First column: original image
        img_path = get_sample_image_for_emotion(emotion)
        if img_path and os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB').resize((64, 64))
            axes[row, 0].imshow(img)
        axes[row, 0].set_title('Original' if row == 0 else '', fontsize=11, fontweight='bold')
        axes[row, 0].set_ylabel(EMOTION_DISPLAY[row], fontsize=11, fontweight='bold')
        axes[row, 0].axis('off')

        # Remaining columns: GradCAM from each version
        for col, v in enumerate(VERSIONS):
            ax = axes[row, col + 1]

            grad_path = os.path.join(RESULTS_PATH, v, 'visualizations', f'grad_{emotion}.png')

            if os.path.exists(grad_path):
                # Load the saved gradcam image and extract just the overlay part
                grad_img = Image.open(grad_path)
                # The image has 2 subplots, we want the right one (GradCAM overlay)
                # Crop to get just the right half
                w, h = grad_img.size
                grad_overlay = grad_img.crop((w//2, 0, w, h))
                ax.imshow(grad_overlay)
            else:
                ax.text(0.5, 0.5, 'Not\nGenerated', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10)

            if row == 0:
                ax.set_title(VERSION_NAMES[v], fontsize=12, fontweight='bold', color=COLORS[v])
            ax.axis('off')

    plt.suptitle('GradCAM Attention Comparison Across Model Versions',
                fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'gradcam_comparison_grid.png')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Created: {save_path}")


def create_gradcam_comparison_live():
    """Create GradCAM comparison by generating fresh overlays (not using saved files)."""
    from models import load_model

    print("\nCreating live GradCAM comparison...")

    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Transform
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Load all models
    models_dict = {}
    gradcams = {}
    for v in VERSIONS:
        try:
            model = load_model(v).to(device)
            model.eval()
            models_dict[v] = model
            target_layer = get_gradcam_target_layer(model, v)
            gradcams[v] = GradCAM(model, target_layer)
            print(f"  Loaded {v}")
        except Exception as e:
            print(f"  Error loading {v}: {e}")

    # Create figure
    n_emotions = len(EMOTION_LABELS)
    n_versions = len(VERSIONS)

    fig, axes = plt.subplots(n_emotions, n_versions + 1, figsize=(4 * (n_versions + 1), 3.5 * n_emotions))

    for row, emotion in enumerate(EMOTION_LABELS):
        emotion_idx = row
        img_path = get_sample_image_for_emotion(emotion)

        if not img_path or not os.path.exists(img_path):
            for col in range(n_versions + 1):
                axes[row, col].text(0.5, 0.5, 'No image', ha='center', va='center')
                axes[row, col].axis('off')
            continue

        # Load image
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((64, 64))
        img_array = np.array(img_resized) / 255.0
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Original image
        axes[row, 0].imshow(img_resized)
        axes[row, 0].set_title('Original' if row == 0 else '', fontsize=11, fontweight='bold')
        axes[row, 0].set_ylabel(EMOTION_DISPLAY[row], fontsize=12, fontweight='bold')
        axes[row, 0].axis('off')

        # GradCAM for each model
        for col, v in enumerate(VERSIONS):
            ax = axes[row, col + 1]

            if v not in models_dict:
                ax.text(0.5, 0.5, 'Model\nNot Loaded', ha='center', va='center')
                ax.axis('off')
                continue

            try:
                # Generate GradCAM
                heatmap, output = gradcams[v].generate(img_tensor.clone(), emotion_idx)

                # Get prediction
                probs = F.softmax(output, dim=1)
                pred_idx = output.argmax(dim=1).item()
                confidence = probs[0, pred_idx].item() * 100

                # Create overlay
                heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
                overlay = 0.5 * img_array + 0.5 * heatmap_colored
                overlay = np.clip(overlay, 0, 1)

                ax.imshow(overlay)

                # Title with prediction
                color = 'green' if pred_idx == emotion_idx else 'red'
                pred_text = f'{EMOTION_DISPLAY[pred_idx][:3]}. ({confidence:.0f}%)'

                if row == 0:
                    ax.set_title(f'{VERSION_NAMES[v]}\n{pred_text}', fontsize=10,
                               fontweight='bold', color=color)
                else:
                    ax.set_title(pred_text, fontsize=9, color=color, fontweight='bold')

            except Exception as e:
                ax.text(0.5, 0.5, f'Error', ha='center', va='center')
                print(f"  Error for {v}/{emotion}: {e}")

            ax.axis('off')

    plt.suptitle('GradCAM Attention Comparison: Where Does Each Model Look?',
                fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'gradcam_comparison_grid.png')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\nCreated: {save_path}")

    # Cleanup
    for v in gradcams:
        gradcams[v].remove_hooks()


def main():
    parser = argparse.ArgumentParser(description='Generate GradCAM visualizations')
    parser.add_argument('--generate-only', action='store_true',
                       help='Only generate individual GradCAM images')
    parser.add_argument('--compare-only', action='store_true',
                       help='Only create comparison grid (uses existing files)')
    parser.add_argument('--live', action='store_true',
                       help='Generate comparison directly without saving individual files')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 50)

    if args.live:
        create_gradcam_comparison_live()
    elif args.compare_only:
        create_gradcam_comparison_grid()
    elif args.generate_only:
        print("Generating individual GradCAM images...")
        for v in VERSIONS:
            print(f"\nProcessing {VERSION_NAMES[v]}...")
            try:
                generate_gradcam_for_version(v, device)
            except Exception as e:
                print(f"  Error: {e}")
    else:
        # Default: do both
        print("Generating individual GradCAM images...")
        for v in VERSIONS:
            print(f"\nProcessing {VERSION_NAMES[v]}...")
            try:
                generate_gradcam_for_version(v, device)
            except Exception as e:
                print(f"  Error: {e}")

        print("\n" + "=" * 50)
        create_gradcam_comparison_grid()

    print("\n" + "=" * 50)
    print("Done!")


if __name__ == "__main__":
    main()
