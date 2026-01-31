"""
Generate attention visualizations for all models.

- V1/V2: GradCAM (works well for simpler models)
- V3 (V4): Direct Spatial Attention maps (shows what model actually learned)

Saves all images individually + creates comparison grids.

Usage:
    python generate_attention_visuals.py
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Setup
VERSIONS = ['v1', 'v2', 'v4']
VERSION_NAMES = {'v1': 'V1', 'v2': 'V2', 'v4': 'V3'}
COLORS = {'v1': '#3498db', 'v2': '#2ecc71', 'v4': '#e74c3c'}
EMOTION_LABELS = ['happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
EMOTION_DISPLAY = ['Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear']

RESULTS_PATH = '../results'
DATA_PATH = '../data/raw/test'
OUTPUT_DIR = '../results/attention_visuals'


class GradCAM:
    """Grad-CAM for V1 and V2."""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.handles = []

        self.handles.append(target_layer.register_forward_hook(self._save_activation))
        self.handles.append(target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        cam = F.interpolate(cam, size=(64, 64), mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy()

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()


class SpatialAttentionExtractor:
    """Extract spatial attention maps from V4 model."""
    def __init__(self, model):
        self.model = model
        self.attention_maps = {}
        self.handles = []

        # Hook into each layer's spatial attention
        for name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(model, name)
            self.handles.append(
                layer.spatial.register_forward_hook(
                    self._make_hook(name)
                )
            )

    def _make_hook(self, name):
        def hook(module, input, output):
            # The spatial attention applies: x * attention_map
            # We can get the attention map from input and output
            # attention_map = output / (input[0] + 1e-8) but this is unstable
            # Better: hook the conv output before sigmoid
            pass
        return hook

    def remove_hooks(self):
        for handle in self.handles:
            handle.remove()


def get_spatial_attention_maps(model, input_tensor, device):
    """Extract spatial attention maps from V4 model by running forward pass."""
    model.eval()
    attention_maps = []

    # Manual forward pass to capture attention maps
    with torch.no_grad():
        x = input_tensor.to(device)

        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(model, layer_name)

            # Get input to spatial attention
            identity = layer.skip(x)
            out = layer.body(x)
            out = layer.se(out)
            out = out + identity
            out = F.relu(out)

            # Compute spatial attention map
            avg_out = torch.mean(out, dim=1, keepdim=True)
            max_out, _ = torch.max(out, dim=1, keepdim=True)
            attention_input = torch.cat([avg_out, max_out], dim=1)
            attention_map = layer.spatial.sigmoid(layer.spatial.conv(attention_input))

            attention_maps.append(attention_map.squeeze().cpu().numpy())

            # Continue forward pass
            x = layer.spatial(out)
            x = model.pool(x)

    return attention_maps


def get_gradcam_target_layer(model, version):
    """Get target layer for GradCAM."""
    if version == 'v1':
        return model.block3.body[-2]
    elif version == 'v2':
        return model.block4.body[-2]
    else:
        return model.layer4.body[-2]


def get_sample_image_path(emotion):
    """Get the grad_{emotion}.jpg image path."""
    grad_path = os.path.join(DATA_PATH, emotion, f'grad_{emotion}.jpg')
    if os.path.exists(grad_path):
        return grad_path

    # Fallback
    emotion_dir = os.path.join(DATA_PATH, emotion)
    if os.path.exists(emotion_dir):
        images = [f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.png'))]
        if images:
            return os.path.join(emotion_dir, images[0])
    return None


def create_overlay(img_array, heatmap, alpha=0.5):
    """Create heatmap overlay on image."""
    heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
    overlay = (1 - alpha) * img_array + alpha * heatmap_colored
    return np.clip(overlay, 0, 1)


def save_individual_image(img, title, save_path, cmap=None):
    """Save a single image with title."""
    fig, ax = plt.subplots(figsize=(4, 4))
    if cmap:
        ax.imshow(img, cmap=cmap)
    else:
        ax.imshow(img)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def generate_v1_v2_gradcam(version, device):
    """Generate GradCAM visualizations for V1 or V2."""
    from models import load_model

    print(f"\n  Processing {VERSION_NAMES[version]} with GradCAM...")

    model = load_model(version).to(device)
    model.eval()

    target_layer = get_gradcam_target_layer(model, version)
    gradcam = GradCAM(model, target_layer)

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    version_dir = os.path.join(OUTPUT_DIR, version)
    os.makedirs(version_dir, exist_ok=True)

    for emotion_idx, emotion in enumerate(EMOTION_LABELS):
        img_path = get_sample_image_path(emotion)
        if not img_path:
            print(f"    Skipping {emotion}: no image found")
            continue

        # Load image
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((64, 64))
        img_array = np.array(img_resized) / 255.0
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            pred_idx = output.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item() * 100

        # Generate GradCAM
        heatmap = gradcam.generate(img_tensor.clone(), pred_idx)

        # Create overlay
        overlay = create_overlay(img_array, heatmap)

        # Save individual images
        # 1. Original
        save_path = os.path.join(version_dir, f'{emotion}_original.png')
        save_individual_image(img_resized, f'Original: {EMOTION_DISPLAY[emotion_idx]}', save_path)
        print(f"    Saved: {emotion}_original.png")

        # 2. Heatmap only
        save_path = os.path.join(version_dir, f'{emotion}_gradcam_heatmap.png')
        save_individual_image(heatmap, f'GradCAM Heatmap', save_path, cmap='jet')
        print(f"    Saved: {emotion}_gradcam_heatmap.png")

        # 3. Overlay
        color = 'green' if pred_idx == emotion_idx else 'red'
        pred_text = f'Pred: {EMOTION_DISPLAY[pred_idx]} ({confidence:.0f}%)'
        save_path = os.path.join(version_dir, f'{emotion}_gradcam_overlay.png')

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(overlay)
        ax.set_title(f'{VERSION_NAMES[version]} GradCAM\n{pred_text}',
                    fontsize=10, fontweight='bold', color=color)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {emotion}_gradcam_overlay.png")

    gradcam.remove_hooks()
    del model


def generate_v4_spatial_attention(device):
    """Generate Spatial Attention visualizations for V4."""
    from models import load_model

    print(f"\n  Processing V3 (V4) with Spatial Attention maps...")

    model = load_model('v4').to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    version_dir = os.path.join(OUTPUT_DIR, 'v4')
    os.makedirs(version_dir, exist_ok=True)

    for emotion_idx, emotion in enumerate(EMOTION_LABELS):
        img_path = get_sample_image_path(emotion)
        if not img_path:
            print(f"    Skipping {emotion}: no image found")
            continue

        # Load image
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((64, 64))
        img_array = np.array(img_resized) / 255.0
        img_tensor = transform(img).unsqueeze(0).to(device)

        # Get prediction
        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            pred_idx = output.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item() * 100

        # Get spatial attention maps from all layers
        attention_maps = get_spatial_attention_maps(model, img_tensor, device)

        # Save individual images
        # 1. Original
        save_path = os.path.join(version_dir, f'{emotion}_original.png')
        save_individual_image(img_resized, f'Original: {EMOTION_DISPLAY[emotion_idx]}', save_path)
        print(f"    Saved: {emotion}_original.png")

        # 2. Attention maps for each layer
        layer_names = ['Layer 1 (64ch)', 'Layer 2 (128ch)', 'Layer 3 (256ch)', 'Layer 4 (512ch)']
        for layer_idx, (att_map, layer_name) in enumerate(zip(attention_maps, layer_names)):
            # Resize attention map to 64x64
            att_resized = np.array(Image.fromarray((att_map * 255).astype(np.uint8)).resize((64, 64))) / 255.0

            # Save heatmap
            save_path = os.path.join(version_dir, f'{emotion}_attention_layer{layer_idx+1}_heatmap.png')
            save_individual_image(att_resized, f'Spatial Attention\n{layer_name}', save_path, cmap='hot')
            print(f"    Saved: {emotion}_attention_layer{layer_idx+1}_heatmap.png")

            # Save overlay
            overlay = create_overlay(img_array, att_resized, alpha=0.6)
            save_path = os.path.join(version_dir, f'{emotion}_attention_layer{layer_idx+1}_overlay.png')
            save_individual_image(overlay, f'Attention Overlay\n{layer_name}', save_path)
            print(f"    Saved: {emotion}_attention_layer{layer_idx+1}_overlay.png")

        # 3. Combined final layer attention with prediction
        final_att = attention_maps[-1]
        final_att_resized = np.array(Image.fromarray((final_att * 255).astype(np.uint8)).resize((64, 64))) / 255.0
        overlay = create_overlay(img_array, final_att_resized, alpha=0.6)

        color = 'green' if pred_idx == emotion_idx else 'red'
        pred_text = f'Pred: {EMOTION_DISPLAY[pred_idx]} ({confidence:.0f}%)'
        save_path = os.path.join(version_dir, f'{emotion}_final_attention_overlay.png')

        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(overlay)
        ax.set_title(f'V3 Spatial Attention\n{pred_text}',
                    fontsize=10, fontweight='bold', color=color)
        ax.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Saved: {emotion}_final_attention_overlay.png")

    del model


def create_comparison_grid():
    """Create comparison grid using saved images."""
    print("\n  Creating comparison grid...")

    n_emotions = len(EMOTION_LABELS)
    fig, axes = plt.subplots(n_emotions, 4, figsize=(14, 3 * n_emotions))

    for row, emotion in enumerate(EMOTION_LABELS):
        # Column 0: Original
        img_path = os.path.join(OUTPUT_DIR, 'v1', f'{emotion}_original.png')
        if os.path.exists(img_path):
            img = Image.open(img_path)
            axes[row, 0].imshow(img)
        axes[row, 0].set_title('Original' if row == 0 else '', fontsize=11, fontweight='bold')
        axes[row, 0].set_ylabel(EMOTION_DISPLAY[row], fontsize=11, fontweight='bold')
        axes[row, 0].axis('off')

        # Column 1: V1 GradCAM
        img_path = os.path.join(OUTPUT_DIR, 'v1', f'{emotion}_gradcam_overlay.png')
        if os.path.exists(img_path):
            img = Image.open(img_path)
            axes[row, 1].imshow(img)
        axes[row, 1].set_title('V1: GradCAM' if row == 0 else '', fontsize=11, fontweight='bold', color=COLORS['v1'])
        axes[row, 1].axis('off')

        # Column 2: V2 GradCAM
        img_path = os.path.join(OUTPUT_DIR, 'v2', f'{emotion}_gradcam_overlay.png')
        if os.path.exists(img_path):
            img = Image.open(img_path)
            axes[row, 2].imshow(img)
        axes[row, 2].set_title('V2: GradCAM' if row == 0 else '', fontsize=11, fontweight='bold', color=COLORS['v2'])
        axes[row, 2].axis('off')

        # Column 3: V3 Spatial Attention
        img_path = os.path.join(OUTPUT_DIR, 'v4', f'{emotion}_final_attention_overlay.png')
        if os.path.exists(img_path):
            img = Image.open(img_path)
            axes[row, 3].imshow(img)
        axes[row, 3].set_title('V3: Spatial Attention' if row == 0 else '', fontsize=11, fontweight='bold', color=COLORS['v4'])
        axes[row, 3].axis('off')

    plt.suptitle('Attention Visualization Comparison\nV1/V2: GradCAM | V3: Learned Spatial Attention',
                fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'attention_comparison_grid.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: attention_comparison_grid.png")


def create_v4_layer_progression():
    """Create visualization showing attention progression through V4 layers."""
    print("\n  Creating V4 layer progression visualization...")

    n_emotions = len(EMOTION_LABELS)
    fig, axes = plt.subplots(n_emotions, 5, figsize=(16, 3 * n_emotions))

    for row, emotion in enumerate(EMOTION_LABELS):
        # Column 0: Original
        img_path = os.path.join(OUTPUT_DIR, 'v4', f'{emotion}_original.png')
        if os.path.exists(img_path):
            img = Image.open(img_path)
            axes[row, 0].imshow(img)
        axes[row, 0].set_title('Original' if row == 0 else '', fontsize=10, fontweight='bold')
        axes[row, 0].set_ylabel(EMOTION_DISPLAY[row], fontsize=10, fontweight='bold')
        axes[row, 0].axis('off')

        # Columns 1-4: Layer attention maps
        for layer_idx in range(4):
            img_path = os.path.join(OUTPUT_DIR, 'v4', f'{emotion}_attention_layer{layer_idx+1}_overlay.png')
            if os.path.exists(img_path):
                img = Image.open(img_path)
                axes[row, layer_idx + 1].imshow(img)
            title = f'Layer {layer_idx+1}' if row == 0 else ''
            axes[row, layer_idx + 1].set_title(title, fontsize=10, fontweight='bold')
            axes[row, layer_idx + 1].axis('off')

    plt.suptitle('V3 Spatial Attention Progression Through Layers',
                fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'v4_attention_layer_progression.png')
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  Saved: v4_attention_layer_progression.png")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate V1 GradCAM
    print("\n[1/4] Generating V1 GradCAM visualizations...")
    try:
        generate_v1_v2_gradcam('v1', device)
    except Exception as e:
        print(f"  Error: {e}")

    # Generate V2 GradCAM
    print("\n[2/4] Generating V2 GradCAM visualizations...")
    try:
        generate_v1_v2_gradcam('v2', device)
    except Exception as e:
        print(f"  Error: {e}")

    # Generate V4 Spatial Attention
    print("\n[3/4] Generating V3 (V4) Spatial Attention visualizations...")
    try:
        generate_v4_spatial_attention(device)
    except Exception as e:
        print(f"  Error: {e}")

    # Create comparison grids
    print("\n[4/4] Creating comparison grids...")
    try:
        create_comparison_grid()
        create_v4_layer_progression()
    except Exception as e:
        print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 60)

    # List generated files
    print("\nGenerated files:")
    for version in ['v1', 'v2', 'v4']:
        version_dir = os.path.join(OUTPUT_DIR, version)
        if os.path.exists(version_dir):
            files = sorted(os.listdir(version_dir))
            print(f"\n  {VERSION_NAMES.get(version, version)}/")
            for f in files:
                print(f"    - {f}")

    # List comparison files
    comparison_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith('.png')]
    if comparison_files:
        print(f"\n  Comparison grids:")
        for f in sorted(comparison_files):
            print(f"    - {f}")


if __name__ == "__main__":
    main()
