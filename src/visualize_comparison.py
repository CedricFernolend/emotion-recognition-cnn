"""
Visualization tools for comparing multiple model versions.

Usage:
    python visualize_comparison.py                    # Compare all trained versions
    python visualize_comparison.py --versions v1 v4   # Compare specific versions
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix

from configs.base_config import EMOTION_LABELS, RESULTS_PATH
from data import get_dataloaders


def load_history(version):
    """Load training history for a version."""
    history_path = os.path.join(RESULTS_PATH, version, 'history.json')
    if not os.path.exists(history_path):
        print(f"Warning: No history found for {version}")
        return None
    with open(history_path, 'r') as f:
        return json.load(f)


def get_available_trained_versions():
    """Get list of versions that have been trained (have history.json)."""
    versions = []
    for v in ['v1', 'v2', 'v4']:
        history_path = os.path.join(RESULTS_PATH, v, 'history.json')
        if os.path.exists(history_path):
            versions.append(v)
    return versions


def plot_training_curves_comparison(versions=None, save_path=None):
    """
    Overlay training curves from all specified versions.

    Args:
        versions: List of versions to compare. If None, uses all trained versions.
        save_path: Path to save the figure. If None, uses default.
    """
    if versions is None:
        versions = get_available_trained_versions()

    if not versions:
        print("No trained versions found!")
        return

    colors = {'v1': 'blue', 'v2': 'green', 'v4': 'red'}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for version in versions:
        history = load_history(version)
        if history is None:
            continue

        epochs = range(1, len(history['train_acc']) + 1)
        color = colors.get(version, 'gray')

        # Accuracy plot
        axes[0].plot(epochs, history['train_acc'], f'{color}', linestyle='--',
                     alpha=0.5, label=f'{version} train')
        axes[0].plot(epochs, history['val_acc'], f'{color}', linestyle='-',
                     label=f'{version} val')

        # Loss plot
        axes[1].plot(epochs, history['train_loss'], f'{color}', linestyle='--',
                     alpha=0.5, label=f'{version} train')
        axes[1].plot(epochs, history['val_loss'], f'{color}', linestyle='-',
                     label=f'{version} val')

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Training and Validation Accuracy')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training and Validation Loss')
    axes[1].legend(loc='upper right')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path is None:
        os.makedirs(os.path.join(RESULTS_PATH, 'comparison'), exist_ok=True)
        save_path = os.path.join(RESULTS_PATH, 'comparison', 'training_curves_comparison.png')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves comparison saved to: {save_path}")


def plot_accuracy_progression(versions=None, save_path=None):
    """
    Bar chart showing test accuracy improvement across versions.

    Args:
        versions: List of versions to compare.
        save_path: Path to save the figure.
    """
    if versions is None:
        versions = get_available_trained_versions()

    if not versions:
        print("No trained versions found!")
        return

    version_names = []
    test_accs = []
    val_accs = []

    for version in versions:
        history = load_history(version)
        if history is None:
            continue

        version_names.append(version.upper())
        test_accs.append(history.get('test_acc', 0))
        val_accs.append(history.get('best_val_acc', max(history['val_acc'])))

    x = np.arange(len(version_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, val_accs, width, label='Best Val Accuracy', color='steelblue')
    bars2 = ax.bar(x + width/2, test_accs, width, label='Test Accuracy', color='darkgreen')

    ax.set_xlabel('Model Version')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title('Model Performance Progression')
    ax.set_xticks(x)
    ax.set_xticklabels(version_names)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()

    if save_path is None:
        os.makedirs(os.path.join(RESULTS_PATH, 'comparison'), exist_ok=True)
        save_path = os.path.join(RESULTS_PATH, 'comparison', 'accuracy_progression.png')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Accuracy progression saved to: {save_path}")


def plot_confusion_matrices_grid(versions=None, save_path=None):
    """
    Create grid of confusion matrices for all versions.

    Args:
        versions: List of versions to compare.
        save_path: Path to save the figure.
    """
    if versions is None:
        versions = get_available_trained_versions()

    if not versions:
        print("No trained versions found!")
        return

    from models import load_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, _, test_loader = get_dataloaders()

    n_versions = len(versions)
    cols = min(2, n_versions)
    rows = (n_versions + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(7*cols, 6*rows))
    if n_versions == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    for idx, version in enumerate(versions):
        row, col = idx // cols, idx % cols
        ax = axes[row, col]

        try:
            model = load_model(version).to(device)
            model.eval()

            all_preds = []
            all_labels = []

            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    outputs = model(images)
                    _, predicted = outputs.max(1)
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.numpy())

            cm = confusion_matrix(all_labels, all_preds)

            # Get accuracy for title
            accuracy = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))

            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                        xticklabels=EMOTION_LABELS,
                        yticklabels=EMOTION_LABELS,
                        ax=ax)
            ax.set_title(f'{version.upper()} (Test Acc: {accuracy:.1f}%)')
            ax.set_ylabel('True Label')
            ax.set_xlabel('Predicted Label')

        except Exception as e:
            ax.text(0.5, 0.5, f'{version}\nNot trained\n{str(e)[:30]}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{version.upper()} - Not Available')

    # Hide empty subplots
    for idx in range(n_versions, rows * cols):
        row, col = idx // cols, idx % cols
        axes[row, col].axis('off')

    plt.tight_layout()

    if save_path is None:
        os.makedirs(os.path.join(RESULTS_PATH, 'comparison'), exist_ok=True)
        save_path = os.path.join(RESULTS_PATH, 'comparison', 'confusion_matrices_grid.png')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrices grid saved to: {save_path}")


def generate_gradcam_comparison(image_path, versions=None, save_path=None):
    """
    Generate GradCAM visualizations for the same image across all model versions.

    Args:
        image_path: Path to the image to analyze.
        versions: List of versions to compare.
        save_path: Path to save the figure.
    """
    if versions is None:
        versions = get_available_trained_versions()

    if not versions:
        print("No trained versions found!")
        return

    from PIL import Image
    from torchvision import transforms
    from models import load_model, get_gradcam_target_layer
    from gradcam import GradCAM

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load and preprocess image
    img = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    img_tensor = transform(img).unsqueeze(0).to(device)

    n_versions = len(versions)
    fig, axes = plt.subplots(1, n_versions + 1, figsize=(4*(n_versions + 1), 4))

    # Original image
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')

    for idx, version in enumerate(versions):
        ax = axes[idx + 1]

        try:
            model = load_model(version).to(device)
            model.eval()

            target_layer = get_gradcam_target_layer(model, version)
            gradcam = GradCAM(model, target_layer)
            heatmap = gradcam.generate(img_tensor)

            # Overlay on image
            img_array = np.array(img.resize((64, 64))) / 255.0
            heatmap_resized = np.array(Image.fromarray((heatmap * 255).astype(np.uint8)).resize((64, 64))) / 255.0

            # Create overlay
            heatmap_colored = plt.cm.jet(heatmap_resized)[:, :, :3]
            overlay = 0.5 * img_array + 0.5 * heatmap_colored

            ax.imshow(overlay)
            ax.set_title(f'{version.upper()} GradCAM')
            ax.axis('off')

        except Exception as e:
            ax.text(0.5, 0.5, f'{version}\nError: {str(e)[:20]}',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{version.upper()} - Error')
            ax.axis('off')

    plt.tight_layout()

    if save_path is None:
        os.makedirs(os.path.join(RESULTS_PATH, 'comparison'), exist_ok=True)
        save_path = os.path.join(RESULTS_PATH, 'comparison', 'gradcam_comparison.png')

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"GradCAM comparison saved to: {save_path}")


def generate_all_comparisons(versions=None):
    """Generate all comparison visualizations."""
    print("Generating all comparison visualizations...")
    print("=" * 50)

    plot_training_curves_comparison(versions)
    plot_accuracy_progression(versions)
    plot_confusion_matrices_grid(versions)

    print("=" * 50)
    print("All comparison visualizations generated!")
    print(f"Output directory: {os.path.join(RESULTS_PATH, 'comparison')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate model comparison visualizations')
    parser.add_argument('--versions', nargs='+', default=None,
                        help='Versions to compare (e.g., v1 v2 v4). Default: all trained versions.')
    parser.add_argument('--gradcam-image', type=str, default=None,
                        help='Path to image for GradCAM comparison.')
    args = parser.parse_args()

    if args.gradcam_image:
        generate_gradcam_comparison(args.gradcam_image, args.versions)
    else:
        generate_all_comparisons(args.versions)
