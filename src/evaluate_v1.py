"""
V1 Model Evaluation Script
Generates: Confusion Matrix, GradCAM examples, Classification Report
"""
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import random

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.base_config import EMOTION_LABELS, IMAGE_SIZE
from data import get_dataloaders, get_transforms
from models import load_model
from configs import load_config


class GradCAM:
    """Grad-CAM for V1 model."""

    def __init__(self, model, target_layer):
        self.model = model
        self.model.eval()
        self.gradients = None
        self.activations = None
        self.target_layer = target_layer
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].clone().detach()

        def forward_hook(module, input, output):
            self.activations = output.detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

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

        # Compute weights (global average pooling of gradients)
        weights = np.mean(gradients, axis=(1, 2))

        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cam / (cam.max() + 1e-8)

        return cam, target_class, output


def evaluate_model(version='v1'):
    """Run full evaluation of the model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config and model
    config = load_config(version)
    print(f"Loading model: {version} from {config.MODEL_SAVE_PATH}")

    if not os.path.exists(config.MODEL_SAVE_PATH):
        print(f"ERROR: Model not found at {config.MODEL_SAVE_PATH}")
        print("Please train the model first.")
        return None

    model = load_model(version)
    model = model.to(device)
    model.eval()

    # Create output directory
    os.makedirs(config.VIZ_PATH, exist_ok=True)

    # Load test data
    print("Loading test data...")
    _, _, test_loader = get_dataloaders()

    # Collect predictions
    all_predictions = []
    all_labels = []
    all_probs = []

    print("Evaluating...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = 100.0 * np.mean(all_predictions == all_labels)

    print(f"\n{'='*50}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"{'='*50}")

    # Classification report
    report = classification_report(all_labels, all_predictions,
                                   target_names=EMOTION_LABELS,
                                   output_dict=True)
    print("\nPer-class results:")
    print(classification_report(all_labels, all_predictions, target_names=EMOTION_LABELS))

    # Generate confusion matrix
    print("\nGenerating confusion matrix...")
    cm = confusion_matrix(all_labels, all_predictions)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_LABELS,
                yticklabels=EMOTION_LABELS)
    plt.title(f'Confusion Matrix - {version.upper()}\nAccuracy: {accuracy:.2f}%')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    cm_path = os.path.join(config.VIZ_PATH, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {cm_path}")

    # Generate normalized confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=EMOTION_LABELS,
                yticklabels=EMOTION_LABELS)
    plt.title(f'Normalized Confusion Matrix - {version.upper()}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    cm_norm_path = os.path.join(config.VIZ_PATH, 'confusion_matrix_normalized.png')
    plt.savefig(cm_norm_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {cm_norm_path}")

    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'report': report,
        'model': model,
        'device': device,
        'config': config
    }


def generate_gradcam_examples(model, device, config, num_examples=3):
    """Generate GradCAM visualizations for sample images."""
    print("\nGenerating GradCAM examples...")

    # Get target layer for V1 (last conv in block3)
    # block3.body is Sequential: [Conv, BN, ReLU, Conv, BN, ReLU, Conv, BN]
    # We want the last Conv2d which is at index -3 (before last BN)
    target_layer = model.block3.body[-3]

    gradcam = GradCAM(model, target_layer)

    # Load test data
    _, _, test_loader = get_dataloaders()

    # Get some sample images from each class
    transform = get_transforms(augment=False)

    # Collect sample images per class
    samples_per_class = {i: [] for i in range(len(EMOTION_LABELS))}

    for images, labels in test_loader:
        for img, label in zip(images, labels):
            label = label.item()
            if len(samples_per_class[label]) < num_examples:
                samples_per_class[label].append(img)

        # Check if we have enough samples
        if all(len(v) >= num_examples for v in samples_per_class.values()):
            break

    # Create GradCAM visualization grid
    fig, axes = plt.subplots(len(EMOTION_LABELS), num_examples * 2,
                             figsize=(num_examples * 4, len(EMOTION_LABELS) * 2))

    for class_idx, emotion in enumerate(EMOTION_LABELS):
        samples = samples_per_class[class_idx][:num_examples]

        for sample_idx, img_tensor in enumerate(samples):
            # Original image
            img_display = img_tensor.permute(1, 2, 0).numpy()
            img_display = (img_display * 0.5 + 0.5).clip(0, 1)  # Denormalize

            # Generate GradCAM
            img_input = img_tensor.unsqueeze(0).to(device)
            cam, pred_class, output = gradcam.generate(img_input)
            probs = torch.softmax(output, dim=1)[0]

            # Resize CAM to image size
            cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)) / 255.0

            # Plot original
            ax_orig = axes[class_idx, sample_idx * 2]
            ax_orig.imshow(img_display)
            ax_orig.axis('off')
            if sample_idx == 0:
                ax_orig.set_ylabel(emotion, fontsize=10)

            # Plot GradCAM overlay
            ax_cam = axes[class_idx, sample_idx * 2 + 1]
            ax_cam.imshow(img_display)
            ax_cam.imshow(cam_resized, cmap='jet', alpha=0.5)
            ax_cam.axis('off')
            pred_emotion = EMOTION_LABELS[pred_class]
            conf = probs[pred_class].item() * 100
            ax_cam.set_title(f'{pred_emotion}\n{conf:.1f}%', fontsize=8)

    plt.suptitle('GradCAM Visualizations - V1 Model\n(Original | Attention Map)', fontsize=12)
    plt.tight_layout()

    gradcam_path = os.path.join(config.VIZ_PATH, 'gradcam_examples.png')
    plt.savefig(gradcam_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {gradcam_path}")

    # Also save individual examples for specific emotions
    for emotion_idx, emotion in enumerate(EMOTION_LABELS):
        if samples_per_class[emotion_idx]:
            img_tensor = samples_per_class[emotion_idx][0]
            img_display = img_tensor.permute(1, 2, 0).numpy()
            img_display = (img_display * 0.5 + 0.5).clip(0, 1)

            img_input = img_tensor.unsqueeze(0).to(device)
            cam, pred_class, output = gradcam.generate(img_input)
            probs = torch.softmax(output, dim=1)[0]

            cam_resized = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)) / 255.0

            fig, axes = plt.subplots(1, 3, figsize=(9, 3))

            axes[0].imshow(img_display)
            axes[0].set_title(f'Original\nTrue: {emotion}')
            axes[0].axis('off')

            axes[1].imshow(cam_resized, cmap='jet')
            axes[1].set_title('Attention Map')
            axes[1].axis('off')

            axes[2].imshow(img_display)
            axes[2].imshow(cam_resized, cmap='jet', alpha=0.5)
            pred_emotion = EMOTION_LABELS[pred_class]
            conf = probs[pred_class].item() * 100
            axes[2].set_title(f'Overlay\nPred: {pred_emotion} ({conf:.1f}%)')
            axes[2].axis('off')

            plt.tight_layout()
            individual_path = os.path.join(config.VIZ_PATH, f'gradcam_{emotion}.png')
            plt.savefig(individual_path, dpi=150, bbox_inches='tight')
            plt.close()

    print(f"Saved individual GradCAM images for each emotion")


def main():
    """Main evaluation function."""
    print("="*60)
    print("V1 MODEL EVALUATION")
    print("="*60)

    results = evaluate_model('v1')

    if results is not None:
        generate_gradcam_examples(
            results['model'],
            results['device'],
            results['config'],
            num_examples=3
        )

        print("\n" + "="*60)
        print("EVALUATION COMPLETE")
        print("="*60)
        print(f"All visualizations saved to: {results['config'].VIZ_PATH}")


if __name__ == "__main__":
    main()
