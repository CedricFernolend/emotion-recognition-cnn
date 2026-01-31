"""
Generate presentation-ready visualizations comparing V1, V2, and V3 (internally V4) models.
Supports models trained with different numbers of epochs.

Usage:
    python generate_presentation_visuals.py
"""
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score

# Setup
plt.style.use('seaborn-v0_8-whitegrid')

# Internal versions (actual folder names)
VERSIONS = ['v1', 'v2', 'v4']

# Display names for presentation (v4 shown as V3)
VERSION_NAMES = {
    'v1': 'V1: Baseline',
    'v2': 'V2: +SE Attention',
    'v4': 'V3: +Spatial Att.'  # v4 displayed as V3
}
VERSION_SHORT = {'v1': 'V1', 'v2': 'V2', 'v4': 'V3'}

COLORS = {'v1': '#3498db', 'v2': '#2ecc71', 'v4': '#e74c3c'}
EMOTION_LABELS = ['Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear']
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

        # Register hooks
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

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
        return cam.squeeze().cpu().numpy()


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


def load_histories():
    """Load training histories for all versions."""
    histories = {}
    for v in VERSIONS:
        path = os.path.join(RESULTS_PATH, v, 'history.json')
        if os.path.exists(path):
            with open(path) as f:
                histories[v] = json.load(f)
    return histories


def create_accuracy_comparison_bar(histories):
    """Create bar chart comparing test accuracy across versions."""
    fig, ax = plt.subplots(figsize=(10, 6))

    versions = []
    test_accs = []
    val_accs = []
    epochs_trained = []

    for v in VERSIONS:
        if v in histories:
            versions.append(VERSION_NAMES[v])
            test_accs.append(histories[v].get('test_acc', 0))
            val_accs.append(histories[v].get('best_val_acc', 0))
            epochs_trained.append(len(histories[v].get('train_acc', [])))

    x = np.arange(len(versions))
    width = 0.35

    bars1 = ax.bar(x - width/2, val_accs, width, label='Best Validation', color='#3498db', alpha=0.8)
    bars2 = ax.bar(x + width/2, test_accs, width, label='Test', color='#2ecc71', alpha=0.8)

    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Model Performance Progression', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    # Show epochs trained in x-axis labels
    xlabels = [f'{v}\n({e} epochs)' for v, e in zip(versions, epochs_trained)]
    ax.set_xticklabels(xlabels, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1 + bars2:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add improvement arrows
    if len(test_accs) >= 2:
        for i in range(1, len(test_accs)):
            improvement = test_accs[i] - test_accs[i-1]
            ax.annotate(f'+{improvement:.1f}%',
                       xy=(i - 0.5, max(test_accs[i-1], test_accs[i]) + 5),
                       fontsize=9, color='#27ae60', fontweight='bold', ha='center')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '1_accuracy_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Created: 1_accuracy_comparison.png")


def create_training_curves_comparison(histories):
    """Create overlaid training curves for all models (handles different epoch counts)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for v in VERSIONS:
        if v not in histories:
            continue
        h = histories[v]
        epochs = range(1, len(h['train_acc']) + 1)
        color = COLORS[v]
        label = VERSION_NAMES[v]
        n_epochs = len(h['train_acc'])

        # Accuracy
        axes[0].plot(epochs, h['val_acc'], color=color, linewidth=2,
                    label=f'{label} ({n_epochs} ep)')
        axes[0].plot(epochs, h['train_acc'], color=color, linewidth=1, linestyle='--', alpha=0.5)

        # Loss
        axes[1].plot(epochs, h['val_loss'], color=color, linewidth=2,
                    label=f'{label} ({n_epochs} ep)')
        axes[1].plot(epochs, h['train_loss'], color=color, linewidth=1, linestyle='--', alpha=0.5)

    axes[0].set_xlabel('Epoch', fontsize=11)
    axes[0].set_ylabel('Accuracy (%)', fontsize=11)
    axes[0].set_title('Validation Accuracy (solid) / Train (dashed)', fontsize=12, fontweight='bold')
    axes[0].legend(loc='lower right', fontsize=10)
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Epoch', fontsize=11)
    axes[1].set_ylabel('Loss', fontsize=11)
    axes[1].set_title('Validation Loss (solid) / Train (dashed)', fontsize=12, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '2_training_curves_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Created: 2_training_curves_comparison.png")


def create_model_architecture_comparison(histories):
    """Create visual comparison of model architectures."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('off')

    # Get actual metrics from histories
    v1_acc = f"{histories.get('v1', {}).get('test_acc', 0):.1f}%"
    v2_acc = f"{histories.get('v2', {}).get('test_acc', 0):.1f}%"
    v4_acc = f"{histories.get('v4', {}).get('test_acc', 0):.1f}%"

    v1_epochs = len(histories.get('v1', {}).get('train_acc', []))
    v2_epochs = len(histories.get('v2', {}).get('train_acc', []))
    v4_epochs = len(histories.get('v4', {}).get('train_acc', []))

    headers = ['Feature', 'V1', 'V2', 'V3']
    rows = [
        ['Conv Blocks', '3', '4', '4'],
        ['Filters', '64-128-256', '64-128-256-512', '64-128-256-512'],
        ['SE Attention', 'No', 'Yes', 'Yes'],
        ['Spatial Attention', 'No', 'No', 'Yes'],
        ['Learning Rate', '0.0003', '0.0001', '0.0003'],
        ['Label Smoothing', 'No', 'No', '0.1'],
        ['LR Scheduler', 'No', 'No', 'Yes'],
        ['Parameters', '~1.8M', '~5.5M', '~5.5M'],
        ['Epochs Trained', str(v1_epochs), str(v2_epochs), str(v4_epochs)],
        ['Test Accuracy', v1_acc, v2_acc, v4_acc],
    ]

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colColours=['#ecf0f1', '#3498db', '#2ecc71', '#e74c3c'],
        colWidths=[0.3, 0.2, 0.2, 0.2]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for i in range(4):
        table[(0, i)].set_text_props(fontweight='bold', color='white')

    # Highlight accuracy row
    for i in range(4):
        table[(10, i)].set_facecolor('#f9e79f')

    ax.set_title('Model Architecture Comparison', fontsize=14, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '3_architecture_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Created: 3_architecture_comparison.png")


def create_confusion_matrices_grid():
    """Create side-by-side confusion matrices for all models."""
    from data import get_dataloaders
    from models import load_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    _, _, test_loader = get_dataloaders()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, v in enumerate(VERSIONS):
        ax = axes[idx]
        try:
            model = load_model(v).to(device)
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

            cm = confusion_matrix(all_labels, all_preds, normalize='true') * 100

            sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues',
                       xticklabels=EMOTION_LABELS,
                       yticklabels=EMOTION_LABELS,
                       ax=ax, cbar=False,
                       annot_kws={'size': 9})

            accuracy = 100.0 * np.mean(np.array(all_preds) == np.array(all_labels))
            ax.set_title(f'{VERSION_NAMES[v]}\nTest Acc: {accuracy:.1f}%', fontsize=11, fontweight='bold')
            ax.set_ylabel('True Label' if idx == 0 else '', fontsize=10)
            ax.set_xlabel('Predicted Label', fontsize=10)
            ax.tick_params(axis='both', labelsize=8)

        except Exception as e:
            ax.text(0.5, 0.5, f'Error loading {v}:\n{str(e)[:30]}',
                   ha='center', va='center', transform=ax.transAxes)

    plt.suptitle('Confusion Matrices (Normalized %)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '4_confusion_matrices.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Created: 4_confusion_matrices.png")


def create_per_class_performance():
    """Create per-class F1 score comparison."""
    from data import get_dataloaders
    from models import load_model

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    _, _, test_loader = get_dataloaders()

    results = {}

    for v in VERSIONS:
        try:
            model = load_model(v).to(device)
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

            f1_scores = f1_score(all_labels, all_preds, average=None)
            results[v] = f1_scores
        except Exception as e:
            print(f"Error loading {v}: {e}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(EMOTION_LABELS))
    width = 0.25

    for i, v in enumerate(VERSIONS):
        if v in results:
            offset = (i - 1) * width
            ax.bar(x + offset, results[v] * 100, width,
                  label=VERSION_NAMES[v], color=COLORS[v], alpha=0.8)

    ax.set_ylabel('F1 Score (%)', fontsize=11)
    ax.set_title('Per-Class F1 Score Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTION_LABELS, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 100)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '5_per_class_f1.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Created: 5_per_class_f1.png")


def create_dataset_distribution():
    """Create visualization of class distribution in dataset."""
    train_counts = {
        'Happiness': 7215,
        'Surprise': 3171,
        'Sadness': 4830,
        'Anger': 3995,
        'Disgust': 436,
        'Fear': 4097
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    colors = plt.cm.Set2(np.linspace(0, 1, 6))
    bars = axes[0].bar(train_counts.keys(), train_counts.values(), color=colors)
    axes[0].set_ylabel('Number of Samples', fontsize=11)
    axes[0].set_title('Training Set Class Distribution', fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')

    for bar in bars:
        height = bar.get_height()
        axes[0].annotate(f'{int(height):,}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    axes[1].pie(train_counts.values(), labels=train_counts.keys(), autopct='%1.1f%%',
               colors=colors, startangle=90)
    axes[1].set_title('Class Proportion', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '6_dataset_distribution.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Created: 6_dataset_distribution.png")


def create_learning_rate_comparison(histories):
    """Show learning rate schedules across models (handles different epoch counts)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for v in VERSIONS:
        if v not in histories:
            continue
        h = histories[v]
        epochs = range(1, len(h['lr']) + 1)
        n_epochs = len(h['lr'])
        ax.plot(epochs, h['lr'], color=COLORS[v], linewidth=2,
               label=f'{VERSION_NAMES[v]} ({n_epochs} ep)', marker='o', markersize=2)

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Learning Rate', fontsize=11)
    ax.set_title('Learning Rate Throughout Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '7_learning_rate.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Created: 7_learning_rate.png")


def create_overfitting_analysis(histories):
    """Show train-validation gap (handles different epoch counts)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    max_epochs = 0
    for v in VERSIONS:
        if v not in histories:
            continue
        h = histories[v]
        n_epochs = len(h['train_acc'])
        max_epochs = max(max_epochs, n_epochs)
        epochs = range(1, n_epochs + 1)
        gap = np.array(h['train_acc']) - np.array(h['val_acc'])
        ax.plot(epochs, gap, color=COLORS[v], linewidth=2,
               label=f'{VERSION_NAMES[v]} ({n_epochs} ep)')

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    if max_epochs > 0:
        ax.fill_between(range(1, max_epochs + 1), 0, 15, alpha=0.1, color='red', label='Overfitting zone')

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Train - Val Accuracy Gap (%)', fontsize=11)
    ax.set_title('Overfitting Analysis (Train-Val Gap)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '8_overfitting_analysis.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Created: 8_overfitting_analysis.png")


def create_prediction_grid_with_gradcam():
    """Create prediction grid with GradCAM overlays for sample images."""
    from data import get_dataloaders
    from models import load_model
    from torchvision import transforms

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')

    # Get sample images from each class - use the pre-selected grad_{emotion}.jpg files
    sample_images = []
    sample_labels = []
    sample_paths = []

    for emotion_idx, emotion in enumerate(EMOTION_LABELS):
        emotion_lower = emotion.lower()
        # Use the pre-selected grad_{emotion}.jpg file
        grad_image_path = os.path.join(DATA_PATH, emotion_lower, f'grad_{emotion_lower}.jpg')
        if os.path.exists(grad_image_path):
            sample_paths.append(grad_image_path)
            sample_labels.append(emotion_idx)
        else:
            # Fallback: pick a random image
            emotion_dir = os.path.join(DATA_PATH, emotion_lower)
            if os.path.exists(emotion_dir):
                images = [f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]
                if images:
                    random.seed(42 + emotion_idx)
                    img_name = random.choice(images)
                    img_path = os.path.join(emotion_dir, img_name)
                    sample_paths.append(img_path)
                    sample_labels.append(emotion_idx)

    if not sample_paths:
        print("No sample images found, skipping prediction grid")
        return

    # Load models
    models_dict = {}
    gradcams = {}
    for v in VERSIONS:
        try:
            model = load_model(v).to(device)
            model.eval()
            models_dict[v] = model
            target_layer = get_gradcam_target_layer(model, v)
            gradcams[v] = GradCAM(model, target_layer)
        except Exception as e:
            print(f"Error loading {v}: {e}")

    # Transform for model input
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Create figure: rows = samples, cols = original + 3 models with gradcam
    n_samples = min(6, len(sample_paths))
    fig, axes = plt.subplots(n_samples, 4, figsize=(14, 3 * n_samples))

    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for row, (img_path, true_label) in enumerate(zip(sample_paths[:n_samples], sample_labels[:n_samples])):
        # Load original image
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((64, 64))
        img_array = np.array(img_resized) / 255.0

        # Original image
        axes[row, 0].imshow(img_resized)
        axes[row, 0].set_title(f'True: {EMOTION_LABELS[true_label]}', fontsize=10, fontweight='bold')
        axes[row, 0].axis('off')
        if row == 0:
            axes[row, 0].set_title(f'Original\nTrue: {EMOTION_LABELS[true_label]}', fontsize=10, fontweight='bold')

        # Prepare input tensor
        img_tensor = transform(img).unsqueeze(0).to(device)

        # For each model
        for col, v in enumerate(VERSIONS):
            ax = axes[row, col + 1]

            if v not in models_dict:
                ax.text(0.5, 0.5, 'Model\nNot Found', ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue

            model = models_dict[v]

            # Get prediction
            with torch.no_grad():
                output = model(img_tensor)
                probs = F.softmax(output, dim=1)
                pred_idx = output.argmax(dim=1).item()
                confidence = probs[0, pred_idx].item() * 100

            # Generate GradCAM
            try:
                heatmap = gradcams[v].generate(img_tensor.clone(), pred_idx)

                # Create overlay
                heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
                overlay = 0.5 * img_array + 0.5 * heatmap_colored
                overlay = np.clip(overlay, 0, 1)

                ax.imshow(overlay)
            except Exception as e:
                ax.imshow(img_resized)
                print(f"GradCAM error for {v}: {e}")

            # Color based on correct/incorrect
            color = 'green' if pred_idx == true_label else 'red'
            pred_text = f'{EMOTION_LABELS[pred_idx]}\n({confidence:.0f}%)'

            if row == 0:
                ax.set_title(f'{VERSION_SHORT[v]}\n{pred_text}', fontsize=10, fontweight='bold', color=color)
            else:
                ax.set_title(pred_text, fontsize=10, fontweight='bold', color=color)
            ax.axis('off')

    plt.suptitle('Prediction Comparison with GradCAM Attention Maps', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '9_prediction_grid_gradcam.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Created: 9_prediction_grid_gradcam.png")


def create_summary_slide(histories):
    """Create a summary visualization for the final slide."""
    fig = plt.figure(figsize=(14, 8))

    fig.suptitle('Facial Expression Recognition: Model Evolution Summary',
                fontsize=16, fontweight='bold', y=0.98)

    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Get actual values
    v1_acc = histories.get('v1', {}).get('test_acc', 0)
    v2_acc = histories.get('v2', {}).get('test_acc', 0)
    v4_acc = histories.get('v4', {}).get('test_acc', 0)
    v4_val_acc = histories.get('v4', {}).get('best_val_acc', 0)
    v4_epochs = len(histories.get('v4', {}).get('train_acc', []))

    v1_to_v2 = v2_acc - v1_acc
    v2_to_v4 = v4_acc - v2_acc

    # 1. Accuracy progression (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    versions = [VERSION_NAMES[v] for v in VERSIONS if v in histories]
    test_accs = [histories[v]['test_acc'] for v in VERSIONS if v in histories]
    colors_list = [COLORS[v] for v in VERSIONS if v in histories]
    bars = ax1.bar(versions, test_accs, color=colors_list)
    ax1.set_ylabel('Test Accuracy (%)')
    ax1.set_title('Performance Progression', fontweight='bold')
    ax1.set_ylim(60, 80)
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', fontweight='bold')

    # 2. Key improvements (top middle)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    improvements = f"""
    Key Improvements:

    V1 -> V2: +{v1_to_v2:.1f}%
    - Added 4th conv block
    - SE channel attention
    - Lower learning rate

    V2 -> V3: +{v2_to_v4:.1f}%
    - Spatial attention
    - Label smoothing
    - LR scheduler
    - Full augmentation
    """
    ax2.text(0.1, 0.9, improvements, transform=ax2.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.set_title('Architecture Changes', fontweight='bold')

    # 3. Final metrics (top right)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    metrics = f"""
    Final Model (V3) Results:

    Test Accuracy:  {v4_acc:.1f}%
    Val Accuracy:   {v4_val_acc:.1f}%

    Training:
    - {v4_epochs} epochs (early stop)
    - LR scheduler active
    - Label smoothing: 0.1

    Dataset: FER2013
    - ~29K images
    - 6 emotion classes
    """
    ax3.text(0.1, 0.9, metrics, transform=ax3.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax3.set_title('Final Results', fontweight='bold')

    # 4. Training curves (bottom, spanning 2 columns)
    ax4 = fig.add_subplot(gs[1, :2])
    for v in VERSIONS:
        if v not in histories:
            continue
        h = histories[v]
        epochs = range(1, len(h['val_acc']) + 1)
        ax4.plot(epochs, h['val_acc'], color=COLORS[v], linewidth=2, label=VERSION_NAMES[v])
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Validation Accuracy (%)')
    ax4.set_title('Training Progress', fontweight='bold')
    ax4.legend(loc='lower right')
    ax4.grid(True, alpha=0.3)

    # 5. Model complexity (bottom right)
    ax5 = fig.add_subplot(gs[1, 2])
    params = [1.8, 5.5, 5.5]
    ax5.bar([VERSION_NAMES[v] for v in VERSIONS], params, color=[COLORS[v] for v in VERSIONS])
    ax5.set_ylabel('Parameters (Millions)')
    ax5.set_title('Model Size', fontweight='bold')
    ax5.tick_params(axis='x', rotation=15)

    plt.savefig(os.path.join(OUTPUT_DIR, '10_summary_slide.png'), dpi=200, bbox_inches='tight')
    plt.close()
    print("Created: 10_summary_slide.png")


def main():
    """Generate all presentation visualizations."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 50)
    print("Note: V4 model is displayed as 'V3' in all visualizations")
    print("=" * 50)

    # Load data
    histories = load_histories()
    print(f"Loaded histories for: {list(histories.keys())}")
    for v in histories:
        n_epochs = len(histories[v].get('train_acc', []))
        test_acc = histories[v].get('test_acc', 0)
        print(f"  {v}: {n_epochs} epochs, {test_acc:.1f}% test acc")
    print("=" * 50)

    print("\n[1/10] Generating accuracy comparison...")
    create_accuracy_comparison_bar(histories)

    print("[2/10] Generating training curves...")
    create_training_curves_comparison(histories)

    print("[3/10] Generating architecture comparison...")
    create_model_architecture_comparison(histories)

    print("[4/10] Generating confusion matrices...")
    try:
        create_confusion_matrices_grid()
    except Exception as e:
        print(f"  Skipped: {e}")

    print("[5/10] Generating per-class F1 scores...")
    try:
        create_per_class_performance()
    except Exception as e:
        print(f"  Skipped: {e}")

    print("[6/10] Generating dataset distribution...")
    create_dataset_distribution()

    print("[7/10] Generating learning rate comparison...")
    create_learning_rate_comparison(histories)

    print("[8/10] Generating overfitting analysis...")
    create_overfitting_analysis(histories)

    print("[9/10] Generating prediction grid with GradCAM...")
    try:
        create_prediction_grid_with_gradcam()
    except Exception as e:
        print(f"  Skipped: {e}")

    print("[10/10] Generating summary slide...")
    create_summary_slide(histories)

    print("\n" + "=" * 50)
    print(f"All visualizations saved to: {OUTPUT_DIR}")
    print("=" * 50)
    print("\nGenerated files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        if f.endswith('.png'):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
