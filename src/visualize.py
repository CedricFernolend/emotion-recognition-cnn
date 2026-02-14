"""
Generate all visualizations for trained emotion recognition models.

Per-version outputs (results/{version}/visualizations/):
  - gradcam_{emotion}.png           GradCAM overlay for each emotion
  - spatial_attention_{emotion}.png  Spatial attention overlay (v3 only)
  - training_curves.png             Train/val loss + accuracy
  - confusion_matrix.png            Normalized confusion matrix
  - f1_scores.png                   Per-class F1 bar chart

Cross-version outputs (results/comparison/):
  - accuracy_comparison.png          Test accuracy bar chart
  - confusion_matrix_comparison.png  Side-by-side normalized confusion matrices
  - gradcam_comparison.png           GradCAM grid (rows=emotions, cols=versions)

Usage:
    python visualize.py                      # All versions
    python visualize.py --version v3         # Single version
    python visualize.py --version v1 v3      # Specific versions
"""
import os
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from tqdm import tqdm

from configs.base_config import EMOTION_LABELS, IMAGE_SIZE, DATA_PATH, RESULTS_PATH, get_version_paths
from models import load_model, get_gradcam_target_layer
from data import get_dataloaders

# --- Style ---
plt.style.use('seaborn-v0_8-whitegrid')
VERSION_NAMES = {'v1': 'V1: Baseline', 'v2': 'V2: +SE Attention', 'v3': 'V3: +Spatial Att.'}
VERSION_SHORT = {'v1': 'V1', 'v2': 'V2', 'v3': 'V3'}
COLORS = {'v1': '#3498db', 'v2': '#2ecc71', 'v3': '#e74c3c'}
EMOTION_DISPLAY = [e.capitalize() for e in EMOTION_LABELS]
SAMPLE_DIR = os.path.join(DATA_PATH, 'test')
COMPARISON_DIR = os.path.join(RESULTS_PATH, 'comparison')

TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])


# --- GradCAM ---

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        self.handles = [
            target_layer.register_forward_hook(self._save_activation),
            target_layer.register_full_backward_hook(self._save_gradient),
        ]

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
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(cam, size=(IMAGE_SIZE, IMAGE_SIZE), mode='bilinear', align_corners=False)
        return cam.squeeze().cpu().numpy(), output

    def remove_hooks(self):
        for h in self.handles:
            h.remove()


# --- Spatial Attention (V3 only) ---

def get_spatial_attention_maps(model, input_tensor, device):
    model.eval()
    attention_maps = []
    with torch.no_grad():
        x = input_tensor.to(device)
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(model, layer_name)
            identity = layer.skip(x)
            out = layer.body(x)
            out = layer.se(out)
            out = out + identity
            out = F.relu(out)
            avg_out = torch.mean(out, dim=1, keepdim=True)
            max_out, _ = torch.max(out, dim=1, keepdim=True)
            att_input = torch.cat([avg_out, max_out], dim=1)
            att_map = layer.spatial.sigmoid(layer.spatial.conv(att_input))
            attention_maps.append(att_map.squeeze().cpu().numpy())
            x = layer.spatial(out)
            x = model.pool(x)
    return attention_maps


# --- Helpers ---

def get_sample_image(emotion):
    path = os.path.join(SAMPLE_DIR, emotion, f'grad_{emotion}.jpg')
    if os.path.exists(path):
        return path
    emotion_dir = os.path.join(SAMPLE_DIR, emotion)
    if os.path.exists(emotion_dir):
        imgs = [f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.png'))]
        if imgs:
            return os.path.join(emotion_dir, imgs[0])
    return None


def load_history(version):
    paths = get_version_paths(version)
    history_path = paths['history_path']
    if not os.path.exists(history_path):
        return None
    with open(history_path) as f:
        return json.load(f)


def make_overlay(img_array, heatmap, alpha=0.5):
    heatmap_colored = plt.cm.jet(heatmap)[:, :, :3]
    return np.clip((1 - alpha) * img_array + alpha * heatmap_colored, 0, 1)


def collect_predictions(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='  Evaluating', leave=False):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    return np.array(all_preds), np.array(all_labels)


# === Per-version visualizations ===

def generate_gradcam(version, model, device, viz_dir):
    print(f'  GradCAM overlays...')
    target_layer = get_gradcam_target_layer(model, version)
    gradcam = GradCAM(model, target_layer)

    for eidx, emotion in enumerate(EMOTION_LABELS):
        img_path = get_sample_image(emotion)
        if not img_path:
            continue
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(img_resized) / 255.0
        img_tensor = TRANSFORM(img).unsqueeze(0).to(device)

        heatmap, output = gradcam.generate(img_tensor.clone(), eidx)
        probs = F.softmax(output, dim=1)
        pred_idx = output.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item() * 100
        overlay = make_overlay(img_array, heatmap)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img_resized)
        axes[0].set_title(f'Original\nTrue: {EMOTION_DISPLAY[eidx]}', fontsize=11)
        axes[0].axis('off')

        color = 'green' if pred_idx == eidx else 'red'
        axes[1].imshow(overlay)
        axes[1].set_title(f'GradCAM\nPred: {EMOTION_DISPLAY[pred_idx]} ({confidence:.0f}%)',
                          fontsize=11, color=color)
        axes[1].axis('off')
        plt.suptitle(f'{VERSION_SHORT[version]} - {EMOTION_DISPLAY[eidx]}',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'gradcam_{emotion}.png'), dpi=150, bbox_inches='tight')
        plt.close()

    gradcam.remove_hooks()


def generate_spatial_attention(model, device, viz_dir):
    print(f'  Spatial attention maps...')
    for eidx, emotion in enumerate(EMOTION_LABELS):
        img_path = get_sample_image(emotion)
        if not img_path:
            continue
        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(img_resized) / 255.0
        img_tensor = TRANSFORM(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = F.softmax(output, dim=1)
            pred_idx = output.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item() * 100

        attention_maps = get_spatial_attention_maps(model, img_tensor, device)
        final_att = attention_maps[-1]
        att_resized = np.array(
            Image.fromarray((final_att * 255).astype(np.uint8)).resize((IMAGE_SIZE, IMAGE_SIZE))
        ) / 255.0
        overlay = make_overlay(img_array, att_resized, alpha=0.6)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(img_resized)
        axes[0].set_title(f'Original\nTrue: {EMOTION_DISPLAY[eidx]}', fontsize=11)
        axes[0].axis('off')

        color = 'green' if pred_idx == eidx else 'red'
        axes[1].imshow(overlay)
        axes[1].set_title(f'Spatial Attention\nPred: {EMOTION_DISPLAY[pred_idx]} ({confidence:.0f}%)',
                          fontsize=11, color=color)
        axes[1].axis('off')
        plt.suptitle(f'V3 - {EMOTION_DISPLAY[eidx]}', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, f'spatial_attention_{emotion}.png'), dpi=150, bbox_inches='tight')
        plt.close()


def generate_training_curves(version, viz_dir):
    print(f'  Training curves...')
    history = load_history(version)
    if history is None:
        print(f'    No history.json found, skipping')
        return
    epochs = range(1, len(history['train_acc']) + 1)
    color = COLORS[version]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, history['train_acc'], color=color, linestyle='--', alpha=0.5, label='Train')
    axes[0].plot(epochs, history['val_acc'], color=color, linewidth=2, label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Accuracy', fontsize=12, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history['train_loss'], color=color, linestyle='--', alpha=0.5, label='Train')
    axes[1].plot(epochs, history['val_loss'], color=color, linewidth=2, label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Loss', fontsize=12, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.suptitle(f'{VERSION_NAMES[version]} Training Curves', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
    plt.close()


def generate_confusion_matrix(version, model, test_loader, device, viz_dir):
    print(f'  Confusion matrix...')
    preds, labels = collect_predictions(model, test_loader, device)
    accuracy = 100.0 * np.mean(preds == labels)
    cm = confusion_matrix(labels, preds, normalize='true') * 100

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues',
                xticklabels=EMOTION_DISPLAY, yticklabels=EMOTION_DISPLAY, ax=ax)
    ax.set_title(f'{VERSION_NAMES[version]}\nTest Accuracy: {accuracy:.1f}%',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('True Label')
    ax.set_xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    return preds, labels


def generate_f1_scores(version, preds, labels, viz_dir):
    print(f'  F1 scores...')
    scores = f1_score(labels, preds, average=None)
    color = COLORS[version]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(EMOTION_LABELS))
    bars = ax.bar(x, scores * 100, color=color, alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(EMOTION_DISPLAY, fontsize=10)
    ax.set_ylabel('F1 Score (%)', fontsize=11)
    ax.set_title(f'{VERSION_NAMES[version]} — Per-Class F1 Score', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'f1_scores.png'), dpi=150, bbox_inches='tight')
    plt.close()


# === Cross-version comparisons ===

def generate_accuracy_comparison(versions):
    print('Generating accuracy comparison...')
    names, accs = [], []
    for v in versions:
        h = load_history(v)
        if h and 'test_acc' in h:
            names.append(VERSION_NAMES[v])
            accs.append(h['test_acc'])

    if not names:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    colors_list = [COLORS[v] for v in versions if load_history(v)]
    bars = ax.bar(names, accs, color=colors_list, alpha=0.8)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f'{h:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords='offset points', ha='center', fontsize=10, fontweight='bold')
    plt.tight_layout()
    os.makedirs(COMPARISON_DIR, exist_ok=True)
    plt.savefig(os.path.join(COMPARISON_DIR, 'accuracy_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def generate_confusion_matrix_comparison(versions, device):
    print('Generating confusion matrix comparison...')
    _, _, test_loader = get_dataloaders()
    n = len(versions)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for idx, v in enumerate(versions):
        ax = axes[idx]
        try:
            model = load_model(v).to(device)
            preds, labels = collect_predictions(model, test_loader, device)
            accuracy = 100.0 * np.mean(preds == labels)
            cm = confusion_matrix(labels, preds, normalize='true') * 100
            sns.heatmap(cm, annot=True, fmt='.0f', cmap='Blues',
                        xticklabels=EMOTION_DISPLAY, yticklabels=EMOTION_DISPLAY,
                        ax=ax, cbar=False, annot_kws={'size': 9})
            ax.set_title(f'{VERSION_NAMES[v]}\nAcc: {accuracy:.1f}%', fontsize=11, fontweight='bold')
            ax.set_ylabel('True Label' if idx == 0 else '', fontsize=10)
            ax.set_xlabel('Predicted Label', fontsize=10)
            ax.tick_params(axis='both', labelsize=8)
            del model
        except Exception as e:
            ax.text(0.5, 0.5, f'Error: {e}', ha='center', va='center', transform=ax.transAxes)

    plt.suptitle('Confusion Matrices (Normalized %)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    os.makedirs(COMPARISON_DIR, exist_ok=True)
    plt.savefig(os.path.join(COMPARISON_DIR, 'confusion_matrix_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()


def generate_gradcam_comparison(versions, device):
    print('Generating GradCAM comparison grid...')
    models_dict = {}
    gradcams = {}
    for v in versions:
        try:
            m = load_model(v).to(device)
            m.eval()
            models_dict[v] = m
            gradcams[v] = GradCAM(m, get_gradcam_target_layer(m, v))
        except Exception as e:
            print(f'  Could not load {v}: {e}')

    if not models_dict:
        return

    n_emotions = len(EMOTION_LABELS)
    n_versions = len(versions)
    fig, axes = plt.subplots(n_emotions, n_versions + 1, figsize=(4 * (n_versions + 1), 3.5 * n_emotions))
    if n_emotions == 1:
        axes = axes.reshape(1, -1)

    for row, emotion in enumerate(EMOTION_LABELS):
        eidx = row
        img_path = get_sample_image(emotion)

        if not img_path:
            for col in range(n_versions + 1):
                axes[row, col].text(0.5, 0.5, 'No image', ha='center', va='center')
                axes[row, col].axis('off')
            continue

        img = Image.open(img_path).convert('RGB')
        img_resized = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img_array = np.array(img_resized) / 255.0
        img_tensor = TRANSFORM(img).unsqueeze(0).to(device)

        # Original
        axes[row, 0].imshow(img_resized)
        axes[row, 0].set_title('Original' if row == 0 else '', fontsize=11, fontweight='bold')
        axes[row, 0].set_ylabel(EMOTION_DISPLAY[row], fontsize=11, fontweight='bold')
        axes[row, 0].axis('off')

        # GradCAM per version
        for col, v in enumerate(versions):
            ax = axes[row, col + 1]
            if v not in models_dict:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center', transform=ax.transAxes)
                ax.axis('off')
                continue
            try:
                heatmap, output = gradcams[v].generate(img_tensor.clone(), eidx)
                probs = F.softmax(output, dim=1)
                pred_idx = output.argmax(dim=1).item()
                confidence = probs[0, pred_idx].item() * 100
                overlay = make_overlay(img_array, heatmap)
                ax.imshow(overlay)
                color = 'green' if pred_idx == eidx else 'red'
                pred_text = f'{EMOTION_DISPLAY[pred_idx][:3]}. ({confidence:.0f}%)'
                title = f'{VERSION_SHORT[v]}\n{pred_text}' if row == 0 else pred_text
                ax.set_title(title, fontsize=10, fontweight='bold', color=color)
            except Exception as e:
                ax.text(0.5, 0.5, 'Error', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')

    plt.suptitle('GradCAM Comparison Across Model Versions',
                 fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout()
    os.makedirs(COMPARISON_DIR, exist_ok=True)
    plt.savefig(os.path.join(COMPARISON_DIR, 'gradcam_comparison.png'), dpi=200, bbox_inches='tight')
    plt.close()

    for gc in gradcams.values():
        gc.remove_hooks()


# === Main ===

def generate_version(version, device, test_loader):
    print(f'\n{"="*50}')
    print(f'Generating visualizations for {VERSION_NAMES[version]}')
    print(f'{"="*50}')

    paths = get_version_paths(version)
    viz_dir = paths['viz_dir']
    os.makedirs(viz_dir, exist_ok=True)

    model = load_model(version).to(device)
    model.eval()

    generate_gradcam(version, model, device, viz_dir)
    if version == 'v3':
        generate_spatial_attention(model, device, viz_dir)
    generate_training_curves(version, viz_dir)
    preds, labels = generate_confusion_matrix(version, model, test_loader, device, viz_dir)
    generate_f1_scores(version, preds, labels, viz_dir)

    del model
    print(f'  All saved to {viz_dir}')


def main():
    parser = argparse.ArgumentParser(description='Generate visualizations for emotion recognition models')
    parser.add_argument('--version', nargs='+', default=['v1', 'v2', 'v3'],
                        choices=['v1', 'v2', 'v3'],
                        help='Model version(s) to visualize (default: all)')
    args = parser.parse_args()
    versions = args.version

    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f'Device: {device}')

    _, _, test_loader = get_dataloaders()

    # Per-version
    for v in versions:
        try:
            generate_version(v, device, test_loader)
        except Exception as e:
            print(f'Error generating {v}: {e}')

    # Cross-version comparisons
    if len(versions) > 1:
        print(f'\n{"="*50}')
        print('Generating cross-version comparisons')
        print(f'{"="*50}')
        generate_accuracy_comparison(versions)
        generate_confusion_matrix_comparison(versions, device)
        generate_gradcam_comparison(versions, device)
        print(f'  All saved to {COMPARISON_DIR}')

    print('\nDone!')


if __name__ == '__main__':
    main()
