import os
import argparse
import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from configs.base_config import EMOTION_LABELS
from data import get_dataloaders


def get_model_and_paths(version):
    """Get model and paths for specified version."""
    from configs import load_config
    from models import load_model
    config = load_config(version)
    return load_model(version), {
        'model_path': config.MODEL_SAVE_PATH,
        'viz_path': config.VIZ_PATH,
    }


def test_model(version='v3'):
    """Test the trained model and generate evaluation metrics."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading model...")
    model, paths = get_model_and_paths(version)
    print(f"Evaluating model version: {version}")

    model = model.to(device)
    model.eval()

    print("Loading test data...")
    _, _, test_loader = get_dataloaders()

    all_predictions = []
    all_labels = []

    print("Testing...")
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    accuracy = 100.0 * np.mean(all_predictions == all_labels)

    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print("\nPer-class results:")
    print(classification_report(all_labels, all_predictions, target_names=EMOTION_LABELS))

    # Generate and save confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_LABELS,
                yticklabels=EMOTION_LABELS)

    title = f'Confusion Matrix - {version}'
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    os.makedirs(paths['viz_path'], exist_ok=True)
    save_path = os.path.join(paths['viz_path'], 'confusion_matrix.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nConfusion matrix saved to {save_path}")

    return accuracy, cm


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate emotion recognition model')
    parser.add_argument('--version', type=str, default='v3',
                        choices=['v1', 'v2', 'v3'],
                        help='Model version to evaluate (v1, v2, v3)')
    args = parser.parse_args()

    test_model(version=args.version)
