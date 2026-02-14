import argparse
import torch
import numpy as np
from sklearn.metrics import classification_report
from tqdm import tqdm

from configs.base_config import EMOTION_LABELS
from models import load_model
from data import get_dataloaders


def test_model(version='v3'):
    """Test the trained model and print evaluation metrics."""
    device = torch.device('cuda' if torch.cuda.is_available() else
                          'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading model {version}...")
    model = load_model(version)

    model = model.to(device)
    model.eval()

    _, _, test_loader = get_dataloaders()

    all_predictions = []
    all_labels = []

    print("Evaluating...")
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

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate emotion recognition model')
    parser.add_argument('--version', type=str, default='v3',
                        choices=['v1', 'v2', 'v3'],
                        help='Model version to evaluate (v1, v2, v3)')
    args = parser.parse_args()

    test_model(version=args.version)
