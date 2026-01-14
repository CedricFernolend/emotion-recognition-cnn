# Evaluation and testing

import torch
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os

from config import EMOTION_LABELS, MODEL_SAVE_PATH, RESULTS_PATH
from model import load_model
from data import get_dataloaders


def test_model(model_path=MODEL_SAVE_PATH):
    """
    Test the trained model and show results.
    Call this after training to see how well your model performs.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = load_model(model_path).to(device)
    model.eval()

    # Load test data
    print("Loading test data...")
    _, val_loader, test_loader = get_dataloaders()

    # Collect predictions
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

    # Calculate accuracy
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    accuracy = 100.0 * np.mean(all_predictions == all_labels)

    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print("\nPer-class results:")
    print(classification_report(all_labels, all_predictions, target_names=EMOTION_LABELS))

    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=EMOTION_LABELS,
                yticklabels=EMOTION_LABELS)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # Save confusion matrix
    os.makedirs(f"{RESULTS_PATH}/visualizations", exist_ok=True)
    plt.savefig(f"{RESULTS_PATH}/visualizations/confusion_matrix.png")
    print(f"\nConfusion matrix saved to {RESULTS_PATH}/visualizations/confusion_matrix.png")

    return accuracy


if __name__ == "__main__":
    test_model()
