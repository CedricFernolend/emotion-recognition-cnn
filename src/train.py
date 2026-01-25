import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from config import LEARNING_RATE, NUM_EPOCHS, MODEL_SAVE_PATH, RESULTS_PATH, USE_CLASS_WEIGHTS
from model import create_model
from data import get_dataloaders, compute_class_weights


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(train_loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100.0 * correct / total
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validation"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(val_loader)
    val_acc = 100.0 * correct / total
    return val_loss, val_acc


def train_model():
    """Main training function with early stopping and learning rate scheduling."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    print("\nInitializing model...")
    model = create_model().to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function - optionally weighted by class frequency
    # Label smoothing (0.1) reduces overconfident predictions and improves generalization
    if USE_CLASS_WEIGHTS:
        print("\nComputing class weights...")
        class_weights = compute_class_weights().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        print("Using weighted CrossEntropyLoss with label smoothing (0.1)")
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        print("Using CrossEntropyLoss with label smoothing (0.1)")

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)

    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    best_val_acc = 0.0
    patience = 10
    patience_counter = 0

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    print(f"\nTraining for up to {NUM_EPOCHS} epochs (early stopping patience: {patience})\n")

    for epoch in range(NUM_EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} (lr: {current_lr:.6f})")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step(val_acc)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        acc_gap = train_acc - val_acc
        print(f"Train: {train_acc:.2f}% (loss: {train_loss:.4f}) | Val: {val_acc:.2f}% (loss: {val_loss:.4f}) | Gap: {acc_gap:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"New best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"No improvement ({patience_counter}/{patience})")

            if patience_counter >= patience:
                print(f"\nEarly stopping after {epoch+1} epochs. Best val acc: {best_val_acc:.2f}%")
                break

        print()

    print("Training complete!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")

    print("\nLoading best model for final test evaluation...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    test_loss, test_acc = validate(model, test_loader, criterion, device)

    print(f"\nFinal Test Results:")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    if test_acc >= 75.0:
        print("Target achieved! (>75%)")
    else:
        print(f"Target not reached. Current: {test_acc:.2f}%, Target: 75%")

    return model, history


if __name__ == "__main__":
    model, history = train_model()
