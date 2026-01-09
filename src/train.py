# Training script

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import os

from config import LEARNING_RATE, NUM_EPOCHS, MODEL_SAVE_PATH, RESULTS_PATH
from model import create_model
from data import get_dataloaders


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
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
    """Validate the model"""
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
    """
    Main training function.
    Call this to train your model from scratch.
    
    Training strategy follows the preliminary report with improvements:
    - Adam optimizer with learning rate 3e-4 and weight decay (L2 regularization)
    - Learning rate scheduler to reduce LR on plateau
    - Batch size 32 or 64 (from config)
    - Cross-entropy loss (hard labels for FER-2013)
    - 80/10/10 train/val/test split
    - Early stopping based on validation performance
    - Regularization: dropout2d in conv blocks, dropout in FC layers, weight decay
    """
    # Setup device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader = get_dataloaders()
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    # Create model
    print("\nInitializing model...")
    model = create_model().to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Cross-entropy loss (using hard labels from FER-2013)
    criterion = nn.CrossEntropyLoss()
    
    # Adam optimizer with lr=3e-4 and weight decay for L2 regularization
    optimizer = optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=1e-4  # L2 regularization to reduce overfitting
    )
    
    # Learning rate scheduler - reduces LR when validation accuracy plateaus
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='max',      # We're maximizing accuracy
        factor=0.5,      # Reduce LR by half
        patience=5,      # Wait 5 epochs before reducing
        min_lr=1e-6      # Don't go below this LR
    )

    # Create results directory
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

    # Early stopping parameters (as mentioned in report)
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience = 10  # Number of epochs without improvement before stopping
    patience_counter = 0
    
    # Training history for analysis
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'lr': []
    }
    
    print(f"\n{'='*70}")
    print(f"Training Configuration:")
    print(f"{'='*70}")
    print(f"Epochs: {NUM_EPOCHS} (max)")
    print(f"Initial Learning Rate: {LEARNING_RATE}")
    print(f"Weight Decay: 1e-4")
    print(f"Batch Size: {train_loader.batch_size}")
    print(f"Early Stopping Patience: {patience}")
    print(f"LR Scheduler Patience: 5")
    print(f"{'='*70}\n")

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        print(f"{'='*70}")
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")

        # Training phase
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validation phase
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Update learning rate scheduler
        scheduler.step(val_acc)

        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)

        # Print results
        print(f"\nResults:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Calculate overfitting gap
        acc_gap = train_acc - val_acc
        loss_gap = val_loss - train_loss
        print(f"Overfitting Gap: {acc_gap:.2f}% (acc) | {loss_gap:.4f} (loss)")

        # Save best model based on validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            patience_counter = 0  # Reset patience counter
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"\nNew best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"\nNo improvement ({patience_counter}/{patience})")
            
            # Early stopping check
            if patience_counter >= patience:
                print(f"\n{'='*70}")
                print(f"Early stopping triggered after {epoch+1} epochs")
                print(f"Best validation accuracy: {best_val_acc:.2f}%")
                print(f"{'='*70}")
                break

    # Training complete
    print(f"\n{'='*70}")
    print(f"Training Complete!")
    print(f"{'='*70}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Total Epochs Trained: {len(history['train_loss'])}")
    
    # Load best model for final evaluation
    print(f"\nLoading best model for final test evaluation...")
    model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    
    # Final test evaluation
    print(f"\nEvaluating on test set...")
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    print(f"\n{'='*70}")
    print(f"FINAL TEST RESULTS:")
    print(f"{'='*70}")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"{'='*70}")
    
    # Check if target achieved
    if test_acc >= 75.0:
        print(f"\nTarget achieved! (>75% as specified in report)")
    else:
        print(f"\nTarget not yet reached. Current: {test_acc:.2f}%, Target: 75%")
        print(f"Consider: longer training, architectural modifications, or hyperparameter tuning")
    
    # Print final overfitting analysis
    print(f"\n{'='*70}")
    print(f"Overfitting Analysis:")
    print(f"{'='*70}")
    print(f"Best Train Acc: {max(history['train_acc']):.2f}%")
    print(f"Best Val Acc: {best_val_acc:.2f}%")
    print(f"Test Acc: {test_acc:.2f}%")
    print(f"Train-Val Gap: {max(history['train_acc']) - best_val_acc:.2f}%")
    print(f"Val-Test Gap: {best_val_acc - test_acc:.2f}%")
    
    if max(history['train_acc']) - best_val_acc > 10:
        print(f"\nHigh overfitting detected (>10% gap)")
        print(f"Consider: stronger regularization, more data augmentation")
    elif max(history['train_acc']) - best_val_acc > 5:
        print(f"\nModerate overfitting (5-10% gap)")
        print(f"Model is learning but could generalize better")
    else:
        print(f"\nGood generalization (<5% gap)")
    
    print(f"{'='*70}\n")
    
    return model, history


if __name__ == "__main__":
    model, history = train_model()
