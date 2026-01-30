import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import matplotlib.pyplot as plt


def get_config_and_model(version):
    """Load config and create model for specified version."""
    if version is None:
        # Backward compatible: use default config and model
        from config import LEARNING_RATE, NUM_EPOCHS, MODEL_SAVE_PATH, USE_CLASS_WEIGHTS, DROPOUT_RATE
        from model import create_model
        return {
            'version': 'default',
            'LEARNING_RATE': LEARNING_RATE,
            'NUM_EPOCHS': NUM_EPOCHS,
            'MODEL_SAVE_PATH': MODEL_SAVE_PATH,
            'USE_CLASS_WEIGHTS': USE_CLASS_WEIGHTS,
            'DROPOUT_RATE': DROPOUT_RATE,
            'LABEL_SMOOTHING': 0.1,
            'WEIGHT_DECAY': 1e-4,
            'USE_LR_SCHEDULER': True,
            'LR_SCHEDULER_FACTOR': 0.5,
            'LR_SCHEDULER_PATIENCE': 5,
            'LR_MIN': 1e-6,
            'EARLY_STOPPING_PATIENCE': 10,
            'AUGMENTATION': None,  # Use default
            'HISTORY_SAVE_PATH': None,
            'VIZ_PATH': 'results/visualizations/',
        }, create_model()
    else:
        from configs import load_config
        from models import create_model
        config = load_config(version)
        dropout_rate = getattr(config, 'DROPOUT_RATE', 0.5)
        model = create_model(version, dropout_rate=dropout_rate)
        return {
            'version': config.VERSION,
            'version_name': getattr(config, 'VERSION_NAME', version),
            'LEARNING_RATE': config.LEARNING_RATE,
            'NUM_EPOCHS': config.NUM_EPOCHS,
            'MODEL_SAVE_PATH': config.MODEL_SAVE_PATH,
            'USE_CLASS_WEIGHTS': getattr(config, 'USE_CLASS_WEIGHTS', False),
            'DROPOUT_RATE': dropout_rate,
            'LABEL_SMOOTHING': getattr(config, 'LABEL_SMOOTHING', 0.0),
            'WEIGHT_DECAY': getattr(config, 'WEIGHT_DECAY', 0.0),
            'USE_LR_SCHEDULER': getattr(config, 'USE_LR_SCHEDULER', False),
            'LR_SCHEDULER_FACTOR': getattr(config, 'LR_SCHEDULER_FACTOR', 0.5),
            'LR_SCHEDULER_PATIENCE': getattr(config, 'LR_SCHEDULER_PATIENCE', 5),
            'LR_MIN': getattr(config, 'LR_MIN', 1e-6),
            'EARLY_STOPPING_PATIENCE': getattr(config, 'EARLY_STOPPING_PATIENCE', 10),
            'AUGMENTATION': getattr(config, 'AUGMENTATION', None),
            'HISTORY_SAVE_PATH': getattr(config, 'HISTORY_SAVE_PATH', None),
            'VIZ_PATH': getattr(config, 'VIZ_PATH', f'results/{version}/visualizations/'),
        }, model


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


def plot_training_history(history, save_path):
    """Plot and save training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    epochs = range(1, len(history['train_acc']) + 1)

    # Accuracy plot
    axes[0].plot(epochs, history['train_acc'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_acc'], 'r-', label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Training and Validation Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Loss plot
    axes[1].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[1].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].set_title('Training and Validation Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to: {save_path}")


def train_model(version=None):
    """Main training function with early stopping and learning rate scheduling."""
    # Load configuration and model
    cfg, model = get_config_and_model(version)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print(f"\nTraining model version: {cfg['version']}")
    if 'version_name' in cfg:
        print(f"Model name: {cfg['version_name']}")

    # Print configuration summary
    print("\n" + "="*50)
    print("Configuration Summary")
    print("="*50)
    print(f"  Learning Rate:     {cfg['LEARNING_RATE']}")
    print(f"  Weight Decay:      {cfg['WEIGHT_DECAY']}")
    print(f"  Dropout Rate:      {cfg['DROPOUT_RATE']}")
    print(f"  Epochs (max):      {cfg['NUM_EPOCHS']}")
    print(f"  Early Stop:        {cfg['EARLY_STOPPING_PATIENCE']} epochs")
    print(f"  Class Weights:     {cfg['USE_CLASS_WEIGHTS']}")
    print(f"  Label Smoothing:   {cfg['LABEL_SMOOTHING']}")
    print(f"  LR Scheduler:      {cfg['USE_LR_SCHEDULER']}")
    print(f"  Model Save Path:   {cfg['MODEL_SAVE_PATH']}")
    print("="*50)

    # Create output directories
    os.makedirs(os.path.dirname(cfg['MODEL_SAVE_PATH']), exist_ok=True)
    os.makedirs(cfg['VIZ_PATH'], exist_ok=True)

    print("\nLoading data...")
    from data import get_dataloaders, compute_class_weights
    train_loader, val_loader, test_loader = get_dataloaders(
        augment_config=cfg['AUGMENTATION']
    )
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")

    print("\nInitializing model...")
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function
    if cfg['USE_CLASS_WEIGHTS']:
        print("\nComputing class weights...")
        class_weights = compute_class_weights().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg['LABEL_SMOOTHING'])
        print(f"Using weighted CrossEntropyLoss with label smoothing ({cfg['LABEL_SMOOTHING']})")
    else:
        if cfg['LABEL_SMOOTHING'] > 0:
            criterion = nn.CrossEntropyLoss(label_smoothing=cfg['LABEL_SMOOTHING'])
            print(f"Using CrossEntropyLoss with label smoothing ({cfg['LABEL_SMOOTHING']})")
        else:
            criterion = nn.CrossEntropyLoss()
            print("Using standard CrossEntropyLoss")

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=cfg['LEARNING_RATE'], weight_decay=cfg['WEIGHT_DECAY'])
    print(f"Optimizer: Adam (lr={cfg['LEARNING_RATE']}, weight_decay={cfg['WEIGHT_DECAY']})")

    # Learning rate scheduler (optional)
    scheduler = None
    if cfg['USE_LR_SCHEDULER']:
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max',
            factor=cfg['LR_SCHEDULER_FACTOR'],
            patience=cfg['LR_SCHEDULER_PATIENCE'],
            min_lr=cfg['LR_MIN']
        )
        print(f"LR Scheduler: ReduceLROnPlateau (factor={cfg['LR_SCHEDULER_FACTOR']}, patience={cfg['LR_SCHEDULER_PATIENCE']})")
    else:
        print("LR Scheduler: None")

    best_val_acc = 0.0
    patience = cfg['EARLY_STOPPING_PATIENCE']
    patience_counter = 0

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': []}

    print(f"\nTraining for up to {cfg['NUM_EPOCHS']} epochs (early stopping patience: {patience})\n")

    for epoch in range(cfg['NUM_EPOCHS']):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{cfg['NUM_EPOCHS']} (lr: {current_lr:.6f})")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if scheduler:
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
            torch.save(model.state_dict(), cfg['MODEL_SAVE_PATH'])
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

    # Save training history
    if cfg['HISTORY_SAVE_PATH']:
        history['best_val_acc'] = best_val_acc
        history['version'] = cfg['version']
        with open(cfg['HISTORY_SAVE_PATH'], 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved to: {cfg['HISTORY_SAVE_PATH']}")

    # Plot training curves
    curves_path = os.path.join(cfg['VIZ_PATH'], 'training_curves.png')
    plot_training_history(history, curves_path)

    # Final test evaluation
    print("\nLoading best model for final test evaluation...")
    model.load_state_dict(torch.load(cfg['MODEL_SAVE_PATH']))
    test_loss, test_acc = validate(model, test_loader, criterion, device)

    print(f"\nFinal Test Results:")
    print(f"Test Accuracy: {test_acc:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Update history with test results
    history['test_acc'] = test_acc
    history['test_loss'] = test_loss
    if cfg['HISTORY_SAVE_PATH']:
        with open(cfg['HISTORY_SAVE_PATH'], 'w') as f:
            json.dump(history, f, indent=2)

    if test_acc >= 75.0:
        print("Target achieved! (>75%)")
    else:
        print(f"Target not reached. Current: {test_acc:.2f}%, Target: 75%")

    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train emotion recognition model')
    parser.add_argument('--version', type=str, default=None,
                        choices=['v1', 'v2', 'v4'],
                        help='Model version to train (v1, v2, v4). If not specified, uses default config.')
    args = parser.parse_args()

    model, history = train_model(version=args.version)
