import os
import random
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from config import IMAGE_SIZE, BATCH_SIZE, DATA_PATH, EMOTION_LABELS


class TransformSubset(Dataset):
    """Subset of a dataset with a specific transform applied."""

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path = self.dataset.images[self.indices[idx]]
        label = self.dataset.labels[self.indices[idx]]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_labels(self):
        return [self.dataset.labels[i] for i in self.indices]


class EmotionDataset(Dataset):
    """Dataset for loading emotion images from folder structure."""

    def __init__(self, root_dir, split='train', transform=None):
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.images = []
        self.labels = []

        for label_idx, emotion in enumerate(EMOTION_LABELS):
            emotion_folder = os.path.join(self.root_dir, emotion)
            if not os.path.exists(emotion_folder):
                continue

            for img_name in os.listdir(emotion_folder):
                if img_name.endswith(('.jpg', '.jpeg', '.png')):
                    self.images.append(os.path.join(emotion_folder, img_name))
                    self.labels.append(label_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_counts(self):
        """Return count of samples per class."""
        return Counter(self.labels)


def get_transforms(augment=True):
    """Get image transformations. Use augment=True for training, False for val/test."""
    if augment:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


def get_dataloaders(val_split=0.1, seed=42, stratified=True):
    """
    Create dataloaders for train, validation, and test sets.

    Args:
        val_split: Fraction of training data to use for validation (default: 0.1)
        seed: Random seed for reproducible splits (default: 42)
        stratified: If True, maintain class proportions in train/val split (default: True)
    """
    full_train_dataset = EmotionDataset(DATA_PATH, 'train', transform=None)
    test_dataset = EmotionDataset(DATA_PATH, 'test', transform=get_transforms(augment=False))

    random.seed(seed)

    if stratified:
        # Stratified split: same proportion from each class
        train_indices = []
        val_indices = []

        # Group indices by label
        label_to_indices = {}
        for idx, label in enumerate(full_train_dataset.labels):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)

        # Split each class proportionally
        for label, indices in label_to_indices.items():
            random.shuffle(indices)
            n_val = int(len(indices) * val_split)
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])

        random.shuffle(train_indices)
        random.shuffle(val_indices)
    else:
        # Random split (original behavior)
        n_total = len(full_train_dataset)
        n_val = int(n_total * val_split)
        indices = list(range(n_total))
        random.shuffle(indices)
        train_indices = indices[n_val:]
        val_indices = indices[:n_val]

    # Create subset datasets with appropriate transforms
    train_transform = get_transforms(augment=True)
    val_transform = get_transforms(augment=False)

    train_dataset = TransformSubset(full_train_dataset, train_indices, train_transform)
    val_dataset = TransformSubset(full_train_dataset, val_indices, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    print(f"Dataset splits: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")

    return train_loader, val_loader, test_loader


def compute_class_weights(data_path=DATA_PATH):
    """
    Compute class weights based on inverse frequency.
    Returns tensor of weights for CrossEntropyLoss.
    """
    dataset = EmotionDataset(data_path, 'train', transform=None)
    class_counts = dataset.get_class_counts()

    # Get counts in label order
    counts = [class_counts.get(i, 1) for i in range(len(EMOTION_LABELS))]
    total = sum(counts)

    # Inverse frequency weighting: weight = total / (num_classes * count)
    num_classes = len(EMOTION_LABELS)
    weights = [total / (num_classes * c) for c in counts]

    print("Class weights (inverse frequency):")
    for i, (emotion, weight) in enumerate(zip(EMOTION_LABELS, weights)):
        print(f"  {emotion}: {weight:.3f} ({counts[i]} samples)")

    return torch.tensor(weights, dtype=torch.float32)
