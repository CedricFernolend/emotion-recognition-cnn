import os
import random
from collections import Counter
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from configs.base_config import IMAGE_SIZE, BATCH_SIZE, DATA_PATH, EMOTION_LABELS


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


def get_transforms(augment=True, augment_config=None):
    """
    Get image transformations.

    Args:
        augment: If True, apply augmentation (for training)
        augment_config: Optional dict with augmentation settings.
                       If None, uses full augmentation pipeline.

    Returns:
        transforms.Compose object
    """
    if not augment:
        # Validation/test transforms - no augmentation
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    # Training transforms - build based on config
    transform_list = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]

    if augment_config is None:
        # Default: full augmentation (v4 style)
        augment_config = {
            'horizontal_flip': True,
            'rotation': 15,
            'translate': 0.1,
            'color_jitter': True,
            'color_jitter_brightness': 0.2,
            'color_jitter_contrast': 0.2,
            'color_jitter_saturation': 0.1,
            'gaussian_blur': True,
            'gaussian_blur_prob': 0.1,
            'random_erasing': True,
            'random_erasing_prob': 0.1,
        }

    # Horizontal flip
    if augment_config.get('horizontal_flip', False):
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    # Rotation
    rotation = augment_config.get('rotation', 0)
    if rotation > 0:
        transform_list.append(transforms.RandomRotation(rotation))

    # Translation (affine)
    translate = augment_config.get('translate', 0)
    if translate > 0:
        transform_list.append(transforms.RandomAffine(degrees=0, translate=(translate, translate)))

    # Color jitter
    if augment_config.get('color_jitter', False):
        brightness = augment_config.get('color_jitter_brightness', 0.2)
        contrast = augment_config.get('color_jitter_contrast', 0.2)
        saturation = augment_config.get('color_jitter_saturation', 0.1)
        transform_list.append(transforms.ColorJitter(
            brightness=brightness, contrast=contrast, saturation=saturation
        ))

    # Gaussian blur
    if augment_config.get('gaussian_blur', False):
        prob = augment_config.get('gaussian_blur_prob', 0.1)
        transform_list.append(transforms.RandomApply(
            [transforms.GaussianBlur(kernel_size=3)], p=prob
        ))

    # To tensor and normalize
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]))

    # Random erasing (must be after ToTensor)
    if augment_config.get('random_erasing', False):
        prob = augment_config.get('random_erasing_prob', 0.1)
        transform_list.append(transforms.RandomErasing(p=prob, scale=(0.02, 0.1)))

    return transforms.Compose(transform_list)


def get_dataloaders(val_split=0.2, seed=42, stratified=True, augment_config=None,
                    data_path=None, batch_size=None):
    """
    Create dataloaders for train, validation, and test sets.

    Args:
        val_split: Fraction of training data to use for validation (default: 0.2)
        seed: Random seed for reproducible splits (default: 42)
        stratified: If True, maintain class proportions in train/val split (default: True)
        augment_config: Optional dict with augmentation settings for training
        data_path: Optional override for data path
        batch_size: Optional override for batch size
    """
    data_path = data_path or DATA_PATH
    batch_size = batch_size or BATCH_SIZE

    full_train_dataset = EmotionDataset(data_path, 'train', transform=None)
    test_dataset = EmotionDataset(data_path, 'test', transform=get_transforms(augment=False))

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
    train_transform = get_transforms(augment=True, augment_config=augment_config)
    val_transform = get_transforms(augment=False)

    train_dataset = TransformSubset(full_train_dataset, train_indices, train_transform)
    val_dataset = TransformSubset(full_train_dataset, val_indices, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

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
