# Data loading and preprocessing

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from config import IMAGE_SIZE, BATCH_SIZE, DATA_PATH, EMOTION_LABELS


class EmotionDataset(Dataset):
    """
    Dataset for loading emotion images.

    Put your images in folders like this:
    data/raw/train/happiness/img1.jpg
    data/raw/train/anger/img2.jpg
    data/raw/test/sadness/img3.jpg
    """

    def __init__(self, root_dir, split='train', transform=None):
        """
        Args:
            root_dir: Path to data folder (e.g., 'data/raw')
            split: 'train', 'val', or 'test'
            transform: Optional image transformations
        """
        self.root_dir = os.path.join(root_dir, split)
        self.transform = transform
        self.images = []
        self.labels = []

        # Load all images and their labels
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


def get_transforms(augment=True):
    """
    Get image transformations.
    augment=True for training data (adds random flips/rotations)
    augment=False for validation/test data
    """
    if augment:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])


def get_dataloaders():
    """
    Create dataloaders for train, validation, and test sets.
    Returns: train_loader, val_loader, test_loader
    """
    train_dataset = EmotionDataset(DATA_PATH, 'train', transform=get_transforms(augment=True))
    val_dataset = EmotionDataset(DATA_PATH, 'val', transform=get_transforms(augment=False))
    test_dataset = EmotionDataset(DATA_PATH, 'test', transform=get_transforms(augment=False))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    return train_loader, val_loader, test_loader
