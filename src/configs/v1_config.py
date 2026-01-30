"""
V1 Configuration: Baseline 3-Block CNN (from Preliminary Report)

This is the proposed model from the preliminary report:
- 3 convolutional blocks (no 4th block)
- No attention mechanisms (no SE, no spatial attention)
- Simple skip connections
- Basic data augmentation
- Standard training without advanced techniques
"""
from configs.base_config import *

VERSION = "v1"
VERSION_NAME = "Baseline 3-Block CNN"
VERSION_DESCRIPTION = "Proposed model from preliminary report - simple architecture without attention"

# Architecture
NUM_BLOCKS = 3
USE_SE_ATTENTION = False
USE_SPATIAL_ATTENTION = False
FILTER_PROGRESSION = [64, 128, 256]  # 3 blocks
CLASSIFIER_HIDDEN = 128

# Training hyperparameters
LEARNING_RATE = 0.0003  # Per preliminary report (3×10⁻⁴)
NUM_EPOCHS = 60
DROPOUT_RATE = 0.5
USE_CLASS_WEIGHTS = True  # Enable to handle class imbalance
LABEL_SMOOTHING = 0.0  # No label smoothing
WEIGHT_DECAY = 0.0  # No weight decay
USE_LR_SCHEDULER = False  # No LR scheduling

# Data augmentation (basic only)
AUGMENTATION = {
    'horizontal_flip': True,
    'rotation': 15,
    'translate': 0.1,
    'color_jitter': False,
    'gaussian_blur': False,
    'random_erasing': False,
}

# Version-specific paths
_paths = get_version_paths(VERSION)
MODEL_SAVE_PATH = _paths['model_path']
HISTORY_SAVE_PATH = _paths['history_path']
VIZ_PATH = _paths['viz_dir']
