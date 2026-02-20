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
LEARNING_RATE = 0.00015
NUM_EPOCHS = 60  # High value - early stopping will handle termination
DROPOUT_RATE = 0.5
USE_CLASS_WEIGHTS = True
LABEL_SMOOTHING = 0.10
WEIGHT_DECAY = 5e-4
USE_LR_SCHEDULER = True
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5
LR_MIN = 1e-6

# Data augmentation (full pipeline)
AUGMENTATION = {
    'horizontal_flip': True,            #p(0.5)
    'rotation': 15,                     #rotation 15 dagree
    'translate': 0.1,                   #shift up to 10% in x/y
    'color_jitter': True,               #:NEW: brightness and contrast variation
        'color_jitter_brightness': 0.2,
        'color_jitter_contrast': 0.2,
        'color_jitter_saturation': 0.2, #needed for rafdb
    'gaussian_blur': True,              #:NEW: 10% chance blur
        'gaussian_blur_prob': 0.1,
    'random_erasing': True,             #:NEW: 10% chance to erase patch scale=[0.02,0.1]
        'random_erasing_prob': 0.1,
    'random_resized_crop': True,
        'random_resized_crop_scale': (0.7, 1.0),
}

# Version-specific paths
_paths = get_version_paths(VERSION)
MODEL_SAVE_PATH = _paths['model_path']
HISTORY_SAVE_PATH = _paths['history_path']
VIZ_PATH = _paths['viz_dir']
