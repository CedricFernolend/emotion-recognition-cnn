"""
V2 Configuration: 4-Block CNN with SE Attention

Changes from V1:
- Lower learning rate (0.0001 vs 0.001) for more stable training
- Class weights enabled to address class imbalance
- 4 blocks instead of 3 (added 512-channel block)
- SE attention enabled
- Light weight decay for regularization

NOT adding yet (save for V3 if needed):
- Focal Loss
- Spatial Attention
- Label Smoothing
"""
from configs.base_config import *

VERSION = "v2"
VERSION_NAME = "V2 - SE Attention + 4 Blocks"
VERSION_DESCRIPTION = "4-block CNN with SE attention, lower LR, class weights"

# Architecture
NUM_BLOCKS = 4
USE_SE_ATTENTION = True
USE_SPATIAL_ATTENTION = False
FILTER_PROGRESSION = [64, 128, 256, 512]
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
