"""
V4 Configuration: Final Model with Full Enhancements

This is the current production model:
- 4 convolutional blocks with ImprovedBlock architecture
- SE attention + Spatial attention in each block
- Full data augmentation pipeline
- Advanced training: label smoothing, class weights, LR scheduling
"""
from configs.base_config import *

VERSION = "v4"
VERSION_NAME = "Final Model with Attention"
VERSION_DESCRIPTION = "Full model with SE + Spatial attention, advanced training techniques"

# Architecture
NUM_BLOCKS = 4
USE_SE_ATTENTION = True
USE_SPATIAL_ATTENTION = True
FILTER_PROGRESSION = [64, 128, 256, 512]  # 4 blocks
CLASSIFIER_HIDDEN = 128

# Training hyperparameters
LEARNING_RATE = 0.0003
NUM_EPOCHS = 60
DROPOUT_RATE = 0.5
USE_CLASS_WEIGHTS = True
LABEL_SMOOTHING = 0.1
WEIGHT_DECAY = 1e-4
USE_LR_SCHEDULER = True
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5
LR_MIN = 1e-6

# Data augmentation (full pipeline)
AUGMENTATION = {
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

# Version-specific paths
_paths = get_version_paths(VERSION)
MODEL_SAVE_PATH = _paths['model_path']
HISTORY_SAVE_PATH = _paths['history_path']
VIZ_PATH = _paths['viz_dir']
