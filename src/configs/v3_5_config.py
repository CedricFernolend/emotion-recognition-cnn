"""
V3.5 Configuration: V4 Architecture without Attention

This model uses the same architecture as V4 but without:
- SE (Squeeze-and-Excitation) attention
- Spatial attention
"""
from configs.base_config import *

VERSION = "v3.5"
VERSION_NAME = "V4 Architecture without Attention"
VERSION_DESCRIPTION = "4-block model without SE or Spatial attention mechanisms"

# Architecture
NUM_BLOCKS = 4
USE_SE_ATTENTION = False
USE_SPATIAL_ATTENTION = False
FILTER_PROGRESSION = [64, 128, 256, 512]  # 4 blocks
CLASSIFIER_HIDDEN = 128

# Training hyperparameters (same as v4)
LEARNING_RATE = 0.0003
NUM_EPOCHS = 60  # High value - early stopping will handle termination
DROPOUT_RATE = 0.6
USE_CLASS_WEIGHTS = True
LABEL_SMOOTHING = 0.15
WEIGHT_DECAY = 5e-4
USE_LR_SCHEDULER = True
LR_SCHEDULER_FACTOR = 0.5
LR_SCHEDULER_PATIENCE = 5
LR_MIN = 1e-6

# Data augmentation (full pipeline, same as v4)
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
