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
LEARNING_RATE = 0.0001      # Lower than V1 (was 0.001)
NUM_EPOCHS = 60
DROPOUT_RATE = 0.5
USE_CLASS_WEIGHTS = True    # Keep from V1 to address imbalance
LABEL_SMOOTHING = 0.0       # Not using yet
WEIGHT_DECAY = 1e-4         # Light regularization
USE_LR_SCHEDULER = False

# Augmentation (same as V1)
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
