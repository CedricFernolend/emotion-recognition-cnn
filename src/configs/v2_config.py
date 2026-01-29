"""
V2 Configuration: To Be Defined

This version will be designed after analyzing v1 results.
Placeholder configuration - copy from v1 and modify as needed.
"""
from configs.base_config import *

VERSION = "v2"
VERSION_NAME = "V2 - TBD"
VERSION_DESCRIPTION = "To be defined after v1 analysis"

# TODO: Define after v1 training results
# Copy settings from v1_config.py and modify based on analysis

# Placeholder - same as v1 for now
NUM_BLOCKS = 3
USE_SE_ATTENTION = False
USE_SPATIAL_ATTENTION = False
FILTER_PROGRESSION = [64, 128, 256]
CLASSIFIER_HIDDEN = 128

LEARNING_RATE = 0.001
NUM_EPOCHS = 60
DROPOUT_RATE = 0.5
USE_CLASS_WEIGHTS = False
LABEL_SMOOTHING = 0.0
WEIGHT_DECAY = 0.0
USE_LR_SCHEDULER = False

AUGMENTATION = {
    'horizontal_flip': True,
    'rotation': 15,
    'translate': 0.1,
    'color_jitter': False,
    'gaussian_blur': False,
    'random_erasing': False,
}

_paths = get_version_paths(VERSION)
MODEL_SAVE_PATH = _paths['model_path']
HISTORY_SAVE_PATH = _paths['history_path']
VIZ_PATH = _paths['viz_dir']
