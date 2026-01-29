"""
Configuration file - backward compatible wrapper.

This file maintains backward compatibility with existing code.
For version-specific configs, use: from configs import load_config
"""
import os

# Import shared constants from base config
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "RAF-DB/")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data/processed/")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results/")

IMAGE_SIZE = 64
NUM_CLASSES = 6
EMOTION_LABELS = ['happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

# Default to v4 settings for backward compatibility
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
NUM_EPOCHS = 60
DROPOUT_RATE = 0.5
USE_CLASS_WEIGHTS = True

MODEL_SAVE_PATH = "results/models/emotion_model.pth"

# Webcam settings
WEBCAM_CAMERA_ID = 0
WEBCAM_CONFIDENCE_THRESHOLD = 0.0
WEBCAM_RESOLUTION = (640, 480)
WEBCAM_FPS_DISPLAY = True
