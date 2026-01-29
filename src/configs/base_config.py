"""Base configuration shared across all model versions."""
import os

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(PROJECT_ROOT, "RAF-DB/")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data/processed/")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results/")

# Data constants
IMAGE_SIZE = 64
NUM_CLASSES = 6
EMOTION_LABELS = ['happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

# Shared training defaults
BATCH_SIZE = 32
VAL_SPLIT = 0.2
SEED = 42

# Early stopping
EARLY_STOPPING_PATIENCE = 10

# Webcam settings (unchanged across versions)
WEBCAM_CAMERA_ID = 0
WEBCAM_CONFIDENCE_THRESHOLD = 0.0
WEBCAM_RESOLUTION = (640, 480)
WEBCAM_FPS_DISPLAY = True


def get_version_paths(version: str):
    """Get version-specific paths for results."""
    version_dir = os.path.join(RESULTS_PATH, version)
    return {
        'model_dir': os.path.join(version_dir, 'models'),
        'viz_dir': os.path.join(version_dir, 'visualizations'),
        'model_path': os.path.join(version_dir, 'models', 'emotion_model.pth'),
        'history_path': os.path.join(version_dir, 'history.json'),
    }
