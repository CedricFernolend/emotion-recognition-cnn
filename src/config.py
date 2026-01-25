import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data/processed_hd/")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data/processed/")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results/")

IMAGE_SIZE = 64
NUM_CLASSES = 6
EMOTION_LABELS = ['happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

BATCH_SIZE = 32
LEARNING_RATE = 0.0003
NUM_EPOCHS = 60
DROPOUT_RATE = 0.5
USE_CLASS_WEIGHTS = True  # Weight loss by inverse class frequency

MODEL_SAVE_PATH = "results/models/emotion_model.pth"

WEBCAM_CAMERA_ID = 0
WEBCAM_CONFIDENCE_THRESHOLD = 0.0
WEBCAM_RESOLUTION = (640, 480)
WEBCAM_FPS_DISPLAY = True
