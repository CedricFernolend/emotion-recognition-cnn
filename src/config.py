# Project configuration - change these settings as needed

# Data paths
DATA_PATH = "data/raw/"
PROCESSED_PATH = "data/processed/"
RESULTS_PATH = "results/"

# Image settings
IMAGE_SIZE = 64
NUM_CLASSES = 6
EMOTION_LABELS = ['happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']

# Training settings
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
NUM_EPOCHS = 50
DROPOUT_RATE = 0.5

# Model will be saved here
MODEL_SAVE_PATH = "results/models/emotion_model.pth"
