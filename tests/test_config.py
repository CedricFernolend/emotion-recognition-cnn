import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest

from config import (
    IMAGE_SIZE, NUM_CLASSES, EMOTION_LABELS, BATCH_SIZE,
    LEARNING_RATE, NUM_EPOCHS, DROPOUT_RATE, DATA_PATH,
    PROCESSED_PATH, RESULTS_PATH, MODEL_SAVE_PATH
)


class TestConfig:
    def test_image_size_positive(self):
        """IMAGE_SIZE should be a positive integer."""
        assert isinstance(IMAGE_SIZE, int)
        assert IMAGE_SIZE > 0

    def test_num_classes_matches_labels(self):
        """NUM_CLASSES should match length of EMOTION_LABELS."""
        assert NUM_CLASSES == len(EMOTION_LABELS)

    def test_emotion_labels_are_strings(self):
        """All emotion labels should be strings."""
        for label in EMOTION_LABELS:
            assert isinstance(label, str)
            assert len(label) > 0

    def test_batch_size_positive(self):
        """BATCH_SIZE should be a positive integer."""
        assert isinstance(BATCH_SIZE, int)
        assert BATCH_SIZE > 0

    def test_learning_rate_valid(self):
        """LEARNING_RATE should be a small positive float."""
        assert isinstance(LEARNING_RATE, float)
        assert 0 < LEARNING_RATE < 1

    def test_num_epochs_positive(self):
        """NUM_EPOCHS should be a positive integer."""
        assert isinstance(NUM_EPOCHS, int)
        assert NUM_EPOCHS > 0

    def test_dropout_rate_valid(self):
        """DROPOUT_RATE should be between 0 and 1."""
        assert isinstance(DROPOUT_RATE, float)
        assert 0 <= DROPOUT_RATE < 1

    def test_paths_are_strings(self):
        """All path configs should be strings."""
        assert isinstance(DATA_PATH, str)
        assert isinstance(PROCESSED_PATH, str)
        assert isinstance(RESULTS_PATH, str)
        assert isinstance(MODEL_SAVE_PATH, str)

    def test_expected_emotions(self):
        """EMOTION_LABELS should contain expected emotions."""
        expected = {'happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear'}
        assert set(EMOTION_LABELS) == expected
