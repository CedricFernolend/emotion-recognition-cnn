"""Setup clean FER2013 dataset from Kaggle archive."""

import os
import shutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE_DIR = os.path.join(PROJECT_ROOT, "archive_extracted")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "data/raw")
OLD_DATA_BACKUP = os.path.join(PROJECT_ROOT, "data/raw_old")

EMOTION_MAP = {
    "angry": "anger",
    "disgust": "disgust",
    "fear": "fear",
    "happy": "happiness",
    "sad": "sadness",
    "surprise": "surprise",
}

def setup_dataset():
    print("Setting up clean FER2013 dataset...\n")

    if os.path.exists(OUTPUT_DIR):
        print(f"Backing up old data to {OLD_DATA_BACKUP}")
        if os.path.exists(OLD_DATA_BACKUP):
            shutil.rmtree(OLD_DATA_BACKUP)
        shutil.move(OUTPUT_DIR, OLD_DATA_BACKUP)

    for split in ["train", "test"]:
        for emotion in EMOTION_MAP.values():
            os.makedirs(os.path.join(OUTPUT_DIR, split, emotion), exist_ok=True)

    for split in ["train", "test"]:
        print(f"Processing {split} data...")
        src_dir = os.path.join(ARCHIVE_DIR, split)

        for archive_emotion, our_emotion in EMOTION_MAP.items():
            emotion_dir = os.path.join(src_dir, archive_emotion)
            if not os.path.exists(emotion_dir):
                continue

            images = [f for f in os.listdir(emotion_dir) if f.endswith(('.jpg', '.png'))]

            for img in images:
                src = os.path.join(emotion_dir, img)
                dst = os.path.join(OUTPUT_DIR, split, our_emotion, img)
                shutil.copy2(src, dst)

            print(f"  {our_emotion}: {len(images)}")

    print("\nDataset setup complete!")
    print(f"Old data backed up to: {OLD_DATA_BACKUP}")

if __name__ == "__main__":
    setup_dataset()
