import os
import sys
import csv
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from sklearn.metrics import classification_report
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from models.v3_model import EmotionCNN

EMOTION_LABELS = ['happiness', 'surprise', 'sadness', 'anger', 'disgust', 'fear']
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'final_model.pth')
DATA_DIR   = os.path.join(os.path.dirname(__file__), 'eval_data')
OUT_DIR    = os.path.join(os.path.dirname(__file__), 'eval_results')

TRANSFORM = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

device = torch.device('cuda' if torch.cuda.is_available() else
                      'mps' if torch.backends.mps.is_available() else 'cpu')

model = EmotionCNN(num_classes=6)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device).eval()

# collect all images with their true label index
samples = []
for idx, emotion in enumerate(EMOTION_LABELS):
    folder = os.path.join(DATA_DIR, emotion)
    if not os.path.isdir(folder):
        continue
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            samples.append((os.path.join(folder, fname), fname, idx))

os.makedirs(OUT_DIR, exist_ok=True)
csv_path = os.path.join(OUT_DIR, 'predictions.csv')

all_true, all_pred = [], []

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['image_name', 'true_label', 'predicted_label', 'match'])

    for img_path, fname, true_idx in tqdm(samples, desc='Predicting'):
        img = Image.open(img_path).convert('RGB')
        tensor = TRANSFORM(img).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_idx = model(tensor).argmax(dim=1).item()

        match = int(pred_idx == true_idx)
        writer.writerow([fname, EMOTION_LABELS[true_idx], EMOTION_LABELS[pred_idx], match])
        all_true.append(true_idx)
        all_pred.append(pred_idx)

all_true = np.array(all_true)
all_pred = np.array(all_pred)

print(f'\nTest Accuracy: {100.0 * np.mean(all_true == all_pred):.2f}%')
print('\nPer-class results:')
print(classification_report(all_true, all_pred, target_names=EMOTION_LABELS))
print(f'CSV saved to: {csv_path}')
