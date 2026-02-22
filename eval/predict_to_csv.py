import os
import sys
import csv
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
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

# collect all images directly from the flat input folder
samples = []
for fname in os.listdir(DATA_DIR):
    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
        samples.append((os.path.join(DATA_DIR, fname), fname))

if not samples:
    print(f'No images found in {DATA_DIR}')
    sys.exit(0)

os.makedirs(OUT_DIR, exist_ok=True)
csv_path = os.path.join(OUT_DIR, 'predictions.csv')

with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['filename'] + EMOTION_LABELS)

    for img_path, fname in tqdm(samples, desc='Predicting'):
        img = Image.open(img_path).convert('RGB')
        tensor = TRANSFORM(img).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(tensor)
            probs = F.softmax(logits, dim=1).squeeze().tolist()

        writer.writerow([fname] + [f'{p:.4f}' for p in probs])

print(f'\nDone. Predictions saved to: {csv_path}')
