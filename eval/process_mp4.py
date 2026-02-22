import os
import sys
import argparse
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from models.v3_model import EmotionCNN

EMOTION_LABELS = ['Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear']
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'final_model.pth')

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

# GradCAM hooks
activations, gradients = {}, {}
model.layer4.body[-2].register_forward_hook(
    lambda _m, _i, o: activations.update({'x': o.detach()}))
model.layer4.body[-2].register_full_backward_hook(
    lambda _m, _gi, go: gradients.update({'x': go[0].detach()}))


def get_gradcam(tensor):
    model.zero_grad()
    tensor.requires_grad = True
    output = model(tensor)
    pred_idx = output.argmax(dim=1).item()
    confidence = F.softmax(output, dim=1)[0, pred_idx].item()

    one_hot = torch.zeros_like(output)
    one_hot[0, pred_idx] = 1
    output.backward(gradient=one_hot)

    weights = gradients['x'].mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * activations['x']).sum(dim=1, keepdim=True))
    cam = cam - cam.min()
    cam = (cam / (cam.max() + 1e-8)).squeeze().cpu().numpy()
    return cam, EMOTION_LABELS[pred_idx], confidence


face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def process_video(mp4_path):
    cap = cv2.VideoCapture(mp4_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    base, ext = os.path.splitext(mp4_path)
    out_path = f'{base}_processed{ext}'
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    print(f'Processing {total} frames -> {out_path}')

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.5, minNeighbors=5, minSize=(60, 60))

        if len(faces) > 0:
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            cx, cy = x + fw // 2, y + fh // 2
            half   = int(max(fw, fh) * 1.1 // 2)
            x1, y1 = max(0, cx - half), max(0, cy - half)
            x2, y2 = min(w, cx + half), min(h, cy + half)

            face_crop = frame[y1:y2, x1:x2].copy()
            tensor = TRANSFORM(
                Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))
            ).unsqueeze(0).to(device)

            cam, label, conf = get_gradcam(tensor)

            cam_resized = cv2.resize(cam, (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
            ksize = max(3, int(min(x2 - x1, y2 - y1) * 0.03) | 1)
            cam_resized = cv2.GaussianBlur(cam_resized, (ksize, ksize), 0)
            cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)
            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            frame[y1:y2, x1:x2] = cv2.addWeighted(face_crop, 0.5, heatmap, 0.5, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, f'{label} {conf*100:.0f}%', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        out.write(frame)
        if (i + 1) % 30 == 0:
            print(f'  {i + 1}/{total}')

    cap.release()
    out.release()
    print('Done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mp4', help='Path to input MP4 file')
    process_video(parser.parse_args().mp4)
