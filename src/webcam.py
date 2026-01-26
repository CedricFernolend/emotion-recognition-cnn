"""Real-time emotion recognition from webcam with Grad-CAM visualization."""

import os
import time
import datetime
import argparse
import cv2
import torch
import numpy as np
from PIL import Image

from config import EMOTION_LABELS, MODEL_SAVE_PATH, RESULTS_PATH
from model import load_model
from data import get_transforms
from gradcam import GradCAM
from spatial_attention import SpatialAttentionExtractor


class EmotionWebcam:
    """Real-time emotion recognition from webcam with Grad-CAM support."""

    EMOTION_COLORS = {
        'happiness': (0, 255, 0),
        'surprise': (255, 255, 0),
        'sadness': (255, 0, 0),
        'anger': (0, 0, 255),
        'disgust': (0, 128, 255),
        'fear': (128, 0, 128)
    }

    def __init__(self, model_path=MODEL_SAVE_PATH, camera_id=0):
        print("Initializing emotion recognition system...")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load Model
        print(f"Loading model from {model_path}")
        self.model = load_model(model_path).to(self.device)
        self.model.eval()
        
        # Initialize Grad-CAM
        self.gradcam = GradCAM(self.model)
        self.show_gradcam = False

        # Initialize Spatial Attention Extractor
        self.attention_extractor = SpatialAttentionExtractor(self.model)
        self.show_attention = False 

        self.transform = get_transforms(augment=False)

        # Face Detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")

        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot access webcam (camera_id={camera_id})")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def preprocess_face(self, face_bgr):
        """Convert BGR face crop to model input tensor."""
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        face_tensor = self.transform(face_pil).unsqueeze(0)
        return face_tensor.to(self.device)

    def get_gradcam_overlay(self, face_bgr, face_tensor):
        """Generates the Grad-CAM heatmap overlaid on the face BGR image."""
        heatmap, pred_idx = self.gradcam.generate(face_tensor)

        # Resize heatmap to match face crop
        heatmap_resized = cv2.resize(heatmap, (face_bgr.shape[1], face_bgr.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        # Overlay heatmap on the BGR face
        overlay = cv2.addWeighted(face_bgr, 0.6, heatmap_color, 0.4, 0)
        return overlay, EMOTION_LABELS[pred_idx]

    def get_attention_overlay(self, face_bgr, face_tensor):
        """Generates spatial attention heatmap overlaid on the face BGR image.

        Uses combined attention from all layers, weighted toward deeper layers.
        """
        heatmap, pred_idx = self.attention_extractor.get_combined_attention(face_tensor)

        # Resize heatmap to match face crop
        heatmap_resized = cv2.resize(heatmap, (face_bgr.shape[1], face_bgr.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)

        # Overlay heatmap on the BGR face
        overlay = cv2.addWeighted(face_bgr, 0.6, heatmap_color, 0.4, 0)
        return overlay, EMOTION_LABELS[pred_idx]

    def get_attention_multiscale(self, face_bgr, face_tensor, target_size):
        """Creates a multi-scale visualization showing attention at each layer."""
        from spatial_attention import MultiScaleAttentionVisualizer
        visualizer = MultiScaleAttentionVisualizer(self.model)
        return visualizer.create_visualization(face_tensor, face_bgr, target_size)

    def draw_emotion_probabilities(self, frame, probabilities):
        """Draw all emotion probabilities in the bottom left corner."""
        x_start, y_start = 10, frame.shape[0] - 150
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start - 5, y_start - 25), (x_start + 200, y_start + 125), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, "Probabilities:", (x_start, y_start - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for i, (emotion, prob) in enumerate(zip(EMOTION_LABELS, probabilities)):
            y_pos = y_start + 20 + i * 20
            color = self.EMOTION_COLORS.get(emotion, (255, 255, 255))
            cv2.putText(frame, f"{emotion:10s}: {prob*100:5.1f}%", (x_start, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def run(self, show_fps=True, confidence_threshold=0.0):
        print("\nStarting webcam...")
        print("Controls: 'q': Quit | 's': Save | 'g': Grad-CAM | 'a': Spatial Attention | 'A': Multi-scale Attention\n")

        frame_count = 0
        fps = 0
        prev_time = time.time()
        latest_probabilities = None

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret: break

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
                
                display_frame = frame.copy()

                if (self.show_gradcam or self.show_attention) and len(faces) > 0:
                    # MODE: HEATMAP VISUALIZATION (Zoomed)
                    (x, y, w, h) = max(faces, key=lambda f: f[2] * f[3])
                    pad = int(w * 0.15)
                    x1, y1 = max(0, x-pad), max(0, y-pad)
                    x2, y2 = min(frame.shape[1], x+w+pad), min(frame.shape[0], y+h+pad)

                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.size > 0:
                        face_tensor = self.preprocess_face(face_crop)

                        if self.show_attention == 'multi':
                            # Multi-scale attention: show all layers side by side
                            display_frame, label = self.get_attention_multiscale(
                                face_crop, face_tensor, (frame.shape[1], frame.shape[0]))
                            cv2.putText(display_frame, f"Spatial Attention (Multi): {label}", (20, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                        elif self.show_attention:
                            # Combined spatial attention overlay
                            attn_overlay, label = self.get_attention_overlay(face_crop, face_tensor)
                            display_frame = cv2.resize(attn_overlay, (frame.shape[1], frame.shape[0]),
                                                       interpolation=cv2.INTER_CUBIC)
                            cv2.putText(display_frame, f"Spatial Attention: {label}", (20, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        else:
                            # Grad-CAM overlay
                            grad_overlay, label = self.get_gradcam_overlay(face_crop, face_tensor)
                            display_frame = cv2.resize(grad_overlay, (frame.shape[1], frame.shape[0]),
                                                       interpolation=cv2.INTER_CUBIC)
                            cv2.putText(display_frame, f"Grad-CAM: {label}", (20, 40),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                elif len(faces) == 0 and (self.show_gradcam or self.show_attention):
                    # No face detected in heatmap mode
                    cv2.putText(display_frame, "No face detected", (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    # MODE: STANDARD
                    for (x, y, w, h) in faces:
                        face_crop = frame[y:y+h, x:x+w]
                        try:
                            face_tensor = self.preprocess_face(face_crop)
                            with torch.no_grad():
                                outputs = self.model(face_tensor)
                                probs = torch.softmax(outputs, dim=1)
                                conf, idx = torch.max(probs, 1)
                                latest_probabilities = probs.cpu().numpy()[0]
                                
                            if conf.item() * 100 >= confidence_threshold:
                                emotion = EMOTION_LABELS[idx.item()]
                                color = self.EMOTION_COLORS.get(emotion, (255, 255, 255))
                                cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                                cv2.putText(display_frame, f"{emotion} {conf.item()*100:.1f}%", (x, y-10), 
                                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        except: continue

                    if latest_probabilities is not None:
                        self.draw_emotion_probabilities(display_frame, latest_probabilities)

                # UI Overlays
                if show_fps:
                    frame_count += 1
                    if time.time() - prev_time >= 1.0:
                        fps = frame_count / (time.time() - prev_time)
                        frame_count, prev_time = 0, time.time()
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (frame.shape[1]-120, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                if self.show_attention == 'multi':
                    status_text = "MODE: Multi-scale Attention"
                elif self.show_attention:
                    status_text = "MODE: Spatial Attention"
                elif self.show_gradcam:
                    status_text = "MODE: Grad-CAM"
                else:
                    status_text = "MODE: Standard (g/a/A)"
                cv2.putText(display_frame, status_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

                cv2.imshow('Emotion Recognition', display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'): break
                elif key == ord('g'):
                    self.show_gradcam = not self.show_gradcam
                    self.show_attention = False  # Disable attention when enabling gradcam
                elif key == ord('a'):
                    # Toggle combined spatial attention
                    if self.show_attention == True:
                        self.show_attention = False
                    else:
                        self.show_attention = True
                        self.show_gradcam = False
                elif key == ord('A'):
                    # Toggle multi-scale attention view
                    if self.show_attention == 'multi':
                        self.show_attention = False
                    else:
                        self.show_attention = 'multi'
                        self.show_gradcam = False
                elif key == ord('s'):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    fn = os.path.join(RESULTS_PATH, f"webcam_{timestamp}.png")
                    cv2.imwrite(fn, display_frame)
                    print(f"Saved: {fn}")

        finally:
            self.cap.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Real-time emotion recognition from webcam')
    parser.add_argument('--model', type=str, default=MODEL_SAVE_PATH, help='Path to trained model')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--threshold', type=float, default=0.0, help='Confidence threshold (0-100)')
    parser.add_argument('--no-fps', action='store_true', help='Hide FPS counter')

    args = parser.parse_args()

    try:
        webcam = EmotionWebcam(model_path=args.model, camera_id=args.camera)
        webcam.run(show_fps=not args.no_fps, confidence_threshold=args.threshold)
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


if __name__ == "__main__":
    main()
