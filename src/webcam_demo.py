"""
Presentation Demo: Real-time Emotion Recognition with V3 Model

Features:
- Real-time face detection and emotion classification
- Live probability bars for all 6 emotions
- Spatial attention visualization (shows where the model looks)
- Clean, presentation-friendly UI

Controls:
- 'a': Toggle spatial attention overlay
- 's': Save screenshot
- 'q': Quit

Usage:
    python webcam_demo.py
"""

import os
import time
import datetime
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms


# Configuration
MODEL_VERSION = 'v4'  # Internal name (displayed as V3)
DISPLAY_NAME = 'V3: Spatial Attention Model'
EMOTION_LABELS = ['Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear']

EMOTION_COLORS = {
    'Happiness': (0, 200, 0),      # Green
    'Surprise': (0, 200, 200),     # Yellow
    'Sadness': (200, 100, 0),      # Blue
    'Anger': (0, 0, 200),          # Red
    'Disgust': (0, 100, 150),      # Brown
    'Fear': (150, 0, 150)          # Purple
}

# Paths (relative to this script's location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')
MODEL_PATH = os.path.join(RESULTS_PATH, MODEL_VERSION, 'models', 'emotion_model.pth')
SAVE_DIR = os.path.join(RESULTS_PATH, 'demo_screenshots')


class SpatialAttentionVisualizer:
    """Extract and visualize spatial attention from V3/V4 model."""

    def __init__(self, model):
        self.model = model
        self.attention_maps = []
        self.handles = []
        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks to capture attention maps."""
        for layer_name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self.model, layer_name)

            def make_hook(name):
                def hook(module, input, output):
                    # Capture the attention weights
                    with torch.no_grad():
                        x = input[0]
                        avg_out = torch.mean(x, dim=1, keepdim=True)
                        max_out, _ = torch.max(x, dim=1, keepdim=True)
                        combined = torch.cat([avg_out, max_out], dim=1)
                        attention = module.sigmoid(module.conv(combined))
                        self.attention_maps.append(attention)
                return hook

            handle = layer.spatial.register_forward_hook(make_hook(layer_name))
            self.handles.append(handle)

    def get_attention(self, input_tensor):
        """Get combined attention map from all layers."""
        self.attention_maps = []

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_idx = output.argmax(dim=1).item()
            confidence = probs[0, pred_idx].item()

        if not self.attention_maps:
            return None, pred_idx, confidence, probs[0].cpu().numpy()

        # Combine attention maps (weight deeper layers more)
        weights = [0.1, 0.2, 0.3, 0.4]  # Layer 1, 2, 3, 4
        combined = None

        for att, weight in zip(self.attention_maps, weights):
            att_resized = F.interpolate(att, size=(64, 64), mode='bilinear', align_corners=False)
            if combined is None:
                combined = att_resized * weight
            else:
                combined = combined + att_resized * weight

        # Normalize
        combined = combined.squeeze().cpu().numpy()
        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)

        return combined, pred_idx, confidence, probs[0].cpu().numpy()

    def cleanup(self):
        for handle in self.handles:
            handle.remove()


class EmotionDemo:
    """Presentation-friendly emotion recognition demo."""

    def __init__(self, camera_id=0):
        print("=" * 60)
        print("  Emotion Recognition Demo")
        print(f"  Model: {DISPLAY_NAME}")
        print("=" * 60)

        # Device
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
            print("Using Apple Silicon GPU (MPS)")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        # Load model
        print(f"\nLoading model from: {MODEL_PATH}")
        self.model = self._load_model()
        self.model.eval()
        print("Model loaded successfully!")

        # Attention visualizer
        self.attention_viz = SpatialAttentionVisualizer(self.model)
        self.show_attention = False

        # Transform
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        # Face detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Camera
        print(f"\nOpening camera {camera_id}...")
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot access camera {camera_id}")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        print("Camera ready!")

        # Create save directory
        os.makedirs(SAVE_DIR, exist_ok=True)

        print("\n" + "=" * 60)
        print("  Controls:")
        print("    'a' - Toggle attention visualization")
        print("    's' - Save screenshot")
        print("    'q' - Quit")
        print("=" * 60 + "\n")

    def _load_model(self):
        """Load the V3/V4 model."""
        from models.v4_final import EmotionCNN
        model = EmotionCNN(num_classes=6)
        state_dict = torch.load(MODEL_PATH, map_location=self.device)
        model.load_state_dict(state_dict)
        return model.to(self.device)

    def preprocess_face(self, face_bgr):
        """Convert BGR face to model input tensor."""
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        face_tensor = self.transform(face_pil).unsqueeze(0)
        return face_tensor.to(self.device)

    def draw_probability_bars(self, frame, probabilities, x, y, width=200, height=20):
        """Draw horizontal probability bars for each emotion."""
        # Background
        bg_height = len(EMOTION_LABELS) * (height + 5) + 40
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 10, y - 30), (x + width + 20, y + bg_height), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Title
        cv2.putText(frame, "Emotion Probabilities", (x, y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Bars
        for i, (emotion, prob) in enumerate(zip(EMOTION_LABELS, probabilities)):
            bar_y = y + i * (height + 5)
            color = EMOTION_COLORS.get(emotion, (200, 200, 200))

            # Background bar
            cv2.rectangle(frame, (x, bar_y), (x + width, bar_y + height), (60, 60, 60), -1)

            # Probability bar
            bar_width = int(width * prob)
            cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + height), color, -1)

            # Label and percentage
            cv2.putText(frame, f"{emotion[:3]}", (x + 5, bar_y + height - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"{prob*100:.1f}%", (x + width + 5, bar_y + height - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def draw_info_panel(self, frame, fps, detected_emotion=None, confidence=None):
        """Draw info panel at the top."""
        h, w = frame.shape[:2]

        # Top bar background
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Model name
        cv2.putText(frame, DISPLAY_NAME, (15, 32),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)

        # Mode indicator
        if self.show_attention:
            mode_text = "MODE: Spatial Attention"
            mode_color = (0, 255, 255)
        else:
            mode_text = "MODE: Detection"
            mode_color = (0, 255, 0)
        cv2.putText(frame, mode_text, (w - 250, 32),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

        # FPS
        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 80, 32),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Detected emotion (large, center bottom)
        if detected_emotion and confidence:
            color = EMOTION_COLORS.get(detected_emotion, (255, 255, 255))
            text = f"{detected_emotion}: {confidence*100:.0f}%"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(frame, text, (text_x, h - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    def run(self):
        """Main demo loop."""
        print("Starting demo... Press 'q' to quit.\n")

        frame_count = 0
        fps = 0
        prev_time = time.time()
        last_probs = None
        last_emotion = None
        last_confidence = None

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                display_frame = frame.copy()

                # Detect faces
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )

                if len(faces) > 0:
                    # Get largest face
                    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])

                    # Add padding
                    pad = int(fw * 0.2)
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(w, x + fw + pad)
                    y2 = min(h, y + fh + pad)

                    face_crop = frame[y1:y2, x1:x2]

                    if face_crop.size > 0:
                        face_tensor = self.preprocess_face(face_crop)

                        # Get attention and prediction
                        attention_map, pred_idx, confidence, probs = \
                            self.attention_viz.get_attention(face_tensor)

                        last_probs = probs
                        last_emotion = EMOTION_LABELS[pred_idx]
                        last_confidence = confidence

                        if self.show_attention and attention_map is not None:
                            # Show attention overlay on face
                            att_resized = cv2.resize(attention_map, (x2 - x1, y2 - y1))
                            att_color = cv2.applyColorMap(
                                np.uint8(255 * att_resized), cv2.COLORMAP_JET
                            )
                            face_with_att = cv2.addWeighted(
                                face_crop, 0.5, att_color, 0.5, 0
                            )
                            display_frame[y1:y2, x1:x2] = face_with_att

                            # Draw attention box
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        else:
                            # Draw detection box
                            color = EMOTION_COLORS.get(last_emotion, (255, 255, 255))
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)

                # Draw probability bars
                if last_probs is not None:
                    self.draw_probability_bars(display_frame, last_probs, w - 280, 70)

                # Calculate FPS
                frame_count += 1
                if time.time() - prev_time >= 1.0:
                    fps = frame_count / (time.time() - prev_time)
                    frame_count = 0
                    prev_time = time.time()

                # Draw info panel
                self.draw_info_panel(display_frame, fps, last_emotion, last_confidence)

                # Show frame
                cv2.imshow('Emotion Recognition Demo', display_frame)

                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('a'):
                    self.show_attention = not self.show_attention
                    mode = "Spatial Attention" if self.show_attention else "Detection"
                    print(f"Mode: {mode}")
                elif key == ord('s'):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(SAVE_DIR, f"demo_{timestamp}.png")
                    cv2.imwrite(filename, display_frame)
                    print(f"Saved: {filename}")

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            self.attention_viz.cleanup()
            print("\nDemo ended.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Emotion Recognition Demo')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    args = parser.parse_args()

    try:
        demo = EmotionDemo(camera_id=args.camera)
        demo.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
