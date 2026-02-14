"""
Presentation Demo V2: Real-time Emotion Recognition with Multiple Visualizations
(Optimized for cleaner/smoother heatmaps)

Visualization Modes:
1. Detection Only - Just bounding box
2. GradCAM - Class-specific activation (what matters for THIS prediction)
3. Spatial Attention - Raw attention maps from model (V3 only)

Controls:
- '1-3': Switch visualization mode
- 'g': Toggle grayscale input mode
- 'm': Switch between V3 and V1 models
- SPACE: Pause/unpause display
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

# --- Configuration ---
EMOTION_LABELS = ['Happiness', 'Surprise', 'Sadness', 'Anger', 'Disgust', 'Fear']

EMOTION_COLORS = {
    'Happiness': (0, 200, 0),
    'Surprise': (0, 200, 200),
    'Sadness': (200, 100, 0),
    'Anger': (0, 0, 200),
    'Disgust': (0, 100, 150),
    'Fear': (150, 0, 150)
}

VIZ_MODES = {
    1: 'Detection',
    2: 'GradCAM',
    3: 'Attention'  # V3 only
}

MODEL_INFO = {
    'v3': {
        'name': 'V3: Dual Attention Model',
        'has_spatial_attention': True
    },
    'v1': {
        'name': 'V1: Baseline Model',
        'has_spatial_attention': False
    }
}

# Paths
from configs.base_config import RESULTS_PATH
SAVE_DIR = os.path.join(RESULTS_PATH, 'demo_screenshots')


class MultiVisualizer:
    """Multiple visualization methods for the emotion model."""

    def __init__(self, model, device, model_version='v3'):
        self.model = model
        self.device = device
        self.model_version = model_version
        self.activations = {}
        self.gradients = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks to capture activations and gradients."""
        # Target the last conv layer in layer4 (V3) or block3 (V1) for GradCAM
        if self.model_version == 'v3':
            target_layer = self.model.layer4.body[-2]  # Last Conv2d
        else:  # v1
            target_layer = self.model.block3.body[-2]  # Last Conv2d

        def forward_hook(module, input, output):
            self.activations['target_layer'] = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['target_layer'] = grad_output[0].detach()

        # Hooks for GradCAM
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

        # Capture spatial attention outputs (V3 only)
        if self.model_version == 'v3':
            for name in ['layer1', 'layer2', 'layer3', 'layer4']:
                layer = getattr(self.model, name)

                def make_attention_hook(layer_name):
                    def hook(module, input, output):
                        x = input[0]
                        avg_out = torch.mean(x, dim=1, keepdim=True)
                        max_out, _ = torch.max(x, dim=1, keepdim=True)
                        concat = torch.cat([avg_out, max_out], dim=1)
                        attn = module.sigmoid(module.conv(concat))
                        self.activations[f'{layer_name}_spatial_attn'] = attn.detach()
                    return hook

                layer.spatial.register_forward_hook(make_attention_hook(name))

    def get_gradcam(self, input_tensor, target_class=None):
        """Generate GradCAM visualization."""
        self.model.zero_grad()
        input_tensor.requires_grad = True

        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        gradients = self.gradients.get('target_layer')
        activations = self.activations.get('target_layer')

        if gradients is None or activations is None:
            return None, target_class, confidence, probs[0].detach().cpu().numpy()

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (64, 64), interpolation=cv2.INTER_CUBIC)
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, target_class, confidence, probs[0].detach().cpu().numpy()

    def get_spatial_attention(self, input_tensor):
        """Get combined spatial attention maps (V3 only)."""
        if self.model_version != 'v3':
            return None, 0, 0.0, np.zeros(6)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()

        weights = {'layer1': 0.1, 'layer2': 0.2, 'layer3': 0.3, 'layer4': 0.4}
        combined = None

        for name, weight in weights.items():
            attn = self.activations.get(f'{name}_spatial_attn')
            if attn is not None:
                attn_resized = F.interpolate(attn, size=(64, 64), mode='bicubic', align_corners=False)
                if combined is None:
                    combined = attn_resized * weight
                else:
                    combined = combined + attn_resized * weight

        if combined is None:
            return None, pred_class, confidence, probs[0].cpu().numpy()

        combined = combined.squeeze().cpu().numpy()
        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)

        return combined, pred_class, confidence, probs[0].cpu().numpy()


class EmotionDemo:
    """Presentation-friendly emotion recognition demo with multiple visualizations."""

    def __init__(self, camera_id=0):
        print("=" * 60)
        print("  Emotion Recognition Demo V2")
        print("=" * 60)

        # Device setup
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device('cpu')
            print("Using CPU")

        # Current model version
        self.current_version = 'v3'
        self.model = None
        self.visualizer = None
        self._load_current_model()

        self.viz_mode = 1  # Default: Detection only
        self.grayscale_mode = False
        self.paused = False
        self.paused_frame = None
        self.paused_face_data = None  # Store face crop and coordinates when paused

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

        os.makedirs(SAVE_DIR, exist_ok=True)

        print("\n" + "=" * 60)
        print("  Controls:")
        print("    '1' - Detection only")
        print("    '2' - GradCAM (class-specific)")
        print("    '3' - Spatial Attention (V3 only)")
        print("    'g' - Toggle grayscale input mode")
        print("    'm' - Switch model (V3 <-> V1)")
        print("    SPACE - Pause/unpause display")
        print("    's' - Save screenshot")
        print("    'q' - Quit")
        print("=" * 60 + "\n")

    def _load_current_model(self):
        """Load the current model version."""
        from models import load_model
        print(f"\nLoading {self.current_version.upper()} model...")

        model = load_model(self.current_version)
        model = model.to(self.device)
        model.eval()

        self.model = model
        self.visualizer = MultiVisualizer(model, self.device, self.current_version)
        print(f"Model loaded: {MODEL_INFO[self.current_version]['name']}")

    def switch_model(self):
        """Switch between V3 and V1 models."""
        new_version = 'v1' if self.current_version == 'v3' else 'v3'
        self.current_version = new_version
        self._load_current_model()

        # If switching to V1 while in Attention mode, fall back to Detection
        if self.viz_mode == 3 and not MODEL_INFO[self.current_version]['has_spatial_attention']:
            self.viz_mode = 1
            print("Spatial Attention not available for V1, switched to Detection mode")

    def preprocess_face(self, face_bgr):
        if self.grayscale_mode:
            gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
            face_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        else:
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        face_tensor = self.transform(face_pil).unsqueeze(0)
        return face_tensor.to(self.device)

    def apply_heatmap(self, face_crop, heatmap, colormap=cv2.COLORMAP_JET, alpha=0.65):
        """Apply heatmap overlay to face crop with smoothing."""
        h, w = face_crop.shape[:2]

        heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)

        kernel_size = int(min(w, h) * 0.03)
        if kernel_size % 2 == 0:
            kernel_size += 1

        heatmap_blurred = cv2.GaussianBlur(heatmap_resized, (kernel_size, kernel_size), 0)

        min_val = heatmap_blurred.min()
        max_val = heatmap_blurred.max()
        if max_val - min_val > 1e-8:
            heatmap_final = (heatmap_blurred - min_val) / (max_val - min_val)
        else:
            heatmap_final = heatmap_blurred

        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_final), colormap)
        overlay = cv2.addWeighted(face_crop, 1 - alpha, heatmap_colored, alpha, 0)
        return overlay

    def draw_probability_bars(self, frame, probabilities, x, y, width=200, height=20):
        bg_height = len(EMOTION_LABELS) * (height + 5) + 40
        overlay = frame.copy()
        cv2.rectangle(overlay, (x - 10, y - 30), (x + width + 60, y + bg_height), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        cv2.putText(frame, "Emotion Probabilities", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for i, (emotion, prob) in enumerate(zip(EMOTION_LABELS, probabilities)):
            bar_y = y + i * (height + 5)
            color = EMOTION_COLORS.get(emotion, (200, 200, 200))

            cv2.rectangle(frame, (x, bar_y), (x + width, bar_y + height), (60, 60, 60), -1)
            bar_width = int(width * prob)
            cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + height), color, -1)

            cv2.putText(frame, f"{emotion[:3]}", (x + 5, bar_y + height - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"{prob*100:.1f}%", (x + width + 5, bar_y + height - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    def draw_info_panel(self, frame, fps, detected_emotion=None, confidence=None):
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 50), (40, 40, 40), -1)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

        # Model name
        model_name = MODEL_INFO[self.current_version]['name']
        cv2.putText(frame, model_name, (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)

        # Mode
        mode_name = VIZ_MODES.get(self.viz_mode, 'Unknown')
        if self.viz_mode == 3 and not MODEL_INFO[self.current_version]['has_spatial_attention']:
            mode_name = "N/A (V3 only)"
        mode_colors = {1: (0, 255, 0), 2: (0, 165, 255), 3: (0, 255, 255)}
        mode_color = mode_colors.get(self.viz_mode, (255, 255, 255))
        cv2.putText(frame, f"MODE: {mode_name}", (w - 280, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 80, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # Status indicators at bottom
        status_y = h - 10
        status_items = []

        if self.grayscale_mode:
            status_items.append("GRAYSCALE")

        if self.paused:
            status_items.append("PAUSED")

        if status_items:
            status_text = " | ".join(status_items)
            cv2.putText(frame, status_text, (15, status_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if detected_emotion and confidence:
            color = EMOTION_COLORS.get(detected_emotion, (255, 255, 255))
            text = f"{detected_emotion}: {confidence*100:.0f}%"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(frame, text, (text_x, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    def process_frame(self, frame, face_crop, x1, y1, x2, y2):
        """Process a frame and return visualization data."""
        h, w = frame.shape[:2]
        display_frame = frame.copy()

        if face_crop is None or face_crop.size == 0:
            return display_frame, None, None, None

        face_tensor = self.preprocess_face(face_crop)

        heatmap = None
        if self.viz_mode == 1:
            with torch.no_grad():
                output = self.model(face_tensor)
                probs = F.softmax(output, dim=1)
                pred_idx = output.argmax(dim=1).item()
                confidence = probs[0, pred_idx].item()
                last_probs = probs[0].cpu().numpy()
        elif self.viz_mode == 2:
            heatmap, pred_idx, confidence, last_probs = \
                self.visualizer.get_gradcam(face_tensor)
        elif self.viz_mode == 3:
            if MODEL_INFO[self.current_version]['has_spatial_attention']:
                heatmap, pred_idx, confidence, last_probs = \
                    self.visualizer.get_spatial_attention(face_tensor)
            else:
                # Fallback to detection for V1
                with torch.no_grad():
                    output = self.model(face_tensor)
                    probs = F.softmax(output, dim=1)
                    pred_idx = output.argmax(dim=1).item()
                    confidence = probs[0, pred_idx].item()
                    last_probs = probs[0].cpu().numpy()

        last_emotion = EMOTION_LABELS[pred_idx]
        last_confidence = confidence

        # Apply visualization
        if heatmap is not None and self.viz_mode > 1:
            face_with_viz = self.apply_heatmap(face_crop, heatmap)
            display_frame[y1:y2, x1:x2] = face_with_viz
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        else:
            color = EMOTION_COLORS.get(last_emotion, (255, 255, 255))
            if self.grayscale_mode:
                gray_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
                gray_face_bgr = cv2.cvtColor(gray_face, cv2.COLOR_GRAY2BGR)
                display_frame[y1:y2, x1:x2] = gray_face_bgr
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)

        return display_frame, last_probs, last_emotion, last_confidence

    def run(self):
        print("Starting demo... Press 'q' to quit.\n")

        frame_count = 0
        fps = 0
        prev_time = time.time()
        last_probs = None
        last_emotion = None
        last_confidence = None

        try:
            while True:
                if not self.paused:
                    ret, frame = self.cap.read()
                    if not ret:
                        break

                    h, w = frame.shape[:2]

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = self.face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                    )

                    face_crop = None
                    x1, y1, x2, y2 = 0, 0, 0, 0

                    if len(faces) > 0:
                        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])

                        center_x, center_y = x + fw // 2, y + fh // 2
                        crop_size = max(fw, fh) * 1.5
                        half_size = int(crop_size // 2)

                        x1 = max(0, center_x - half_size)
                        y1 = max(0, center_y - half_size)
                        x2 = min(w, center_x + half_size)
                        y2 = min(h, center_y + half_size)

                        face_crop = frame[y1:y2, x1:x2].copy()

                    # Store for potential pause
                    self.paused_frame = frame.copy()
                    self.paused_face_data = (face_crop, x1, y1, x2, y2) if face_crop is not None else None

                    display_frame, last_probs, last_emotion, last_confidence = \
                        self.process_frame(frame, face_crop, x1, y1, x2, y2)

                else:
                    # Paused mode - reprocess stored frame (allows grayscale toggle and model switch)
                    if self.paused_frame is not None and self.paused_face_data is not None:
                        face_crop, x1, y1, x2, y2 = self.paused_face_data
                        display_frame, last_probs, last_emotion, last_confidence = \
                            self.process_frame(self.paused_frame, face_crop, x1, y1, x2, y2)
                    else:
                        display_frame = self.paused_frame.copy() if self.paused_frame is not None else np.zeros((720, 1280, 3), dtype=np.uint8)

                h, w = display_frame.shape[:2]

                if last_probs is not None:
                    self.draw_probability_bars(display_frame, last_probs, w - 280, 70)

                frame_count += 1
                if time.time() - prev_time >= 1.0:
                    fps = frame_count / (time.time() - prev_time)
                    frame_count = 0
                    prev_time = time.time()

                self.draw_info_panel(display_frame, fps, last_emotion, last_confidence)

                cv2.imshow('Emotion Recognition Demo V2', display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key in [ord('1'), ord('2'), ord('3')]:
                    new_mode = int(chr(key))
                    # Check if Attention mode is valid for current model
                    if new_mode == 3 and not MODEL_INFO[self.current_version]['has_spatial_attention']:
                        print("Spatial Attention not available for V1")
                    else:
                        self.viz_mode = new_mode
                        print(f"Mode: {VIZ_MODES[self.viz_mode]}")
                elif key == ord('g'):
                    self.grayscale_mode = not self.grayscale_mode
                    mode_str = "GRAYSCALE" if self.grayscale_mode else "COLOR"
                    print(f"Input mode: {mode_str}")
                elif key == ord('m'):
                    self.switch_model()
                elif key == ord(' '):  # Space bar
                    self.paused = not self.paused
                    state_str = "PAUSED" if self.paused else "RESUMED"
                    print(f"Display: {state_str}")
                elif key == ord('s'):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(SAVE_DIR, f"demo_{timestamp}.png")
                    cv2.imwrite(filename, display_frame)
                    print(f"Saved: {filename}")

        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print("\nDemo ended.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Emotion Recognition Demo V2')
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
