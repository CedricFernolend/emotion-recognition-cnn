"""
Presentation Demo V2: Real-time Emotion Recognition with Multiple Visualizations

Visualization Modes:
1. Detection Only - Just bounding box
2. GradCAM - Class-specific activation (what matters for THIS prediction)
3. Feature Activation - Where neurons fire strongest (layer4)
4. Spatial Attention - Raw attention maps from model
5. Occlusion - Sensitivity map (slower but most interpretable)

Controls:
- '1-5': Switch visualization mode
- 's': Save screenshot
- 'q': Quit

Usage:
    python webcam_demo_v2.py
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
MODEL_VERSION = 'v4'
DISPLAY_NAME = 'V3: Spatial Attention Model'
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
    3: 'Activation',
    4: 'Attention',
    5: 'Occlusion'
}

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
RESULTS_PATH = os.path.join(PROJECT_ROOT, 'results')
MODEL_PATH = os.path.join(RESULTS_PATH, MODEL_VERSION, 'models', 'emotion_model.pth')
SAVE_DIR = os.path.join(RESULTS_PATH, 'demo_screenshots')


class MultiVisualizer:
    """Multiple visualization methods for the emotion model."""

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.activations = {}
        self.gradients = {}
        self._register_hooks()

    def _register_hooks(self):
        """Register hooks to capture activations and gradients."""
        # Target the last conv layer in layer4 for GradCAM
        target_layer = self.model.layer4.body[-2]  # Last Conv2d

        def forward_hook(module, input, output):
            self.activations['layer4'] = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients['layer4'] = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

        # Also capture spatial attention outputs
        for name in ['layer1', 'layer2', 'layer3', 'layer4']:
            layer = getattr(self.model, name)

            def make_attention_hook(layer_name):
                def hook(module, input, output):
                    # Output is x * attention_map, input[0] is x
                    # We can derive attention as output / (input + eps)
                    # But better: capture the attention map directly
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

        # Forward pass
        output = self.model(input_tensor)
        probs = F.softmax(output, dim=1)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        confidence = probs[0, target_class].item()

        # Backward pass for target class
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Get gradients and activations
        gradients = self.gradients.get('layer4')
        activations = self.activations.get('layer4')

        if gradients is None or activations is None:
            return None, target_class, confidence, probs[0].detach().cpu().numpy()

        # GradCAM computation
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # Global average pooling of gradients
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # Only positive contributions

        # Normalize
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (64, 64))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, target_class, confidence, probs[0].detach().cpu().numpy()

    def get_activation_map(self, input_tensor):
        """Get feature activation heatmap from layer4."""
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()

        activations = self.activations.get('layer4')
        if activations is None:
            return None, pred_class, confidence, probs[0].cpu().numpy()

        # Sum across channels and normalize
        act_map = torch.mean(activations, dim=1).squeeze().cpu().numpy()
        act_map = cv2.resize(act_map, (64, 64))
        act_map = (act_map - act_map.min()) / (act_map.max() - act_map.min() + 1e-8)

        return act_map, pred_class, confidence, probs[0].cpu().numpy()

    def get_spatial_attention(self, input_tensor):
        """Get combined spatial attention maps."""
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)
            pred_class = output.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()

        # Combine attention maps from all layers (weighted by depth)
        weights = {'layer1': 0.1, 'layer2': 0.2, 'layer3': 0.3, 'layer4': 0.4}
        combined = None

        for name, weight in weights.items():
            attn = self.activations.get(f'{name}_spatial_attn')
            if attn is not None:
                attn_resized = F.interpolate(attn, size=(64, 64), mode='bilinear', align_corners=False)
                if combined is None:
                    combined = attn_resized * weight
                else:
                    combined = combined + attn_resized * weight

        if combined is None:
            return None, pred_class, confidence, probs[0].cpu().numpy()

        combined = combined.squeeze().cpu().numpy()
        combined = (combined - combined.min()) / (combined.max() - combined.min() + 1e-8)

        return combined, pred_class, confidence, probs[0].cpu().numpy()

    def get_occlusion_sensitivity(self, input_tensor, patch_size=8, stride=4):
        """Generate occlusion sensitivity map (slower but interpretable)."""
        with torch.no_grad():
            base_output = self.model(input_tensor)
            base_probs = F.softmax(base_output, dim=1)
            pred_class = base_output.argmax(dim=1).item()
            base_confidence = base_probs[0, pred_class].item()

        _, _, H, W = input_tensor.shape
        sensitivity = np.zeros((H, W))
        counts = np.zeros((H, W))

        # Slide occlusion patch
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                # Create occluded input
                occluded = input_tensor.clone()
                occluded[:, :, y:y+patch_size, x:x+patch_size] = 0  # Black patch

                with torch.no_grad():
                    occ_output = self.model(occluded)
                    occ_probs = F.softmax(occ_output, dim=1)
                    occ_confidence = occ_probs[0, pred_class].item()

                # Sensitivity = confidence drop when this region is occluded
                drop = base_confidence - occ_confidence
                sensitivity[y:y+patch_size, x:x+patch_size] += drop
                counts[y:y+patch_size, x:x+patch_size] += 1

        # Average overlapping regions
        counts[counts == 0] = 1
        sensitivity = sensitivity / counts

        # Normalize to 0-1
        sensitivity = (sensitivity - sensitivity.min()) / (sensitivity.max() - sensitivity.min() + 1e-8)

        return sensitivity, pred_class, base_confidence, base_probs[0].cpu().numpy()


class EmotionDemo:
    """Presentation-friendly emotion recognition demo with multiple visualizations."""

    def __init__(self, camera_id=0):
        print("=" * 60)
        print("  Emotion Recognition Demo V2")
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

        # Visualizer
        self.visualizer = MultiVisualizer(self.model, self.device)
        self.viz_mode = 1  # Default: Detection only

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
        print("    '3' - Feature Activation")
        print("    '4' - Spatial Attention")
        print("    '5' - Occlusion Sensitivity (slow)")
        print("    's' - Save screenshot")
        print("    'q' - Quit")
        print("=" * 60 + "\n")

    def _load_model(self):
        from models.v4_final import EmotionCNN
        model = EmotionCNN(num_classes=6)
        state_dict = torch.load(MODEL_PATH, map_location=self.device, weights_only=True)
        model.load_state_dict(state_dict)
        return model.to(self.device)

    def preprocess_face(self, face_bgr):
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        face_tensor = self.transform(face_pil).unsqueeze(0)
        return face_tensor.to(self.device)

    def apply_heatmap(self, face_crop, heatmap, colormap=cv2.COLORMAP_JET, alpha=0.5):
        """Apply heatmap overlay to face crop."""
        h, w = face_crop.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Apply colormap
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), colormap)

        # Blend with original
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

        cv2.putText(frame, DISPLAY_NAME, (15, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 200, 255), 2)

        # Current mode
        mode_name = VIZ_MODES.get(self.viz_mode, 'Unknown')
        mode_colors = {1: (0, 255, 0), 2: (0, 165, 255), 3: (255, 100, 100),
                       4: (0, 255, 255), 5: (255, 0, 255)}
        mode_color = mode_colors.get(self.viz_mode, (255, 255, 255))
        cv2.putText(frame, f"MODE: {mode_name}", (w - 280, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_color, 2)

        cv2.putText(frame, f"FPS: {fps:.0f}", (w - 80, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        if detected_emotion and confidence:
            color = EMOTION_COLORS.get(detected_emotion, (255, 255, 255))
            text = f"{detected_emotion}: {confidence*100:.0f}%"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)[0]
            text_x = (w - text_size[0]) // 2
            cv2.putText(frame, text, (text_x, h - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    def run(self):
        print("Starting demo... Press 'q' to quit.\n")

        frame_count = 0
        fps = 0
        prev_time = time.time()
        last_probs = None
        last_emotion = None
        last_confidence = None
        occlusion_cache = None
        occlusion_frame_count = 0

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break

                h, w = frame.shape[:2]
                display_frame = frame.copy()

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )

                if len(faces) > 0:
                    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])

                    pad = int(fw * 0.2)
                    x1 = max(0, x - pad)
                    y1 = max(0, y - pad)
                    x2 = min(w, x + fw + pad)
                    y2 = min(h, y + fh + pad)

                    face_crop = frame[y1:y2, x1:x2]

                    if face_crop.size > 0:
                        face_tensor = self.preprocess_face(face_crop)

                        # Get visualization based on mode
                        heatmap = None

                        if self.viz_mode == 1:  # Detection only
                            with torch.no_grad():
                                output = self.model(face_tensor)
                                probs = F.softmax(output, dim=1)
                                pred_idx = output.argmax(dim=1).item()
                                confidence = probs[0, pred_idx].item()
                                last_probs = probs[0].cpu().numpy()

                        elif self.viz_mode == 2:  # GradCAM
                            heatmap, pred_idx, confidence, last_probs = \
                                self.visualizer.get_gradcam(face_tensor)

                        elif self.viz_mode == 3:  # Activation
                            heatmap, pred_idx, confidence, last_probs = \
                                self.visualizer.get_activation_map(face_tensor)

                        elif self.viz_mode == 4:  # Spatial Attention
                            heatmap, pred_idx, confidence, last_probs = \
                                self.visualizer.get_spatial_attention(face_tensor)

                        elif self.viz_mode == 5:  # Occlusion (cached for performance)
                            occlusion_frame_count += 1
                            if occlusion_cache is None or occlusion_frame_count >= 15:
                                heatmap, pred_idx, confidence, last_probs = \
                                    self.visualizer.get_occlusion_sensitivity(face_tensor)
                                occlusion_cache = (heatmap, pred_idx, confidence, last_probs)
                                occlusion_frame_count = 0
                            else:
                                heatmap, pred_idx, confidence, last_probs = occlusion_cache

                        last_emotion = EMOTION_LABELS[pred_idx]
                        last_confidence = confidence

                        # Apply visualization
                        if heatmap is not None and self.viz_mode > 1:
                            face_with_viz = self.apply_heatmap(face_crop, heatmap)
                            display_frame[y1:y2, x1:x2] = face_with_viz
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                        else:
                            color = EMOTION_COLORS.get(last_emotion, (255, 255, 255))
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)

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
                elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
                    self.viz_mode = int(chr(key))
                    occlusion_cache = None  # Reset cache when changing modes
                    print(f"Mode: {VIZ_MODES[self.viz_mode]}")
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
