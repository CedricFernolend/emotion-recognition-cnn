"""Real-time emotion recognition from webcam."""

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


class EmotionWebcam:
    """Real-time emotion recognition from webcam."""

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

        print(f"Loading model from {model_path}")
        self.model = load_model(model_path).to(self.device)
        self.model.eval()
        print("Model loaded successfully")

        self.transform = get_transforms(augment=False)

        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")
        print("Face detector initialized")

        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot access webcam (camera_id={camera_id})")

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print("Warming up camera...")
        for _ in range(5):
            ret, _ = self.cap.read()
            if ret:
                break
            time.sleep(0.1)

        if not ret:
            self.cap.release()
            raise RuntimeError("Webcam opened but cannot read frames")

        print("Webcam initialized")

    def preprocess_face(self, face_bgr):
        """Convert BGR face crop to model input tensor."""
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        face_pil = Image.fromarray(face_rgb)
        face_tensor = self.transform(face_pil).unsqueeze(0)
        return face_tensor.to(self.device)

    def predict_emotion(self, face_tensor):
        """Run inference on preprocessed face."""
        with torch.no_grad():
            outputs = self.model(face_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted_idx = torch.max(probabilities, dim=1)
            emotion_label = EMOTION_LABELS[predicted_idx.item()]
            confidence_pct = confidence.item() * 100
        return emotion_label, confidence_pct, probabilities.cpu().numpy()[0]

    def draw_prediction(self, frame, x, y, w, h, emotion, confidence):
        """Draw bounding box and emotion label on frame."""
        color = self.EMOTION_COLORS.get(emotion, (255, 255, 255))
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        label = f"{emotion}: {confidence:.1f}%"
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        cv2.rectangle(frame, (x, y - text_height - 10), (x + text_width, y), color, -1)
        cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def draw_emotion_probabilities(self, frame, probabilities):
        """Draw all emotion probabilities in the bottom left corner."""
        x_start = 10
        y_start = frame.shape[0] - 150

        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start - 5, y_start - 25), (x_start + 200, y_start + 125), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        cv2.putText(frame, "Emotion Probabilities:", (x_start, y_start - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        for i, (emotion, prob) in enumerate(zip(EMOTION_LABELS, probabilities)):
            y_pos = y_start + 20 + i * 20
            text = f"{emotion:12s}: {prob*100:5.1f}%"
            color = self.EMOTION_COLORS.get(emotion, (255, 255, 255))
            cv2.putText(frame, text, (x_start, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def detect_faces(self, frame_gray):
        """Detect faces in grayscale frame using Haar Cascades."""
        return self.face_cascade.detectMultiScale(
            frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

    def run(self, show_fps=True, confidence_threshold=0.0):
        """Main webcam loop for real-time emotion recognition."""
        print("\nStarting webcam emotion recognition...")
        print("Controls: 'q' to quit, 's' to save frame\n")

        frame_count = 0
        fps = 0
        prev_time = time.time()
        consecutive_failures = 0
        latest_probabilities = None

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= 10:
                        print("Failed to grab frames. Exiting.")
                        break
                    time.sleep(0.1)
                    continue

                consecutive_failures = 0
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = self.detect_faces(gray)

                for (x, y, w, h) in faces:
                    padding = int(0.1 * max(w, h))
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    y2 = min(frame.shape[0], y + h + padding)

                    face_crop = frame[y1:y2, x1:x2]
                    if face_crop.shape[0] < 30 or face_crop.shape[1] < 30:
                        continue

                    try:
                        face_tensor = self.preprocess_face(face_crop)
                        emotion, confidence, probabilities = self.predict_emotion(face_tensor)
                        latest_probabilities = probabilities

                        if confidence >= confidence_threshold:
                            self.draw_prediction(frame, x, y, w, h, emotion, confidence)
                    except Exception as e:
                        print(f"Error processing face: {e}")

                if latest_probabilities is not None:
                    self.draw_emotion_probabilities(frame, latest_probabilities)

                if show_fps:
                    frame_count += 1
                    curr_time = time.time()
                    if curr_time - prev_time >= 1.0:
                        fps = frame_count / (curr_time - prev_time)
                        frame_count = 0
                        prev_time = curr_time

                    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.imshow('Emotion Recognition', frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("\nQuitting...")
                    break
                elif key == ord('s'):
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_dir = os.path.join(RESULTS_PATH, "visualizations")
                    os.makedirs(save_dir, exist_ok=True)
                    filename = os.path.join(save_dir, f"webcam_{timestamp}.png")
                    cv2.imwrite(filename, frame)
                    print(f"Saved frame to {filename}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.cleanup()

    def cleanup(self):
        """Release resources and close windows."""
        print("Releasing resources...")
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Cleanup complete")


def main():
    """Command-line interface for webcam emotion recognition."""
    parser = argparse.ArgumentParser(description='Real-time emotion recognition from webcam')
    parser.add_argument('--model', type=str, default=MODEL_SAVE_PATH, help='Path to trained model')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--threshold', type=float, default=0.0, help='Confidence threshold (0-100)')
    parser.add_argument('--no-fps', action='store_true', help='Hide FPS counter')

    args = parser.parse_args()

    if not 0 <= args.threshold <= 100:
        parser.error("Threshold must be between 0 and 100")

    try:
        webcam = EmotionWebcam(model_path=args.model, camera_id=args.camera)
        webcam.run(show_fps=not args.no_fps, confidence_threshold=args.threshold)
    except FileNotFoundError as e:
        print(f"\nError: Model file not found - {e}")
        print("Please train a model first by running: python train.py")
        return 1
    except RuntimeError as e:
        print(f"\nError: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
