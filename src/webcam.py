"""
Real-time emotion recognition from webcam.

This module provides real-time emotion recognition using a webcam feed.
It detects faces using OpenCV Haar Cascades, processes them through the
trained emotion recognition model, and displays predictions on the video feed.

Usage:
    python webcam.py [--model PATH] [--camera ID] [--threshold N] [--no-fps]
"""

import cv2
import torch
import numpy as np
from PIL import Image
import argparse
import time
import datetime
import os

from config import EMOTION_LABELS, MODEL_SAVE_PATH, RESULTS_PATH
from model import load_model
from data import get_transforms


class EmotionWebcam:
    """
    Real-time emotion recognition from webcam.

    This class handles webcam capture, face detection, emotion prediction,
    and visualization of results.

    Attributes:
        device: torch device (cuda or cpu)
        model: loaded emotion recognition model
        transform: image preprocessing transforms
        face_cascade: OpenCV Haar Cascade face detector
        cap: OpenCV VideoCapture object
        colors: emotion-specific colors for visualization
    """

    def __init__(self, model_path=MODEL_SAVE_PATH, camera_id=0):
        """
        Initialize the emotion webcam system.

        Args:
            model_path: Path to trained model weights
            camera_id: Camera device ID (default: 0)

        Raises:
            RuntimeError: If webcam cannot be accessed
            FileNotFoundError: If model file not found
        """
        print("Initializing emotion recognition system...")

        # Device selection
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        # Load model in eval mode
        print(f"Loading model from {model_path}")
        self.model = load_model(model_path).to(self.device)
        self.model.eval()
        print("Model loaded successfully")

        # Get preprocessing transforms (no augmentation for inference)
        self.transform = get_transforms(augment=False)

        # Initialize face detector
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")
        print("Face detector initialized")

        # Initialize webcam
        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot access webcam (camera_id={camera_id}). Check permissions.")

        # Set resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # Warm up camera - read a few frames to ensure it's ready (macOS fix)
        print("Warming up camera...")
        for i in range(5):
            ret, _ = self.cap.read()
            if ret:
                break
            import time
            time.sleep(0.1)

        if not ret:
            self.cap.release()
            raise RuntimeError(f"Webcam opened but cannot read frames. Camera may be in use by another application.")

        print("Webcam initialized")

        # Colors for visualization (BGR format for OpenCV)
        self.colors = {
            'happiness': (0, 255, 0),      # Green
            'surprise': (255, 255, 0),     # Cyan
            'sadness': (255, 0, 0),        # Blue
            'anger': (0, 0, 255),          # Red
            'disgust': (0, 128, 255),      # Orange
            'fear': (128, 0, 128)          # Purple
        }

    def preprocess_face(self, face_bgr):
        """
        Convert BGR face crop to model input tensor.

        Critical: OpenCV uses BGR format, but the model was trained on RGB.
        This function handles the conversion and applies the same preprocessing
        used during training.

        Args:
            face_bgr: numpy array (H, W, 3) in BGR format from OpenCV

        Returns:
            tensor: (1, 3, 64, 64) normalized to [-1, 1] range
        """
        # Convert BGR to RGB (OpenCV uses BGR, PIL uses RGB)
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image for torchvision transforms compatibility
        face_pil = Image.fromarray(face_rgb)

        # Apply transforms: Resize(64x64), ToTensor, Normalize to [-1, 1]
        face_tensor = self.transform(face_pil)

        # Add batch dimension: (3, 64, 64) -> (1, 3, 64, 64)
        face_tensor = face_tensor.unsqueeze(0)

        return face_tensor.to(self.device)

    def predict_emotion(self, face_tensor):
        """
        Run inference on preprocessed face.

        Args:
            face_tensor: (1, 3, 64, 64) tensor, normalized to [-1, 1]

        Returns:
            tuple: (emotion_label, confidence_percentage, probabilities)
                - emotion_label: string (e.g., 'happiness')
                - confidence_percentage: float (0-100)
                - probabilities: numpy array of all class probabilities
        """
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = self.model(face_tensor)

            # Get probabilities using softmax
            probabilities = torch.softmax(outputs, dim=1)

            # Get prediction and confidence
            confidence, predicted_idx = torch.max(probabilities, dim=1)

            emotion_label = EMOTION_LABELS[predicted_idx.item()]
            confidence_pct = confidence.item() * 100

        return emotion_label, confidence_pct, probabilities.cpu().numpy()[0]

    def draw_prediction(self, frame, x, y, w, h, emotion, confidence):
        """
        Draw bounding box and emotion label on frame.

        Args:
            frame: OpenCV frame (BGR format)
            x, y, w, h: Face bounding box coordinates
            emotion: Predicted emotion label
            confidence: Confidence percentage (0-100)
        """
        # Get color for this emotion
        color = self.colors.get(emotion, (255, 255, 255))

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # Prepare label text
        label = f"{emotion}: {confidence:.1f}%"

        # Get text size for background rectangle
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        # Draw background rectangle for text (filled)
        cv2.rectangle(
            frame,
            (x, y - text_height - 10),
            (x + text_width, y),
            color,
            -1  # Filled rectangle
        )

        # Draw text in white
        cv2.putText(
            frame,
            label,
            (x, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),  # White text
            2
        )

    def draw_emotion_probabilities(self, frame, probabilities):
        """
        Draw all emotion probabilities in the bottom left corner.

        Args:
            frame: OpenCV frame (BGR format)
            probabilities: numpy array of probabilities for all 6 emotions
        """
        # Starting position (bottom left)
        x_start = 10
        y_start = frame.shape[0] - 150  # 150 pixels from bottom

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_start - 5, y_start - 25), (x_start + 200, y_start + 125), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw title
        cv2.putText(
            frame,
            "Emotion Probabilities:",
            (x_start, y_start - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )

        # Draw each emotion with its probability
        for i, (emotion, prob) in enumerate(zip(EMOTION_LABELS, probabilities)):
            y_pos = y_start + 20 + i * 20
            text = f"{emotion:12s}: {prob*100:5.1f}%"
            color = self.colors.get(emotion, (255, 255, 255))

            cv2.putText(
                frame,
                text,
                (x_start, y_pos),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1
            )

    def detect_faces(self, frame_gray):
        """
        Detect faces in grayscale frame using Haar Cascades.

        Args:
            frame_gray: Grayscale OpenCV frame

        Returns:
            faces: List of (x, y, w, h) tuples representing face bounding boxes
        """
        faces = self.face_cascade.detectMultiScale(
            frame_gray,
            scaleFactor=1.1,      # Scale factor between successive scans
            minNeighbors=5,       # Min neighbors for valid detection (reduces false positives)
            minSize=(30, 30),     # Minimum face size in pixels
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        return faces

    def run(self, show_fps=True, confidence_threshold=0.0):
        """
        Main webcam loop for real-time emotion recognition.

        Args:
            show_fps: Display FPS counter on frame
            confidence_threshold: Minimum confidence (0-100) to show prediction
        """
        print("\n" + "="*50)
        print("Starting webcam emotion recognition...")
        print("="*50)
        print("Controls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save current frame")
        print("="*50 + "\n")

        frame_count = 0
        fps = 0
        prev_time = time.time()
        consecutive_failures = 0
        max_failures = 10
        latest_probabilities = None  # Store latest emotion probabilities

        try:
            while True:
                # Capture frame
                ret, frame = self.cap.read()
                if not ret:
                    consecutive_failures += 1
                    if consecutive_failures >= max_failures:
                        print(f"\nFailed to grab frame {consecutive_failures} times. Exiting.")
                        print("Possible issues:")
                        print("  - Camera is being used by another application")
                        print("  - Camera permissions not granted")
                        print("  - Hardware malfunction")
                        break
                    time.sleep(0.1)
                    continue

                # Reset failure counter on success
                consecutive_failures = 0

                # Convert to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect faces
                faces = self.detect_faces(gray)

                # Process each detected face
                for (x, y, w, h) in faces:
                    # Extract face region with padding (10% on each side)
                    padding = int(0.1 * max(w, h))
                    x1 = max(0, x - padding)
                    y1 = max(0, y - padding)
                    x2 = min(frame.shape[1], x + w + padding)
                    y2 = min(frame.shape[0], y + h + padding)

                    face_crop = frame[y1:y2, x1:x2]

                    # Skip if face crop is too small
                    if face_crop.shape[0] < 30 or face_crop.shape[1] < 30:
                        continue

                    try:
                        # Preprocess and predict
                        face_tensor = self.preprocess_face(face_crop)
                        emotion, confidence, probabilities = self.predict_emotion(face_tensor)

                        # Store latest probabilities for display
                        latest_probabilities = probabilities

                        # Draw if confidence above threshold
                        if confidence >= confidence_threshold:
                            self.draw_prediction(frame, x, y, w, h, emotion, confidence)

                    except Exception as e:
                        print(f"Error processing face: {e}")
                        continue

                # Draw emotion probabilities in bottom left (if we have any)
                if latest_probabilities is not None:
                    self.draw_emotion_probabilities(frame, latest_probabilities)

                # Calculate and display FPS
                if show_fps:
                    frame_count += 1
                    curr_time = time.time()
                    if curr_time - prev_time >= 1.0:
                        fps = frame_count / (curr_time - prev_time)
                        frame_count = 0
                        prev_time = curr_time

                    # Draw FPS counter
                    cv2.putText(
                        frame,
                        f"FPS: {fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )

                # Display frame
                cv2.imshow('Emotion Recognition', frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\nQuitting...")
                    break

                elif key == ord('s'):
                    # Save current frame
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_dir = os.path.join(RESULTS_PATH, "visualizations")
                    os.makedirs(save_dir, exist_ok=True)
                    filename = os.path.join(save_dir, f"webcam_{timestamp}.png")
                    cv2.imwrite(filename, frame)
                    print(f"Saved frame to {filename}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        except Exception as e:
            print(f"\nError in main loop: {e}")
            raise

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
    parser = argparse.ArgumentParser(
        description='Real-time emotion recognition from webcam',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python webcam.py
  python webcam.py --model results/models/best_model.pth
  python webcam.py --camera 1 --threshold 50
  python webcam.py --no-fps
        """
    )

    parser.add_argument(
        '--model',
        type=str,
        default=MODEL_SAVE_PATH,
        help=f'Path to trained model (default: {MODEL_SAVE_PATH})'
    )

    parser.add_argument(
        '--camera',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )

    parser.add_argument(
        '--threshold',
        type=float,
        default=0.0,
        help='Confidence threshold 0-100 (default: 0 = show all predictions)'
    )

    parser.add_argument(
        '--no-fps',
        action='store_true',
        help='Hide FPS counter'
    )

    args = parser.parse_args()

    # Validate threshold
    if not 0 <= args.threshold <= 100:
        parser.error("Threshold must be between 0 and 100")

    try:
        # Create and run webcam
        webcam = EmotionWebcam(model_path=args.model, camera_id=args.camera)
        webcam.run(show_fps=not args.no_fps, confidence_threshold=args.threshold)

    except FileNotFoundError as e:
        print(f"\nError: Model file not found - {e}")
        print(f"Please train a model first by running: python train.py")
        return 1

    except RuntimeError as e:
        print(f"\nError: {e}")
        return 1

    except Exception as e:
        print(f"\nUnexpected error: {e}")
        raise

    return 0


if __name__ == "__main__":
    exit(main())
