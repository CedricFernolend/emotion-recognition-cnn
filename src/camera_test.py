"""Camera test utility to verify webcam is working before running emotion recognition."""

import sys
import time
import argparse
import cv2


def test_camera(camera_id=0):
    """Test if camera can be accessed and read frames."""
    print(f"Testing camera (ID={camera_id})...\n")

    print("Step 1: Opening camera...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("FAILED: Could not open camera")
        print("\nPossible solutions:")
        print("  1. Check if camera is connected")
        print("  2. Try a different camera ID (--camera 1, 2, etc.)")
        print("  3. On macOS: System Preferences > Security & Privacy > Camera")
        print("  4. Close other applications using the camera")
        return False

    print("Camera opened successfully\n")

    print("Step 2: Reading frames...")
    success_count = 0

    for i in range(10):
        ret, frame = cap.read()
        if ret:
            success_count += 1
            print(f"  Frame {i+1}: OK - Shape: {frame.shape}")
        else:
            print(f"  Frame {i+1}: Failed")
        time.sleep(0.1)

    print()

    if success_count == 0:
        print("FAILED: Could not read any frames")
        print("Try closing other video apps (FaceTime, Zoom, etc.)")
        cap.release()
        return False

    if success_count < 10:
        print(f"WARNING: Only {success_count}/10 frames read successfully")
    else:
        print(f"All {success_count}/10 frames read successfully")

    print("\nStep 3: Testing video display...")
    print("Press 'q' to close the test window.\n")

    frame_count = 0
    while frame_count < 30:
        ret, frame = cap.read()
        if ret:
            cv2.putText(frame, "Camera Test - Press 'q' to quit", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Camera Test', frame)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    print("\nCAMERA TEST PASSED")
    print("You can now run: python webcam.py\n")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test camera before running emotion recognition')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    args = parser.parse_args()

    success = test_camera(args.camera)
    sys.exit(0 if success else 1)
