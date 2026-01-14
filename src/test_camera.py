"""
Simple camera test script to diagnose webcam issues.
Run this to verify your camera is working before running the full webcam emotion recognition.
"""

import cv2
import sys

def test_camera(camera_id=0):
    """Test if camera can be accessed and read frames."""
    print(f"Testing camera (ID={camera_id})...")
    print()

    # Try to open camera
    print("Step 1: Opening camera...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print("✗ FAILED: Could not open camera")
        print("\nPossible solutions:")
        print("  1. Check if camera is connected")
        print("  2. Try a different camera ID (use --camera 1, 2, etc.)")
        print("  3. On macOS: System Preferences → Security & Privacy → Camera")
        print("     Make sure Terminal/iTerm/Python has camera access")
        print("  4. Close other applications that might be using the camera")
        return False

    print("✓ Camera opened successfully")
    print()

    # Try to read frames
    print("Step 2: Reading frames (warming up camera)...")
    success_count = 0

    for i in range(10):
        ret, frame = cap.read()
        if ret:
            success_count += 1
            print(f"  Frame {i+1}: ✓ Success - Shape: {frame.shape}")
        else:
            print(f"  Frame {i+1}: ✗ Failed")

        # Small delay between reads
        import time
        time.sleep(0.1)

    print()

    if success_count == 0:
        print("✗ FAILED: Could not read any frames")
        print("\nThe camera opened but cannot provide video frames.")
        print("Possible issues:")
        print("  - Camera is being used by another application")
        print("  - Camera permissions not granted")
        print("  - Try closing FaceTime, Zoom, or other video apps")
        cap.release()
        return False

    if success_count < 10:
        print(f"⚠ WARNING: Only {success_count}/10 frames read successfully")
        print("Camera may be unstable")
    else:
        print(f"✓ All {success_count}/10 frames read successfully!")

    print()

    # Show a test window
    print("Step 3: Testing video display...")
    print("A window should open showing your camera feed.")
    print("Press 'q' to close the test window.")
    print()

    frame_count = 0
    while frame_count < 30:  # Show 30 frames
        ret, frame = cap.read()
        if ret:
            # Add text overlay
            cv2.putText(
                frame,
                "Camera Test - Press 'q' to quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )
            cv2.imshow('Camera Test', frame)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("="*60)
    print("✓ CAMERA TEST PASSED!")
    print("="*60)
    print()
    print("Your camera is working correctly.")
    print("You can now run: python3 webcam.py")
    print()

    return True


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test camera before running emotion recognition')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID (default: 0)')
    args = parser.parse_args()

    success = test_camera(args.camera)
    sys.exit(0 if success else 1)
