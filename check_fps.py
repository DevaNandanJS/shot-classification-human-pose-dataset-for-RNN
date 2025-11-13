import cv2
from pathlib import Path

def check_video_fps():
    """
    Checks and prints the frames per second (FPS) of a given video file.
    """
    # Hardcoded video path
    video_path = r"C:\RNN dataset creation\input\left-handed\f5.mp4"

    if not Path(video_path).exists():
        print(f"Error: Video file not found at {video_path}")
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"The FPS of the video '{video_path}' is: {fps:.2f}")

    cap.release()

if __name__ == "__main__":
    check_video_fps()