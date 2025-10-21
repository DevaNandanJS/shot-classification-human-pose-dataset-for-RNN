"""
This script will produce shot annotation on a tennis video.
It will output a csv file containing frame id and shot name by pressing your key board keys.
'd' to mark a shot as FOREHAND
'a' to mark a shot as BACKHAND
's' to mark a shot as SERVE
We advise you to hit the key when the player hits the ball.
"""

from argparse import ArgumentParser
from pathlib import Path
import pandas as pd
import cv2


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Annotate a video and write a csv file containing tennis shots"
    )
    parser.add_argument("video")
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Factor to scale the display window (e.g., 0.5 for 50%% size).",
    )
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    annotations = []
    FRAME_ID = 0
    last_shot_text = ""
    display_text_for_frames = 0  # Frame counter to display the text

    print("Starting annotation... Press 'd' for forehand, 'a' for backhand, 's' for serve. Press ESC to exit.")

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame if a custom scale is provided
        if args.scale != 1.0:
            width = int(frame.shape[1] * args.scale)
            height = int(frame.shape[0] * args.scale)
            dim = (width, height)
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # If the display counter is active, put text on the frame
        if display_text_for_frames > 0:
            cv2.putText(frame, last_shot_text,
                        (50, 80),  # Position (x, y from top-left)
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,  # Font scale
                        (50, 255, 50),  # BGR Color (light green)
                        3)  # Thickness
            display_text_for_frames -= 1

        cv2.imshow("Frame", frame)
        # Use a fixed, slower delay for playback
        k = cv2.waitKey(100)

        if k == ord('d'):  # forehand
            annotations.append({"Shot": "forehand", "FrameId": FRAME_ID})
            print(f"Frame {FRAME_ID}: Add forehand")
            last_shot_text = "Forehand"
            display_text_for_frames = 20  # Display text for the next 20 frames
        elif k == ord('a'):  # backhand
            annotations.append({"Shot": "backhand", "FrameId": FRAME_ID})
            print(f"Frame {FRAME_ID}: Add backhand")
            last_shot_text = "Backhand"
            display_text_for_frames = 20
        elif k == ord('s'):  # serve
            annotations.append({"Shot": "serve", "FrameId": FRAME_ID})
            print(f"Frame {FRAME_ID}: Add serve")
            last_shot_text = "Serve"
            display_text_for_frames = 20

        # Press ESC on keyboard to exit
        if k == 27:
            break

        FRAME_ID += 1

    # After the loop, save the annotations to a CSV file if any were made
    if annotations:
        df = pd.DataFrame.from_records(annotations)
        out_file = f"annotation_{Path(args.video).stem}.csv"
        df.to_csv(out_file, index=False)
        print(f"\nAnnotation file with {len(annotations)} shots was written to {out_file}")
    else:
        print("\nNo annotations were made. No file was saved.")