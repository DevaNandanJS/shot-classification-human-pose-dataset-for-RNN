"""
Detailed Report on annotator.py

**1. Purpose:**

This script serves as a manual annotation tool for creating ground truth data for a 7-class
tennis shot classifier. Its primary function is to allow a human annotator to watch a video
and label shots for both right-handed and left-handed players by pressing specific keys.
The output is a CSV file that maps frame numbers to their corresponding shot labels
(e.g., `r_forehand`, `l_serve`), which is essential for training and evaluating the RNN model.

**2. Tech and Components:**

*   **OpenCV (cv2):** This is the core library used for all video processing tasks. It handles
    reading the video file, decoding it frame by frame, displaying frames in a window, and
    overlaying text annotations onto the video for user feedback.
*   **Pandas:** This library is used for data manipulation and storage. It takes the list of
    collected annotations and structures it into a DataFrame, which is then easily exported
    to a CSV file.
*   **pathlib:** A standard Python library used for object-oriented filesystem paths. It is
    used here to safely construct the output CSV filename based on the input video's name.

**3. Logic and Data Flow:**

1.  **Configuration:** The script is configured by hardcoding the `video` path variable at the
    top of the `if __name__ == "__main__":` block. The user must set this path to the
    video file they wish to annotate.

2.  **Initialization:**
    *   A `cv2.VideoCapture` object is created to open and read the specified video file.
    *   An empty list, `annotations`, is initialized to store annotation data (shot type and
      frame ID) as dictionaries.
    *   User instructions are printed to the console, explaining the key mappings for each shot.

3.  **Main Annotation Loop:**
    *   The script enters a `while cap.isOpened()` loop to process the video frame by frame.
    *   `cap.read()` retrieves the current frame.
    *   `cv2.imshow("Frame", frame)` displays the video in a window.

4.  **Keyboard Input Handling:**
    *   `cv2.waitKey(delay)` pauses execution and waits for a keypress, making the tool
      interactive. This is the core of the annotation logic.
    *   **Playback Control:** The user can press `+` to increase the delay (slow down) or `-`
      to decrease it (speed up).
    *   **Pause/Resume:** Pressing the `SPACE` bar will pause the video. Pressing it again
      will resume playback.
    *   **7-Class Annotation Keys:**
        -   **Right-Handed Shots:**
            -   `d`: Appends `{"Shot": "r_forehand", "FrameId": FRAME_ID}`.
            -   `a`: Appends `{"Shot": "r_backhand", "FrameId": FRAME_ID}`.
            -   `s`: Appends `{"Shot": "r_serve", "FrameId": FRAME_ID}`.
        -   **Left-Handed Shots:**
            -   `j`: Appends `{"Shot": "l_forehand", "FrameId": FRAME_ID}`.
            -   `l`: Appends `{"Shot": "l_backhand", "FrameId": FRAME_ID}`.
            -   `k`: Appends `{"Shot": "l_serve", "FrameId": FRAME_ID}`.

5.  **Visual Feedback:**
    *   When a valid shot key is pressed, the script stores the shot name (e.g., "R_FOREHAND")
      in the `last_shot_text` variable.
    *   For the next 20 frames, `cv2.putText()` overlays this text onto the video, giving the
      annotator immediate visual confirmation of their action.

6.  **Termination:** The loop ends when the video finishes or the user presses the `ESC` key.

7.  **Data Persistence:**
    *   After the loop, if the `annotations` list contains data, it is converted into a
      Pandas DataFrame.
    *   The output directory (`C:\RNN dataset creation\annotation`) is created if it doesn't exist.
    *   The DataFrame is saved as a CSV file with a descriptive name,
      `annotation_<original_video_name>.csv`, in the specified output directory.

**4. How to Use It:**

1.  **Set Video Path:** Open the `annotator.py` script and find the line `video = r"..."`.
    Change the path to the absolute path of the tennis video you want to annotate.
2.  **Run the Script:** Execute the script from your terminal: `python RNN/annotator.py`
3.  **Annotate the Video:** A window will appear playing the video. When a player hits the
    ball, press the corresponding key:
    *   Right-Handed Forehand: `d`
    *   Right-Handed Backhand: `a`
    *   Right-Handed Serve:   `s`
    *   Left-Handed Forehand:  `j`
    *   Left-Handed Backhand:  `l`
    *   Left-Handed Serve:     `k`
4.  **Control Playback:** Use the `+` and `-` keys to adjust the video speed. Use the
    `SPACE` bar to pause and resume.
5.  **Exit and Save:** Press the `ESC` key to stop the video. The script will automatically
    save your annotations into a CSV file in the `C:\RNN dataset creation\annotation` directory.
"""

from pathlib import Path
import pandas as pd
import cv2


def format_time(frame_id, fps):
    """Formats frame count into MM:SS string."""
    if fps == 0:
        return "00:00"
    seconds = int(frame_id / fps)
    minutes = seconds // 60
    seconds %= 60
    return f"{minutes:02d}:{seconds:02d}"


if __name__ == "__main__":
    video = r"C:\RNN dataset creation\input\right-handed\r2.mp4"
    scale = 1.0

    cap = cv2.VideoCapture(video)

    # Get video properties for progress display
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps == 0:
        fps = 30  # Default to 30 if FPS is not available
    total_duration_str = format_time(total_frames, fps)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error opening video stream or file")

    annotations = []
    FRAME_ID = 0
    last_shot_text = ""
    display_text_for_frames = 0  # Frame counter to display the text

    print("Starting annotation...")
    print("RIGHT-HANDED: 'd' for forehand, 'a' for backhand, 's' for serve.")
    print("LEFT-HANDED:  'j' for forehand, 'l' for backhand, 'k' for serve.")
    print("Press '+' to slow down, '-' to speed up. Press SPACE to pause/resume. Press ESC to exit.")

    delay = 120  # Initial delay for playback speed

    # Read until video is completed
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Resize frame if a custom scale is provided
        if scale != 1.0:
            width = int(frame.shape[1] * scale)
            height = int(frame.shape[0] * scale)
            dim = (width, height)
            frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

        # --- Progress Bar and Time Display ---
        (h, w) = frame.shape[:2]
        progress = FRAME_ID / total_frames if total_frames > 0 else 0

        # Draw background of progress bar
        cv2.rectangle(frame, (0, 0), (w, 20), (50, 50, 50), -1)

        # Draw foreground of progress bar
        cv2.rectangle(frame, (0, 0), (int(w * progress), 20), (0, 165, 255), -1)

        # Draw time text
        current_time_str = format_time(FRAME_ID, fps)
        time_text = f"{current_time_str} / {total_duration_str}"
        cv2.putText(
            frame, time_text, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1
        )

        # If the display counter is active, put text on the frame
        if display_text_for_frames > 0:
            cv2.putText(
                frame,
                last_shot_text,
                (50, 80),  # Position (x, y from top-left)
                cv2.FONT_HERSHEY_SIMPLEX,
                2,  # Font scale
                (50, 255, 50),  # BGR Color (light green)
                3,
            )  # Thickness
            display_text_for_frames -= 1

        cv2.imshow("Frame", frame)
        # Use a variable delay for playback, waitKeyEx is used for arrow keys
        k = cv2.waitKeyEx(delay)

        # --- SEEK FORWARD/BACKWARD ---
        # Key codes for arrow keys can vary: 2424832=left, 2555904=right
        if k == 2555904 or k == 2424832:
            seek_frames = int(1 * fps) * (1 if k == 2555904 else -1)
            direction = "forward" if seek_frames > 0 else "backward"
            
            new_frame_id = FRAME_ID + seek_frames
            # Clamp the new frame ID to be within video bounds
            new_frame_id = max(0, min(new_frame_id, total_frames - 1))
            
            if new_frame_id != FRAME_ID:
                FRAME_ID = new_frame_id
                cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_ID)
                print(f"Skipping {direction} 5s to frame {FRAME_ID}")
                continue # Restart loop to read the new frame

        # --- PAUSE/RESUME LOGIC ---
        if k == ord(' '):
            # Display "PAUSED" text and wait for another spacebar press to resume
            (h, w) = frame.shape[:2]
            overlay = frame.copy()
            cv2.putText(overlay, "PAUSED",
                        (w // 2 - 70, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2, (50, 50, 255), 3)
            cv2.imshow("Frame", overlay)
            
            while True:
                pause_k = cv2.waitKey(0)
                if pause_k == ord(' '): # Resume
                    k = -1 # Reset k so it doesn't trigger other actions
                    break
                if pause_k == 27: # Exit from pause
                    k = 27
                    break

        shot_to_add = None
        if k == ord("+"):
            delay += 25
            print(f"Slowing down. New delay: {delay}")
        elif k == ord("-"):
            delay = max(1, delay - 25)
            print(f"Speeding up. New delay: {delay}")

        # Right-handed shots
        elif k == ord("d"):
            shot_to_add = "r_forehand"
        elif k == ord("a"):
            shot_to_add = "r_backhand"
        elif k == ord("s"):
            shot_to_add = "r_serve"

        # Left-handed shots
        elif k == ord("j"):
            shot_to_add = "l_forehand"
        elif k == ord("l"):
            shot_to_add = "l_backhand"
        elif k == ord("k"):
            shot_to_add = "l_serve"

        # If a shot key was pressed, add the annotation
        if shot_to_add:
            annotations.append({"Shot": shot_to_add, "FrameId": FRAME_ID})
            print(f"Frame {FRAME_ID}: Add {shot_to_add}")
            last_shot_text = shot_to_add.upper()
            display_text_for_frames = 20  # Display text for the next 20 frames

        # Press ESC on keyboard to exit
        if k == 27:
            break

        FRAME_ID += 1

    # After the loop, save the annotations to a CSV file if any were made
    if annotations:
        df = pd.DataFrame.from_records(annotations)
        output_dir = Path(r"C:\RNN dataset creation\annotation")
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / f"annotation_{Path(video).stem}.csv"
        df.to_csv(out_file, index=False)
        print(
            f"\nAnnotation file with {len(annotations)} shots was written to {out_file}"
        )
    else:
        print("\nNo annotations were made. No file was saved.")