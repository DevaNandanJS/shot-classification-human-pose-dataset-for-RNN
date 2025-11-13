"""
This script visualizes human pose data from a CSV file as an animated GIF.

It reads a CSV file containing normalized keypoint coordinates for each frame of a video,
and then generates an animation of the pose, optionally displaying joint angles and
saving the output as a GIF.

Key Features:
- Visualizes pose data from CSV files.
- Displays joint angles on the animation.
- Exports the animation as a GIF.
- Allows for easy customization of visualization parameters (size, colors, etc.).
"""

import argparse
import math
from pathlib import Path

import cv2
import imageio
import numpy as np
import pandas as pd

# --- Visualization Parameters ---

# Canvas dimensions
HEIGHT = 1000
WIDTH = 1000

# Scale factor for the pose. A larger value will make the pose appear larger.
SCALE_FACTOR = 400

# Radius of the keypoint circles.
RADIUS = 8

# Thickness of the edge lines.
THICKNESS = 4

# Font scale for the angle text.
FONT_SCALE = 1.0

# Thickness of the angle text.
FONT_THICKNESS = 2

# --- Keypoint and Edge Definitions ---

KEYPOINT_DICT = {
    "nose": 0,
    "left_eye": 1,
    "right_eye": 2,
    "left_ear": 3,
    "right_ear": 4,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
}

EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("right_shoulder", "right_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
]

# --- Drawing Functions ---

def draw_keypoints(frame, keypoints):
    """Draws keypoints on the frame."""
    for keypoint in keypoints:
        if keypoint[2] > 0:  # Check if the keypoint is visible
            cv2.circle(
                frame,
                (int(keypoint[0]), int(keypoint[1])),
                RADIUS,
                (0, 255, 0),
                -1,
            )

def draw_edges(frame, keypoints):
    """Draws edges between keypoints."""
    for edge in EDGES:
        p1_name, p2_name = edge
        p1_idx = KEYPOINT_DICT[p1_name]
        p2_idx = KEYPOINT_DICT[p2_name]

        p1 = keypoints[p1_idx]
        p2 = keypoints[p2_idx]

        if p1[2] > 0 and p2[2] > 0:  # Check if both keypoints are visible
            cv2.line(
                frame,
                (int(p1[0]), int(p1[1])),
                (int(p2[0]), int(p2[1])),
                (255, 255, 0),
                THICKNESS,
            )

def calculate_angle(p1, p2, p3):
    """Calculates the angle between three points."""
    if p1[2] == 0 or p2[2] == 0 or p3[2] == 0:
        return 0

    v1 = (p1[0] - p2[0], p1[1] - p2[1])
    v2 = (p3[0] - p2[0], p3[1] - p2[1])

    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
    mag_v1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
    mag_v2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

    if mag_v1 == 0 or mag_v2 == 0:
        return 0

    cosine_angle = dot_product / (mag_v1 * mag_v2)
    angle = math.acos(max(min(cosine_angle, 1), -1))
    return math.degrees(angle)

def display_angles(frame, keypoints):
    """Displays joint angles on the frame."""
    angles_to_display = [
        ("left_shoulder", "left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow", "right_wrist"),
        ("left_hip", "left_knee", "left_ankle"),
        ("right_hip", "right_knee", "right_ankle"),
        ("left_shoulder", "left_hip", "left_knee"),
        ("right_shoulder", "right_hip", "right_knee"),
    ]

    for p1_name, p2_name, p3_name in angles_to_display:
        p1 = keypoints[KEYPOINT_DICT[p1_name]]
        p2 = keypoints[KEYPOINT_DICT[p2_name]]
        p3 = keypoints[KEYPOINT_DICT[p3_name]]

        angle = calculate_angle(p1, p2, p3)

        cv2.putText(
            frame,
            f"{angle:.1f}",
            (int(p2[0]), int(p2[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            (255, 255, 255),
            FONT_THICKNESS,
        )

# --- Main Function ---

def main():
    """
    Main function to parse arguments, read data, and generate the visualization.
    """
    parser = argparse.ArgumentParser(
        description="Visualize human pose data from a CSV file as an animated GIF."
    )
    parser.add_argument(
        "input_file", type=Path, help="Path to the input CSV file."
    )
    parser.add_argument(
        "--gif", type=Path, help="Path to save the output GIF file."
    )
    parser.add_argument(
        "--show-angles", action="store_true", help="Display joint angles."
    )
    args = parser.parse_args()

    if not args.input_file.is_file():
        print(f"Error: Input file not found at {args.input_file}")
        return

    try:
        df = pd.read_csv(args.input_file)
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    frames = []
    for _, row in df.iterrows():
        frame = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
        keypoints = []
        for keypoint_name in KEYPOINT_DICT:
            x_col = f"{keypoint_name}_x"
            y_col = f"{keypoint_name}_y"
            vis_col = f"{keypoint_name}_v"

            if x_col in row and y_col in row:
                x = row[x_col] * SCALE_FACTOR + WIDTH / 2
                y = row[y_col] * SCALE_FACTOR + HEIGHT / 2
                visibility = row.get(vis_col, 1)  # Assume visible if not specified
                keypoints.append((x, y, visibility))
            else:
                keypoints.append((0, 0, 0))  # Not present in the data

        draw_edges(frame, keypoints)
        draw_keypoints(frame, keypoints)

        if args.show_angles:
            display_angles(frame, keypoints)

        frames.append(frame)

    if args.gif:
        print(f"Saving GIF to {args.gif}...")
        imageio.mimsave(args.gif, frames, fps=30)
        print("GIF saved successfully.")

    # Display the animation in a window
    for frame in frames:
        cv2.imshow("Pose Visualization", frame)
        if cv2.waitKey(30) & 0xFF == 27:  # Exit on ESC
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()