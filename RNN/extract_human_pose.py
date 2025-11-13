"""
This script provides a robust solution for real-time human pose estimation in video streams.
It leverages the Ultralytics YOLO model for efficient and accurate keypoint detection and
OpenCV for video processing and visualization.

Working of the Code:
The core functionality is encapsulated within the `HumanPoseExtractor` class.
1.  **Initialization**: The `HumanPoseExtractor` is initialized with a path to a pre-trained YOLO pose estimation model (e.g., an ONNX model).
2.  **Pose Extraction**: The `extract` method takes a video frame as input, processes it using the YOLO model's `track` method to detect human figures and their keypoints, and stores the detected keypoints along with their confidence scores. The `track` method also provides object tracking IDs, allowing for consistent identification of individuals across frames.
3.  **Keypoint Discarding**: The `discard` method allows selective removal of specific keypoints (e.g., eyes, ears) by setting their confidence scores to zero. This is useful for focusing on particular body parts or reducing noise.
4.  **Result Drawing**: The `draw_results_frame` method overlays the detected keypoints and their connections (edges) onto the original video frame, providing a visual representation of the human pose.

Logic:
- The script utilizes a pre-trained YOLO model, specifically designed for pose estimation, to identify 17 keypoints on the human body.
- Keypoints are represented as `[y, x, confidence]` arrays, where `y` and `x` are pixel coordinates and `confidence` indicates the detection certainty.
- The `KEYPOINT_DICT` maps keypoint names to their corresponding indices, and `EDGES` defines the connections between keypoints to form a skeletal structure.
- Drawing functions (`draw_keypoints`, `draw_edges`) apply confidence thresholds to ensure only reliably detected keypoints and connections are visualized.

Technologies and Components Used:
-   **Python**: The primary programming language.
-   **OpenCV (`cv2`)**: Used for video file handling (reading frames, displaying output), and drawing operations (circles for keypoints, lines for edges).
-   **NumPy**: Essential for efficient numerical operations, particularly for handling keypoint arrays.
-   **Ultralytics YOLO**: The core library for state-of-the-art object detection and pose estimation. It provides the `YOLO` model class and its `track` method for real-time inference.
-   **Argparse**: Used for parsing command-line arguments, enabling users to specify input video files and custom model paths.

Data Flow:
1.  **Input**: The script takes a video file path and an optional YOLO model path as command-line arguments.
2.  **Video Stream**: `cv2.VideoCapture` reads the video frame by frame.
3.  **Frame Processing**: Each frame is fed into the `HumanPoseExtractor.extract()` method.
    -   The YOLO model performs inference on the frame, detecting human instances and their 17 keypoints.
    -   The extracted keypoints (pixel coordinates and confidence) for each detected person are stored.
4.  **Keypoint Filtering (Optional)**: If specified, `HumanPoseExtractor.discard()` modifies the confidence scores of certain keypoints.
5.  **Visualization**: `HumanPoseExtractor.draw_results_frame()` uses OpenCV to draw the filtered keypoints and their connections onto the current video frame.
6.  **Output**: The annotated frame is displayed in a window using `cv2.imshow()`.
7.  **Loop Termination**: The process continues until the video ends or the user presses the 'Esc' key.

Purpose:
The main purpose of this script is to:
-   Provide a ready-to-use tool for visualizing human pose estimations in videos.
-   Serve as a foundational component for more complex applications requiring human activity analysis, such as sports analytics, fitness tracking, or gesture recognition.
-   Offer a clear example of integrating YOLO-based pose estimation with OpenCV for video processing.

How to Use:
To run this script, execute it from the command line, providing the path to your video file.

Example:
```bash
python extract_human_pose.py "path/to/your/video.mp4"
```

You can also specify a custom YOLO model path using the `--model` argument:

Example with custom model:
```bash
python extract_human_pose.py "path/to/your/video.mp4" --model "path/to/your/custom_model.onnx"
```
Press 'Esc' to exit the video playback window.
"""

import cv2
import numpy as np
from ultralytics import YOLO
import argparse
import os

# A dictionary to map keypoint indices to their names
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# A list of tuples representing the edges to draw on the pose
EDGES = (
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'),
    ('right_shoulder', 'right_elbow'),
    ('left_elbow', 'left_wrist'),
    ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'),
    ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'),
    ('right_hip', 'right_knee'),
    ('left_knee', 'left_ankle'),
    ('right_knee', 'right_ankle')
)

class HumanPoseExtractor:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        # Change data structures to dictionaries keyed by tracking ID
        self.keypoints_pixels_frame = {}
        self.keypoints_normalized_frame = {}
        self.boxes_frame = {}
        self.confidences_frame = {}

    def extract(self, frame):
        results = self.model.track(frame, persist=True, verbose=False, conf=0.5, tracker="botsort.yaml")

        # Reset for the current frame
        self.keypoints_pixels_frame = {}
        self.keypoints_normalized_frame = {}
        self.boxes_frame = {}
        self.confidences_frame = {}

        if results and results[0].keypoints and results[0].boxes.id is not None:
            # Get all data from the results object
            tracking_ids = results[0].boxes.id.int().cpu().tolist()
            all_keypoints_xy = results[0].keypoints.xy.cpu().numpy()
            all_keypoints_xyn = results[0].keypoints.xyn.cpu().numpy()
            all_kpt_confidences = results[0].keypoints.conf.cpu().numpy()
            all_boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
            all_box_confidences = results[0].boxes.conf.cpu().numpy()

            for i, track_id in enumerate(tracking_ids):
                # Get data for the current person
                keypoints_xy = all_keypoints_xy[i]
                keypoints_xyn = all_keypoints_xyn[i]
                kpt_confidences = all_kpt_confidences[i]
                box_xyxy = all_boxes_xyxy[i]
                box_conf = all_box_confidences[i]

                person_keypoints_pixels = []
                person_keypoints_normalized = []
                for kp_xy, kp_xyn, conf in zip(keypoints_xy, keypoints_xyn, kpt_confidences):
                    person_keypoints_pixels.append([kp_xy[1], kp_xy[0], conf]) # y, x, conf
                    person_keypoints_normalized.append([kp_xyn[1], kp_xyn[0], conf]) # y, x, conf

                # Populate dictionaries with the tracking ID as the key
                self.keypoints_pixels_frame[track_id] = person_keypoints_pixels
                self.keypoints_normalized_frame[track_id] = person_keypoints_normalized
                self.boxes_frame[track_id] = box_xyxy
                self.confidences_frame[track_id] = box_conf

    def discard(self, list_of_keypoints):
        for track_id, person_keypoints in self.keypoints_pixels_frame.items():
            for keypoint_name in list_of_keypoints:
                keypoint_index = KEYPOINT_DICT[keypoint_name]
                if keypoint_index < len(person_keypoints):
                    person_keypoints[keypoint_index][2] = 0 # Set confidence to 0

    def draw_results_frame(self, frame, target_id=None, boxes=None):
        """
        Draws results. If target_id is provided, only draws for that ID with a bounding box.
        Otherwise, draws all detected poses.
        """
        if target_id is not None and target_id in self.keypoints_pixels_frame:
            # Draw only the target player
            keypoints = self.keypoints_pixels_frame[target_id]
            box = boxes.get(target_id)

            # Draw bounding box and ID for the target
            if box is not None:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID: {target_id}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Draw keypoints and edges for the target
            draw_edges(frame, keypoints, EDGES, 0.1)
            draw_keypoints(frame, keypoints, 0.1)
        elif target_id is None:
            # Fallback to drawing all if no target is specified
            for keypoints in self.keypoints_pixels_frame.values():
                draw_edges(frame, keypoints, EDGES, 0.1)
                draw_keypoints(frame, keypoints, 0.1)

def draw_keypoints(frame, keypoints, confidence_threshold):
    for kp in keypoints:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

def draw_edges(frame, keypoints, edges, confidence_threshold):
    for edge in edges:
        p1_name, p2_name = edge
        p1_idx = KEYPOINT_DICT[p1_name]
        p2_idx = KEYPOINT_DICT[p2_name]

        if p1_idx < len(keypoints) and p2_idx < len(keypoints):
            y1, x1, c1 = keypoints[p1_idx]
            y2, x2, c2 = keypoints[p2_idx]

            if c1 > confidence_threshold and c2 > confidence_threshold:
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Display human pose on a video")
    parser.add_argument("video")
    parser.add_argument("--model", default="C:\Prototype Ultra\model-weights\yolo11s-pose.onnx", help="Path to ONNX model")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Could not open video file.")
    else:
        human_pose_extractor = HumanPoseExtractor(args.model)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            human_pose_extractor.extract(frame)
            human_pose_extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])
            human_pose_extractor.draw_results_frame(frame)

            cv2.imshow("Frame", frame)

            k = cv2.waitKey(1)
            if k == 27:
                break

        cap.release()
        cv2.destroyAllWindows()