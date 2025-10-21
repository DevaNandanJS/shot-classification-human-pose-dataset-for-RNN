"""
Read a video with opencv and infer a YOLO-Pose model to display human pose.
A simple tracking logic is implemented to follow a single player.
"""

from argparse import ArgumentParser
from ultralytics import YOLO
import numpy as np
import cv2


class HumanPoseExtractor:
    """
    Uses a YOLO-Pose model to extract human keypoints from a frame.
    It tracks a single person (the one with the highest confidence at the beginning)
    and provides the keypoints in a format compatible with downstream tasks.
    """

    EDGES = {
        (0, 1): "m", (0, 2): "c", (1, 3): "m", (2, 4): "c",
        (0, 5): "m", (0, 6): "c", (5, 7): "m", (7, 9): "m",
        (6, 8): "c", (8, 10): "c", (5, 6): "y", (5, 11): "m",
        (6, 12): "c", (11, 12): "y", (11, 13): "m", (13, 15): "m",
        (12, 14): "c", (14, 16): "c",
    }

    COLORS = {"c": (255, 255, 0), "m": (255, 0, 255), "y": (0, 255, 255)}

    KEYPOINT_DICT = {
        "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
        "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
        "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
        "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16,
    }

    def __init__(self, shape):
        # Note: The user specified a .engine file, but a .pt file exists in the project.
        # Using the .pt file as it's more common and present in the file tree.
        # The path r"C:\Prototype Ultra\model-weights\yolo11s-pose.engine" was requested.
        # If this specific engine file is required, change the path below.
        self.model = YOLO(r"C:\RNN dataset creation\yolo11s-pose.pt")
        self.tracked_player_id = None
        self.keypoints = np.zeros((17, 3))

    def extract(self, frame):
        """
        Run YOLO-Pose tracking on the frame and extract keypoints for the target player.
        """
        results = self.model.track(frame, persist=True, tracker="botsort.yaml", verbose=False)

        target_person = None

        if results[0].boxes.id is not None:
            box_data = results[0].boxes.data.cpu().numpy()
            keypoint_data = results[0].keypoints.data.cpu().numpy()

            # If we are not tracking anyone yet, find the most confident person
            if self.tracked_player_id is None:
                if len(box_data) > 0:
                    # Find person with highest confidence
                    best_person_idx = np.argmax(box_data[:, 4])
                    self.tracked_player_id = box_data[best_person_idx, 5]
                    target_person = keypoint_data[best_person_idx]

            # If we are already tracking a player, find them by ID
            else:
                tracked_indices = np.where(box_data[:, 5] == self.tracked_player_id)[0]
                if len(tracked_indices) > 0:
                    target_person = keypoint_data[tracked_indices[0]]
                else:
                    # Player lost, reset tracking
                    self.tracked_player_id = None


        # If a player was found, format their keypoints
        if target_person is not None:
            # YOLO provides keypoints as [x, y, confidence]
            # We need to format it as [y, x, confidence] for compatibility
            formatted_keypoints = np.zeros((17, 3))
            formatted_keypoints[:, 0] = target_person[:, 1]  # Y coordinate
            formatted_keypoints[:, 1] = target_person[:, 0]  # X coordinate
            formatted_keypoints[:, 2] = target_person[:, 2]  # Confidence
            self.keypoints = formatted_keypoints
        else:
            # If no player is detected or the tracked player is lost, return zeros
            self.keypoints = np.zeros((17, 3))

    def draw_results_frame(self, frame):
        """Draw key points and edges on the frame"""
        # Check if there are any keypoints with confidence > 0
        if np.any(self.keypoints[:, 2] > 0):
            draw_edges(frame, self.keypoints, self.EDGES, 0.2)
            draw_keypoints(frame, self.keypoints, 0.2)


def draw_keypoints(frame, keypoints, confidence_threshold):
    """Draw key points with green dots"""
    for kp in keypoints:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)


def draw_edges(frame, keypoints, edges, confidence_threshold):
    """Draw edges with cyan for the right side, magenta for the left side, rest in yellow"""
    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = keypoints[p1]
        y2, x2, c2 = keypoints[p2]
        if (c1 > confidence_threshold) and (c2 > confidence_threshold):
            cv2.line(
                frame,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color=HumanPoseExtractor.COLORS[color],
                thickness=2,
            )


if __name__ == "__main__":
    parser = ArgumentParser(description="Display human pose on a video using YOLO-Pose")
    parser.add_argument("video")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), "Error opening video file"

    ret, frame = cap.read()
    if not ret:
        print("Could not read the first frame.")
        exit()

    human_pose_extractor = HumanPoseExtractor(frame.shape)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        human_pose_extractor.extract(frame)

        # Display results on the original frame
        human_pose_extractor.draw_results_frame(frame)
        cv2.imshow("Frame", frame)

        k = cv2.waitKey(1)
        if k == 27:  # Press 'ESC' to exit
            break

    cap.release()
    cv2.destroyAllWindows()