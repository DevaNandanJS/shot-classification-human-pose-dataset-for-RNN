"""
This script engineers a "13-Feature Master Set" for tennis shot classification by processing
video footage and an accompanying annotation file. It extracts human pose sequences, normalizes
them to be position- and scale-invariant, calculates advanced biomechanical features, and
saves them as individual CSV files for machine learning model training.

--- PURPOSE ---
The primary goal is to create a high-quality, structured dataset for training a Recurrent
Neural Network (RNN) to recognize 7 distinct tennis shot classes (e.g., r_serve, l_forehand).
The key innovation is a handedness-agnostic 13-feature vector that provides a comprehensive
biomechanical snapshot, allowing the model to learn from a unified data representation.

--- LOGIC & WORKING ---
1.  **Initialization**: Parses command-line arguments for input video, annotation CSV, and an
    output directory. The process is now fully handedness-agnostic.
2.  **Data Loading**: Loads the annotation CSV, which maps frame numbers to specific shot
    labels (e.g., "r_forehand", "l_backhand").
3.  **Video Processing**: Iterates through the video frame by frame using OpenCV.
4.  **Pose Extraction**: In each frame, it uses the `HumanPoseExtractor` class (YOLO-based)
    to get the raw pixel coordinates of 17 human keypoints.
5.  **Pose-Centric Normalization**: For each detected pose, it performs a neutral normalization:
    a. **Centering**: Translates all keypoints so the hip center becomes the origin (0,0).
    b. **Scaling**: Scales all keypoint coordinates by the torso length (shoulder-to-hip distance).
    This process removes variance from player position and camera distance, producing a
    standardized 26-feature vector (13 keypoints x 2 coordinates).
6.  **Feature Slicing**: When the video's current frame number matches a labeled shot, the script
    captures a fixed-length sequence (`NB_IMAGES`) of these normalized 26-feature poses.
7.  **13-Feature "Master Set" Calculation**: The (30, 26) sequence is passed to the
    `calculate_13_biomech_features` function. This function computes the final 13-feature
    vector for each frame, consisting of:
    - 3 Shared Features (e.g., vertical hip velocity, trunk flexion)
    - 5 Right-Side Features (e.g., right knee angle, right elbow angle)
    - 5 Left-Side Features (e.g., left knee angle, left elbow angle)
8.  **"Neutral" Pose Generation**: To create a negative class, the script also extracts pose
    sequences from the midpoint of large time gaps between annotated shots.
9.  **Dynamic Data Export**: Each captured sequence (a (30, 13) array) is saved as a separate
    CSV file. The filename is dynamically generated based on the shot class from the
    annotation file and a unique index (e.g., `r_serve_001.csv`, `l_forehand_002.csv`).

--- TECHNOLOGIES ---
- Python, OpenCV: For video and image processing.
- Pandas, NumPy: For data manipulation and numerical operations.
- `HumanPoseExtractor`: A custom class using a YOLO-based model for pose estimation.
- Argparse, Pathlib: For command-line interface and file path management.

--- USAGE ---
Run from the command line, providing paths to the video, annotations, and output folder.
`python features_pose_centric.py --video <path_to_video> --annotation <path_to_csv> --out <output_folder>`
"""

from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import cv2
import pandas as pd
from scipy.signal import savgol_filter

from extract_human_pose import (
    HumanPoseExtractor,
)

COLUMNS_13_FEATURES = [
    # Shared Features (3)
    "vertical_hip_velocity",
    "lateral_trunk_flexion_angle",
    "transverse_rotation_proxy",
    # Right-Side Features (5)
    "right_knee_angle",
    "right_shoulder_abduction",
    "right_elbow_angle",
    "right_wrist_y_rel_hip_y",
    "right_forearm_swing_path",
    # Left-Side Features (5)
    "left_knee_angle",
    "left_shoulder_abduction",
    "left_elbow_angle",
    "left_wrist_y_rel_hip_y",
    "left_forearm_swing_path",
]


def draw_shot(frame, shot):
    """Draw shot name on frame (user-friendly)"""
    cv2.putText(
        frame,
        shot,
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=(0, 165, 255),
        thickness=2,
    )
    print(f"Capturing {shot}")


def torso_scale_and_normalize(keypoints):
    """
    Normalizes raw pixel keypoints using torso-scaling, adapted from the DTW pipeline.
    Returns a flattened NumPy array of 26 features in (y, x) order and the un-normalized
    y-coordinate of the hip center.
    """
    # Keypoint indices from COCO
    l_shoulder_idx, r_shoulder_idx, l_hip_idx, r_hip_idx = 5, 6, 11, 12
    
    # Check if essential keypoints are detected (confidence > 0)
    if any(keypoints[i, 2] == 0 for i in [l_shoulder_idx, r_shoulder_idx, l_hip_idx, r_hip_idx]):
        return None, None

    # Keypoints are (y, x, conf). We need (x, y) for geometric calculations.
    raw_kpts_xy = keypoints[:, [1, 0]]
    
    # 1. Calculate centers (in xy)
    l_shoulder, r_shoulder = raw_kpts_xy[l_shoulder_idx], raw_kpts_xy[r_shoulder_idx]
    l_hip, r_hip = raw_kpts_xy[l_hip_idx], raw_kpts_xy[r_hip_idx]
    hip_center = (l_hip + r_hip) / 2
    shoulder_center = (l_shoulder + r_shoulder) / 2
    
    # 2. Calculate scaling factor (torso length)
    torso_length = np.linalg.norm(shoulder_center - hip_center)
    if torso_length < 1e-6:
        return None, None

    # 3. Select 13 keypoints to keep (y, x, conf) and their xy coordinates
    indices_to_keep = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    kpts_13_all_data = keypoints[indices_to_keep]
    kpts_13_xy = kpts_13_all_data[:, [1, 0]]

    # 4. Normalize (translate and scale)
    normalized_kpts_xy = (kpts_13_xy - hip_center) / torso_length
    
    # 5. Zero out low-confidence keypoints
    confidences = kpts_13_all_data[:, 2]
    normalized_kpts_xy[confidences < 0.2] = 0.0
    
    # 6. Reorder to (y, x) and flatten to get the 26-feature vector
    normalized_kpts_yx = normalized_kpts_xy[:, [1, 0]]
    
    # Return both the normalized vector and the original hip_center y-coordinate
    unnormalized_hip_y = hip_center[1]
    return normalized_kpts_yx.flatten(), unnormalized_hip_y


def calculate_angle(p1, p2, p3):
    """Computes the angle in degrees between three points."""
    v1 = np.array(p1) - np.array(p2)
    v2 = np.array(p3) - np.array(p2)
    dot_product = np.dot(v1, v2)
    norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm_product == 0:
        return np.nan
    cosine_angle = dot_product / norm_product
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)


def calculate_13_biomech_features(seq_26_features_yx, seq_unnormalized_hip_y):
    """
    Calculates the 13 expert-defined biomechanical features from a full sequence
    of 26 (y,x) torso-scaled keypoints. This version is handedness-agnostic.

    Args:
        seq_26_features_yx: A (30, 26) NumPy array of torso-scaled keypoints
                            in (y, x) order.
        seq_unnormalized_hip_y: A list or array of 30 un-normalized y-coordinates
                                for the hip center, used for velocity calculation.

    Returns:
        A (30, 13) NumPy array of the new biomechanical features.
    """
    # --- 1. Define feature names and keypoint indices ---
    feature_names_yx = [
        'nose_y', 'nose_x', 'left_shoulder_y', 'left_shoulder_x', 'right_shoulder_y', 'right_shoulder_x',
        'left_elbow_y', 'left_elbow_x', 'right_elbow_y', 'right_elbow_x', 'left_wrist_y', 'left_wrist_x',
        'right_wrist_y', 'right_wrist_x', 'left_hip_y', 'left_hip_x', 'right_hip_y', 'right_hip_x',
        'left_knee_y', 'left_knee_x', 'right_knee_y', 'right_knee_x', 'left_ankle_y', 'left_ankle_x',
        'right_ankle_y', 'right_ankle_x'
    ]
    
    df = pd.DataFrame(seq_26_features_yx, columns=feature_names_yx)
    num_frames = len(df)
    new_features = np.zeros((num_frames, 13))

    # --- 2. Pre-calculate helper values (vectors and points) ---
    
    # Side-specific keypoint names
    r_s_x, r_s_y = 'right_shoulder_x', 'right_shoulder_y'
    r_e_x, r_e_y = 'right_elbow_x', 'right_elbow_y'
    r_w_x, r_w_y = 'right_wrist_x', 'right_wrist_y'
    r_h_x, r_h_y = 'right_hip_x', 'right_hip_y'
    r_k_x, r_k_y = 'right_knee_x', 'right_knee_y'
    r_a_x, r_a_y = 'right_ankle_x', 'right_ankle_y'

    l_s_x, l_s_y = 'left_shoulder_x', 'left_shoulder_y'
    l_e_x, l_e_y = 'left_elbow_x', 'left_elbow_y'
    l_w_x, l_w_y = 'left_wrist_x', 'left_wrist_y'
    l_h_x, l_h_y = 'left_hip_x', 'left_hip_y'
    l_k_x, l_k_y = 'left_knee_x', 'left_knee_y'
    l_a_x, l_a_y = 'left_ankle_x', 'left_ankle_y'

    # Hip & Shoulder Centers (from normalized data for angles)
    shoulder_center_y = (df[r_s_y] + df[l_s_y]) / 2.0
    shoulder_center_x = (df[r_s_x] + df[l_s_x]) / 2.0
    
    # Trunk Vector
    trunk_vec_x = shoulder_center_x
    trunk_vec_y = shoulder_center_y

    # --- 3. Calculate Shared Features (3) ---

    # Feature 1: vertical_hip_velocity (using UN-NORMALIZED hip data)
    hip_center_y = np.array(seq_unnormalized_hip_y)
    window_len = min(5, num_frames - (1 if num_frames % 2 == 0 else 0))
    if window_len > 1:
        hip_center_y_smooth = savgol_filter(hip_center_y, window_len, 3)
    else:
        hip_center_y_smooth = hip_center_y
    new_features[:, 0] = np.gradient(hip_center_y_smooth)

    # Feature 2: lateral_trunk_flexion_angle
    trunk_mag = np.linalg.norm(np.stack([trunk_vec_x, trunk_vec_y], axis=1), axis=1)
    trunk_mag[trunk_mag < 1e-6] = 1e-6 # Avoid division by zero
    cos_theta_trunk = trunk_vec_y / trunk_mag
    new_features[:, 1] = np.degrees(np.arccos(np.clip(cos_theta_trunk, -1.0, 1.0)))

    # Feature 3: transverse_rotation_proxy
    shoulder_width = np.linalg.norm(df[[r_s_x, r_s_y]].values - df[[l_s_x, l_s_y]].values, axis=1)
    hip_width = np.linalg.norm(df[[r_h_x, r_h_y]].values - df[[l_h_x, l_h_y]].values, axis=1)
    hip_width[hip_width < 1e-6] = 1e-6
    new_features[:, 2] = shoulder_width / hip_width

    # --- 4. Calculate Side-Specific Features (Angles and Positions) ---
    for i in range(num_frames):
        # Right side points
        r_hip_pt = (df.at[i, r_h_x], df.at[i, r_h_y])
        r_knee_pt = (df.at[i, r_k_x], df.at[i, r_k_y])
        r_ankle_pt = (df.at[i, r_a_x], df.at[i, r_a_y])
        r_shoulder_pt = (df.at[i, r_s_x], df.at[i, r_s_y])
        r_elbow_pt = (df.at[i, r_e_x], df.at[i, r_e_y])
        r_wrist_pt = (df.at[i, r_w_x], df.at[i, r_w_y])

        # Left side points
        l_hip_pt = (df.at[i, l_h_x], df.at[i, l_h_y])
        l_knee_pt = (df.at[i, l_k_x], df.at[i, l_k_y])
        l_ankle_pt = (df.at[i, l_a_x], df.at[i, l_a_y])
        l_shoulder_pt = (df.at[i, l_s_x], df.at[i, l_s_y])
        l_elbow_pt = (df.at[i, l_e_x], df.at[i, l_e_y])
        l_wrist_pt = (df.at[i, l_w_x], df.at[i, l_w_y])
        
        # Shared trunk vector for this frame
        trunk_vec_i = np.array([trunk_vec_x[i], trunk_vec_y[i]])
        trunk_mag_i = trunk_mag[i]

        # --- RIGHT-SIDE FEATURES (5) ---
        # Feature 4: right_knee_angle
        new_features[i, 3] = calculate_angle(r_hip_pt, r_knee_pt, r_ankle_pt)
        
        # Feature 5: right_shoulder_abduction
        r_upper_arm_vec = np.array(r_elbow_pt) - np.array(r_shoulder_pt)
        r_upper_arm_mag = np.linalg.norm(r_upper_arm_vec)
        if trunk_mag_i > 0 and r_upper_arm_mag > 0:
            dot = np.dot(trunk_vec_i, r_upper_arm_vec)
            cos_theta = dot / (trunk_mag_i * r_upper_arm_mag)
            new_features[i, 4] = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        # Feature 6: right_elbow_angle
        new_features[i, 5] = calculate_angle(r_shoulder_pt, r_elbow_pt, r_wrist_pt)

        # Feature 7: right_wrist_y_rel_hip_y
        new_features[i, 6] = df.at[i, r_w_y]

        # Feature 8: right_forearm_swing_path
        r_forearm_vec = np.array(r_wrist_pt) - np.array(r_elbow_pt)
        r_forearm_mag = np.linalg.norm(r_forearm_vec)
        if r_forearm_mag > 0:
            cos_theta = r_forearm_vec[0] / r_forearm_mag # x-component
            new_features[i, 7] = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        # --- LEFT-SIDE FEATURES (5) ---
        # Feature 9: left_knee_angle
        new_features[i, 8] = calculate_angle(l_hip_pt, l_knee_pt, l_ankle_pt)

        # Feature 10: left_shoulder_abduction
        l_upper_arm_vec = np.array(l_elbow_pt) - np.array(l_shoulder_pt)
        l_upper_arm_mag = np.linalg.norm(l_upper_arm_vec)
        if trunk_mag_i > 0 and l_upper_arm_mag > 0:
            dot = np.dot(trunk_vec_i, l_upper_arm_vec)
            cos_theta = dot / (trunk_mag_i * l_upper_arm_mag)
            new_features[i, 9] = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        # Feature 11: left_elbow_angle
        new_features[i, 10] = calculate_angle(l_shoulder_pt, l_elbow_pt, l_wrist_pt)

        # Feature 12: left_wrist_y_rel_hip_y
        new_features[i, 11] = df.at[i, l_w_y]

        # Feature 13: left_forearm_swing_path
        l_forearm_vec = np.array(l_wrist_pt) - np.array(l_elbow_pt)
        l_forearm_mag = np.linalg.norm(l_forearm_vec)
        if l_forearm_mag > 0:
            cos_theta = l_forearm_vec[0] / l_forearm_mag # x-component
            new_features[i, 12] = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    return np.nan_to_num(new_features)


if __name__ == "__main__":
    # --- Hardcoded paths for easy execution ---
    # TODO: Manually update these paths before running the script.
    VIDEO_PATH = r"C:\RNN dataset creation\input\right-handed\r2.mp4"
    ANNOTATION_PATH = r"C:\RNN dataset creation\annotation\annotation_r2.csv"
    OUTPUT_PATH = r"C:\RNN dataset creation\dataset-non\r2"
    
    # Create a mock 'args' object to hold the paths and flags
    from argparse import Namespace
    args = Namespace(
        video=VIDEO_PATH,
        annotation=ANNOTATION_PATH,
        out=OUTPUT_PATH,
        show=True, # Set to True to display video frames during processing
        debug=False
    )

    # Ensure output directory exists
    Path(args.out).mkdir(parents=True, exist_ok=True)
    print(f"Output directory ensured to exist at: {args.out}")

    shots = pd.read_csv(args.annotation)
    CURRENT_ROW = 0

    NB_IMAGES = 30
    sequence_26_features = []
    unnormalized_hip_y_sequence = []

    FRAME_ID = 1
    shot_counters = {}
    
    # --- NEW STATE VARIABLE for robust shot capture ---
    capturing_shot_sequence = False
    
    # --- NEW TRACKING STATE VARIABLES ---
    target_id = None
    frames_since_target_lost = 0
    LOST_THRESHOLD = 30 # Number of frames to wait before re-acquiring a target

    # New variables for display
    current_csv_filename = ""
    display_filename_for_frames = 0
    MAX_DISPLAY_WIDTH = 1280 # Maximum width for the display window

    cap = cv2.VideoCapture(args.video)
    assert cap.isOpened(), f"Failed to open video: {args.video}"

    # --- Get video properties ---
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_center = (frame_w / 2, frame_h / 2)

    def format_time(seconds):
        """Helper to format seconds into MM:SS string."""
        mins, secs = divmod(seconds, 60)
        return f"{int(mins):02d}:{int(secs):02d}"

    # Use a valid model path
    model_path = r"C:\RNN dataset creation\yolo11s-pose.pt"
    human_pose_extractor = HumanPoseExtractor(model_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if CURRENT_ROW >= len(shots):
            print("Done, no more shots in annotation!")
            break

        # --- POSE EXTRACTION AND TRACKING LOGIC ---
        human_pose_extractor.extract(frame)
        human_pose_extractor.discard(["left_eye", "right_eye", "left_ear", "right_ear"])

        # --- TARGET ACQUISITION AND RE-ACQUISITION ---
        current_detections = human_pose_extractor.keypoints_pixels_frame

        # If we have a target, check if they are still in the frame
        if target_id is not None:
            if target_id not in current_detections:
                frames_since_target_lost += 1
                if frames_since_target_lost > LOST_THRESHOLD:
                    print(f"Target ID {target_id} lost for {LOST_THRESHOLD} frames. Re-acquiring...")
                    target_id = None # Declare target lost, trigger re-acquisition
            else:
                frames_since_target_lost = 0 # Target found, reset counter

        # If we don't have a target (initial state or after being lost), find one
        if target_id is None and len(current_detections) > 0:
            # Find the person with the highest confidence score
            if human_pose_extractor.confidences_frame:
                best_id = max(human_pose_extractor.confidences_frame, key=human_pose_extractor.confidences_frame.get)
                target_id = best_id
                frames_since_target_lost = 0
                print(f"Locked on to new target by highest confidence: ID {target_id}")

        # --- FEATURE EXTRACTION (only if we have a valid target) ---
        features_17_3 = None
        if target_id is not None and target_id in current_detections:
            features_17_3 = np.array(current_detections[target_id])
        
        # If no target is being tracked for this frame, we can't collect features
        if features_17_3 is None:
            # If we were in the middle of a sequence, scrap it
            if len(sequence_26_features) > 0:
                print("Target lost mid-sequence. Discarding current sequence.")
                # If we were capturing a shot, we failed. Advance to the next annotation.
                if capturing_shot_sequence:
                    print(f"Advancing to next shot annotation because shot at frame {shots.iloc[CURRENT_ROW]['FrameId']} was missed.")
                    CURRENT_ROW += 1
                
                sequence_26_features = []
                unnormalized_hip_y_sequence = []
                capturing_shot_sequence = False # Reset flag
        else:
            # --- SHOT/NEUTRAL CAPTURE LOGIC (unchanged) ---
            is_shot_frame = (
                shots.iloc[CURRENT_ROW]["FrameId"] - NB_IMAGES // 2
                <= FRAME_ID
                <= shots.iloc[CURRENT_ROW]["FrameId"] + NB_IMAGES // 2
            )
            is_neutral_frame = False
            if CURRENT_ROW > 0:
                gap = shots.iloc[CURRENT_ROW]["FrameId"] - shots.iloc[CURRENT_ROW - 1]["FrameId"]
                if gap > NB_IMAGES * 2:
                    frame_id_between_shots = (shots.iloc[CURRENT_ROW - 1]["FrameId"] + shots.iloc[CURRENT_ROW]["FrameId"]) // 2
                    if (frame_id_between_shots - NB_IMAGES // 2 < FRAME_ID <= frame_id_between_shots + NB_IMAGES // 2):
                        is_neutral_frame = True
            
            # --- PROCESS IF SHOT OR NEUTRAL ---
            if is_shot_frame or is_neutral_frame:
                if len(sequence_26_features) == 0:
                    print("Starting new sequence capture...")
                    # Set the flag if we are starting a sequence for a real shot
                    if is_shot_frame:
                        capturing_shot_sequence = True

                normalized_26_features, unnormalized_hip_y = torso_scale_and_normalize(features_17_3)
                
                if normalized_26_features is None:
                    print(f"Skipping frame {FRAME_ID}: Could not normalize pose (missing keypoints).")
                    sequence_26_features = []
                    unnormalized_hip_y_sequence = []
                    capturing_shot_sequence = False # Reset flag
                    if is_shot_frame:
                        CURRENT_ROW += 1
                    FRAME_ID += 1
                    continue

                sequence_26_features.append(normalized_26_features)
                unnormalized_hip_y_sequence.append(unnormalized_hip_y)
                
                shot_class = "neutral" if is_neutral_frame else shots.iloc[CURRENT_ROW]["Shot"]
                draw_shot(frame, shot_class)

                if len(sequence_26_features) == NB_IMAGES:
                    sequence_array = np.array(sequence_26_features)
                    features_13 = calculate_13_biomech_features(sequence_array, unnormalized_hip_y_sequence)
                    shots_df = pd.DataFrame(features_13, columns=COLUMNS_13_FEATURES)
                    shots_df["shot"] = shot_class
                    
                    shot_counters[shot_class] = shot_counters.get(shot_class, 0) + 1
                    idx = shot_counters[shot_class]
                    outpath = Path(args.out).joinpath(f"{shot_class}_{idx:03d}.csv")
                    
                    shots_df.to_csv(outpath, index=False)
                    print(f"Saving {shot_class} ({NB_IMAGES} frames, 13 features) to {outpath}")

                    current_csv_filename = outpath.name
                    display_filename_for_frames = NB_IMAGES

                    sequence_26_features = []
                    unnormalized_hip_y_sequence = []
                    capturing_shot_sequence = False # Reset flag
                    if is_shot_frame:
                        CURRENT_ROW += 1

        # --- CLEANUP LOGIC for incomplete sequences ---
        # Check if we have just exited a shot's capture window with an incomplete sequence
        if CURRENT_ROW < len(shots): # Ensure we don't read past the end of the dataframe
            was_shot_frame_previously = (
                shots.iloc[CURRENT_ROW]["FrameId"] - NB_IMAGES // 2
                <= FRAME_ID - 1 # Check the previous frame
                <= shots.iloc[CURRENT_ROW]["FrameId"] + NB_IMAGES // 2
            )
            is_shot_frame_currently = (
                shots.iloc[CURRENT_ROW]["FrameId"] - NB_IMAGES // 2
                <= FRAME_ID
                <= shots.iloc[CURRENT_ROW]["FrameId"] + NB_IMAGES // 2
            )

            if was_shot_frame_previously and not is_shot_frame_currently and 0 < len(sequence_26_features) < NB_IMAGES:
                print(f"Discarding incomplete sequence ({len(sequence_26_features)}/{NB_IMAGES} frames) for shot at frame {shots.iloc[CURRENT_ROW]['FrameId']}.")
                sequence_26_features = []
                unnormalized_hip_y_sequence = []
                # If we were trying to capture a shot, we failed. Move on.
                if capturing_shot_sequence:
                    CURRENT_ROW += 1
                capturing_shot_sequence = False

        # --- DISPLAY AND UI LOGIC ---
        if args.show:
            # Draw Progress Bar / Timestamp
            current_time_sec = FRAME_ID / fps
            progress_text = f"Frame: {FRAME_ID}/{total_frames} | Time: {format_time(current_time_sec)} / {format_time(total_frames / fps)}"
            cv2.putText(frame, progress_text, (20, frame_h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Display CSV filename if active
            if display_filename_for_frames > 0:
                cv2.putText(frame, current_csv_filename, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                display_filename_for_frames -= 1

            # --- DRAW TRACKING RESULTS ---
            # This will now draw the bounding box and ID for the target player
            human_pose_extractor.draw_results_frame(frame, target_id=target_id, boxes=human_pose_extractor.boxes_frame)
            
            # Resize frame for display if it's too large
            display_frame = frame.copy()
            if display_frame.shape[1] > MAX_DISPLAY_WIDTH:
                scale_factor = MAX_DISPLAY_WIDTH / display_frame.shape[1]
                new_height = int(display_frame.shape[0] * scale_factor)
                display_frame = cv2.resize(display_frame, (MAX_DISPLAY_WIDTH, new_height))
            
            cv2.imshow("Frame", display_frame)

        k = cv2.waitKeyEx(1)

        # --- UI CONTROLS (SEEK, PAUSE, EXIT) ---
        if args.show and (k == 2555904 or k == 2424832): # Right/Left Arrow
            seek_frames = int(5 * fps) * (1 if k == 2555904 else -1)
            direction = "forward" if seek_frames > 0 else "backward"
            new_frame_id = max(0, min(FRAME_ID + seek_frames, total_frames - 1))
            if new_frame_id != FRAME_ID:
                FRAME_ID = new_frame_id
                cap.set(cv2.CAP_PROP_POS_FRAMES, FRAME_ID)
                if len(sequence_26_features) > 0:
                    sequence_26_features = []
                    unnormalized_hip_y_sequence = []
                    print("Sequence buffer cleared due to seek.")
                print(f"Skipping {direction} 5s to frame {FRAME_ID}")
                continue
        
        if args.show and k == ord(' '): # Spacebar for Pause
            overlay = frame.copy()
            cv2.putText(overlay, "PAUSED", (frame_w // 2 - 70, frame_h // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 50, 255), 3)
            cv2.imshow("Frame", overlay)
            while True:
                pause_k = cv2.waitKey(0)
                if pause_k == ord(' '): k = -1; break
                if pause_k == 27: k = 27; break
        
        if k == 27: break # ESC to exit
        FRAME_ID += 1

    cap.release()
    cv2.destroyAllWindows()
