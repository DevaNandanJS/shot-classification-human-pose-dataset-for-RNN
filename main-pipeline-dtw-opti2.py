'''
This script is the main executable for a unified sports analysis pipeline, specifically tailored for tennis.
It integrates real-time pose estimation, RNN-based shot classification, and on-demand Dynamic Time Warping (DTW)
similarity analysis into a seamless workflow that processes a video and generates an annotated output.

--- PURPOSE & SIGNIFICANCE ---
This script represents the complete, end-to-end application. It takes a raw video and produces actionable
insights by:
1.  **Tracking Players:** Identifying and tracking a target player throughout the video.
2.  **Classifying Shots:** Using a lightweight RNN to classify every action (e.g., forehand, backhand, serve)
    in real-time from a continuous stream of pose data.
3.  **Scoring Serves:** When the RNN detects a "serve," it triggers a more detailed DTW analysis to compare
    the user's serve motion against a pre-recorded professional template, yielding a similarity score.

The architecture is designed for efficiency, using the fast RNN for general classification and reserving the
more computationally expensive DTW comparison for specific, triggered events.

--- ARCHITECTURE & LOGIC ---
1.  **Initialization**: Loads all necessary models and assets: YOLO for pose estimation, the ONNX RNN model for
    classification, and the DTW assets (ground truth template and feature scaler).
2.  **Centralized Sliding Windows**: The core of the design. Two `collections.deque` buffers are maintained in
    the main loop: one for the RNN's 26 features and one for the DTW's 26 features (which are then used to engineer 9 more features). This makes the analysis
    components stateless.
3.  **Batch Video Processing**: The video is read frame-by-frame but processed in batches for efficient pose
    estimation with the YOLO model.
4.  **Sticky Player Tracking**: An algorithm locks onto a single `target_player_id`. If the player is lost,
    it attempts to reacquire them for a grace period based on proximity to their last known position.
5.  **Unified Pipeline Execution (per frame)**:
    a. The 26 pose features for the target player are extracted and added to both the RNN and DTW data buffers.
    b. **RNN Branch**: Once the RNN buffer is full, the sequence is sent to the `ShotClassifier`. It returns
       shot probabilities in real-time.
    c. **DTW Branch (Triggered)**: If the `ShotClassifier` confidently detects a "serve", it triggers the
       `DTWComparator`.
    d. **On-Demand Feature Engineering**: The `DTWComparator` takes the 26-feature sequence from its buffer,
       and then calculates 9 additional "engineered features" (angles, velocities, etc.).
    e. **Normalization**: The 9 engineered features are normalized using a pre-loaded `StandardScaler`.
    f. **DTW Comparison**: The newly created 9-feature user sequence is compared against the 9-feature
       professional ground truth using `fastdtw` with a custom weighted Euclidean distance, producing a similarity score.
6.  **Output Generation**: An annotated video is created showing the player's skeleton, tracking status,
    real-time shot probabilities, and the DTW similarity score when a serve is analyzed.

--- DYNAMIC TIME WARPING (DTW) ANALYSIS ---
The DTW analysis is a key component of this pipeline, providing a detailed comparison of a user's serve
motion to a professional's. Here's a breakdown of the process:

1.  **Feature Engineering**:
    - The `calculate_9_biomech_features` function takes a sequence of 26 torso-scaled keypoints (y, x coordinates
      for 13 keypoints) and calculates 9 biomechanical features. These features are designed to capture the
      essential aspects of the tennis serve motion, such as leg drive, trunk rotation, and arm action.
    - The 9 features are:
        1.  `vertical_hip_velocity`: The vertical velocity of the hip, which is a key indicator of leg drive.
        2.  `dominant_knee_angle`: The angle of the dominant knee, which is important for power generation.
        3.  `lateral_trunk_flexion_angle`: The angle of the trunk, which is a measure of trunk tilt.
        4.  `dominant_shoulder_abduction`: The angle of the dominant shoulder, which is a measure of how high the arm is raised.
        5.  `dominant_elbow_angle`: The angle of the dominant elbow.
        6.  `dom_wrist_y_rel_hip_y`: The vertical position of the dominant wrist relative to the hip.
        7.  `forearm_swing_path`: A proxy for forearm pronation, calculated as the angle of the forearm vector relative to the horizontal.
        8.  `transverse_rotation_proxy`: A proxy for transverse rotation, calculated as the ratio of shoulder width to hip width.
        9.  `toss_arm_timing_proxy`: The vertical position of the non-dominant wrist, which is a proxy for the timing of the toss.

2.  **Normalization**:
    - The raw keypoints are first normalized using `torso_scale_normalize`. This function centers the keypoints
      at the hip and scales them by the torso length. This makes the features invariant to the player's size
      and position in the frame.
    - The 9 engineered features are then scaled using a `StandardScaler` that was pre-fitted on a dataset of
      professional serves. This ensures that all features have a mean of 0 and a standard deviation of 1,
      which is important for the DTW calculation.

3.  **Weighted DTW**:
    - The `compare_shot` method in the `DTWComparator` class uses the `fastdtw` library to perform the DTW
      comparison.
    - A custom distance metric, `weighted_euclidean`, is used. This metric calculates the Euclidean distance
      between two feature vectors, but with each feature weighted according to its biomechanical importance.
    - The weights are defined in the `DTWComparator` class and are divided into three tiers:
        - **Tier 1 (Engine)**: The most important features for power generation (hip velocity, knee angle, trunk flexion).
        - **Tier 2 (Whip)**: Features related to the "whip" of the arm (shoulder abduction, elbow angle, wrist position).
        - **Tier 3 (Proxies)**: Proxies for more complex motions (pronation, rotation, toss timing).
    - The weighted Euclidean distance is calculated as: `sqrt(sum(w[i] * (u[i] - v[i])**2))`, where `w` is the
      weight vector, and `u` and `v` are the feature vectors.

4.  **Similarity Score**:
    - The raw DTW distance is then converted to a similarity score between 0 and 100. This is done by
      linearly scaling the distance based on a pre-calibrated `MIN_DIST` and `MAX_DIST`. These values
      represent the expected minimum and maximum DTW distances for a given serve and need to be
      re-calibrated if the ground truth or feature set changes.

--- TECHNOLOGIES ---
- OpenCV: For video I/O and all drawing/UI operations.
- Ultralytics YOLO: For high-performance, batched pose estimation and player tracking.
- ONNX Runtime: For efficient, cross-platform inference with the RNN classification model.
- Scikit-learn, Joblib: For loading and using the pre-fitted scaler for DTW feature normalization.
- fastdtw: For performing the Dynamic Time Warping calculation with a custom distance metric.
- TQDM: For displaying progress during video processing.

--- USAGE ---
Configure the file paths and processing settings in the 'Configuration' section, then run the script
directly from the command line.
`python main-pipeline-dtw-opti.py`
'''
# --- 1. Imports ---
import os
import time
from datetime import datetime
import cv2
import numpy as np
import pandas as pd
import onnxruntime
from ultralytics import YOLO
from tqdm import tqdm
import collections
import joblib
from scipy.signal import savgol_filter
from fastdtw import fastdtw

# --- 2. Configuration (EDIT THESE VALUES) ---
# --- FILE PATHS ---
VIDEO_PATH = r"C:\Prototype Ultra\Test\NewTest11-720.mp4"
YOLO_MODEL_PATH = r"C:\Prototype Ultra\model-weights\yolo11s-pose.onnx"
RNN_MODEL_PATH = r"C:\Prototype Ultra\model-weights\tennis-RNN-cassi-v2.onnx"
GROUND_TRUTH_PATH = r"C:\Prototype Ultra\tennis-pipeline\DTW-GroundTruthData\ProServeFeatures\serve_pro_9_feature.csv"
ENGINEERED_FEATURES_SCALER_PATH = r"C:\Prototype Ultra\tennis-pipeline\DTW-GroundTruthData\ProServeFeatures\scaler_9_feature.pkl"
OUTPUT_DIR = r"output"

# --- BATCH PROCESSING CONFIGURATION ---
BATCH_SIZE = 8

# --- PROCESSING SETTINGS ---
IS_PLAYER_LEFT_HANDED = True
START_FRAME = 0

# --- STICKY TRACKING CONFIGURATION ---
TRACKING_CONF_THRESHOLD = 0.50
REACQUISITION_GRACE_PERIOD = 30
PROXIMITY_THRESHOLD = 150

# --- MODEL AND CONSTANTS ---
SEQUENCE_LENGTH = 30
NUM_RNN_FEATURES = 26
NUM_DTW_FEATURES = 40 # Updated to 40 features
MIN_FRAMES_BETWEEN_SHOTS = 60
PROCESSING_WIDTH = 640

# --- UI CONSTANTS ---
BAR_WIDTH = 30; BAR_HEIGHT = 170; MARGIN_ABOVE_BAR = 30; SPACE_BETWEEN_BARS = 55
TEXT_ORIGIN_X = 1075; BAR_ORIGIN_X = 1070

# --- 3. GPU Configuration ---
def setup_tf_gpu():
    """Configures TensorFlow to use the GPU with memory growth if TF is used."""
    try:
        import tensorflow as tf
        physical_devices = tf.config.experimental.list_physical_devices("GPU")
        if physical_devices:
            print("TensorFlow GPU detected. Setting up memory growth.")
            tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except (ImportError, RuntimeError) as e:
        print(f"Skipping TensorFlow GPU setup: {e}")

# --- 4. Feature Engineering Helpers ---
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

def calculate_9_biomech_features(seq_26_features_yx, is_left_handed):
    """
    Calculates the 9 expert-defined biomechanical features from a full sequence
    of 26 (y,x) torso-scaled keypoints.
    
    Args:
        seq_26_features_yx: A (30, 26) NumPy array of torso-scaled keypoints
                            in (y, x) order.
        is_left_handed: Boolean flag to determine dominant side.
        
    Returns:
        A (30, 9) NumPy array of the new biomechanical features.
    """
    
    # --- 1. Define feature names and keypoint indices ---
    feature_names_yx = [
        'nose_y', 'nose_x', 'left_shoulder_y', 'left_shoulder_x', 'right_shoulder_y', 'right_shoulder_x',
        'left_elbow_y', 'left_elbow_x', 'right_elbow_y', 'right_elbow_x', 'left_wrist_y', 'left_wrist_x',
        'right_wrist_y', 'right_wrist_x', 'left_hip_y', 'left_hip_x', 'right_hip_y', 'right_hip_x',
        'left_knee_y', 'left_knee_x', 'right_knee_y', 'right_knee_x', 'left_ankle_y', 'left_ankle_x',
        'right_ankle_y', 'right_ankle_x'
    ]
    
    # Create DataFrame for easy column access
    df = pd.DataFrame(seq_26_features_yx, columns=feature_names_yx)
    
    # Define dominant/non-dominant sides
    # Note: Our `torso_scale_normalize` flips the x-axis for lefties, so
    # 'right_shoulder_x' *always* refers to the dominant shoulder's x-coord.
    # The plan assumes a right-handed player, so we use 'right_' as dominant.
    dom_s_x, dom_s_y = 'right_shoulder_x', 'right_shoulder_y'
    dom_e_x, dom_e_y = 'right_elbow_x', 'right_elbow_y'
    dom_w_x, dom_w_y = 'right_wrist_x', 'right_wrist_y'
    dom_h_x, dom_h_y = 'right_hip_x', 'right_hip_y'
    dom_k_x, dom_k_y = 'right_knee_x', 'right_knee_y'
    dom_a_x, dom_a_y = 'right_ankle_x', 'right_ankle_y'
    
    non_dom_w_y = 'left_wrist_y'
    
    l_s_x, l_s_y = 'left_shoulder_x', 'left_shoulder_y'
    l_h_x, l_h_y = 'left_hip_x', 'left_hip_y'
    
    # Helper lists for empty feature array
    num_frames = len(df)
    new_features = np.zeros((num_frames, 9))
    
    # --- 2. Pre-calculate helper values (vectors) ---
    
    # Hip Center (y-component only, for velocity)
    # Hip center is (0,0) in our torso-scaled world, but y-coords are not 0
    hip_center_y = (df[dom_h_y] + df[l_h_y]) / 2.0
    
    # Shoulder Center
    shoulder_center_y = (df[dom_s_y] + df[l_s_y]) / 2.0
    shoulder_center_x = (df[dom_s_x] + df[l_s_x]) / 2.0
    
    # Trunk Vector (Shoulder Center - Hip Center)
    # Note: Hip Center is (0,0), so Trunk Vector is just Shoulder Center
    trunk_vec_x = shoulder_center_x
    trunk_vec_y = shoulder_center_y
    
    # Dominant Upper Arm Vector (Elbow - Shoulder)
    upper_arm_vec_x = df[dom_e_x] - df[dom_s_x]
    upper_arm_vec_y = df[dom_e_y] - df[dom_s_y]
    
    # Dominant Forearm Vector (Wrist - Elbow)
    forearm_vec_x = df[dom_w_x] - df[dom_e_x]
    forearm_vec_y = df[dom_w_y] - df[dom_e_y]

    # --- 3. Calculate 9 Features (Vectorized where possible) ---

    # Feature 1: vertical_hip_velocity (Tier 1)
    # Apply Savitzky-Golay filter to smooth the Yhip signal
    # window_length must be odd and < num_frames. 5 is a good default.
    window_len = min(5, num_frames - (1 if num_frames % 2 == 0 else 0))
    if window_len > 1:
        hip_center_y_smooth = savgol_filter(hip_center_y, window_len, 3)
    else:
        hip_center_y_smooth = hip_center_y
    # Calculate first derivative (velocity)
    # np.gradient provides a more robust calculation than a simple diff
    new_features[:, 0] = np.gradient(hip_center_y_smooth)

    # Features 2, 3, 4, 5, 7 (Angles)
    # We must iterate frame by frame for angle calculations
    for i in range(num_frames):
        # --- (x, y) tuples for calculate_angle ---
        dom_hip_pt = (df.at[i, dom_h_x], df.at[i, dom_h_y])
        dom_knee_pt = (df.at[i, dom_k_x], df.at[i, dom_k_y])
        dom_ankle_pt = (df.at[i, dom_a_x], df.at[i, dom_a_y])
        
        dom_shoulder_pt = (df.at[i, dom_s_x], df.at[i, dom_s_y])
        dom_elbow_pt = (df.at[i, dom_e_x], df.at[i, dom_e_y])
        dom_wrist_pt = (df.at[i, dom_w_x], df.at[i, dom_w_y])
        
        # Feature 2: dominant_knee_angle (Tier 1)
        new_features[i, 1] = calculate_angle(dom_hip_pt, dom_knee_pt, dom_ankle_pt)
        
        # Feature 3: lateral_trunk_flexion_angle (Tier 1)
        # Angle between Trunk Vector and a pure Vertical Vector (0, 1)
        # Using dot product formula: A=[0,1], B=[trunk_x, trunk_y]
        # cos(theta) = (A . B) / (|A| * |B|) = trunk_y / |B|
        trunk_vec = np.array([trunk_vec_x[i], trunk_vec_y[i]])
        trunk_mag = np.linalg.norm(trunk_vec)
        if trunk_mag > 0:
            cos_theta = trunk_vec_y[i] / trunk_mag
            new_features[i, 2] = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))
        
        # Feature 4: dominant_shoulder_abduction (Tier 2)
        # Angle between Trunk Vector and Upper Arm Vector
        upper_arm_vec = np.array([upper_arm_vec_x[i], upper_arm_vec_y[i]])
        upper_arm_mag = np.linalg.norm(upper_arm_vec)
        if trunk_mag > 0 and upper_arm_mag > 0:
            dot = np.dot(trunk_vec, upper_arm_vec)
            cos_theta = dot / (trunk_mag * upper_arm_mag)
            new_features[i, 3] = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

        # Feature 5: dominant_elbow_angle (Tier 2)
        new_features[i, 4] = calculate_angle(dom_shoulder_pt, dom_elbow_pt, dom_wrist_pt)
        
        # Feature 7: forearm_swing_path (Tier 3)
        # Angle of forearm vector relative to horizontal (1, 0)
        forearm_vec = np.array([forearm_vec_x[i], forearm_vec_y[i]])
        forearm_mag = np.linalg.norm(forearm_vec)
        if forearm_mag > 0:
            # cos(theta) = (A . B) / (|A| * |B|) = forearm_x / |B|
            cos_theta = forearm_vec_x[i] / forearm_mag
            new_features[i, 6] = np.degrees(np.arccos(np.clip(cos_theta, -1.0, 1.0)))

    # Feature 6: dom_wrist_y_rel_hip_y (Tier 2)
    # Hip center y is 0 in torso-scaled space
    new_features[:, 5] = df[dom_w_y]
    
    # Feature 8: transverse_rotation_proxy (Tier 3)
    # Ratio = ShoulderWidth / HipWidth
    shoulder_width = np.linalg.norm(
        df[[dom_s_x, dom_s_y]].values - df[[l_s_x, l_s_y]].values, axis=1
    )
    hip_width = np.linalg.norm(
        df[[dom_h_x, dom_h_y]].values - df[[l_h_x, l_h_y]].values, axis=1
    )
    # Avoid division by zero
    hip_width[hip_width < 1e-6] = 1e-6
    new_features[:, 7] = shoulder_width / hip_width
    
    # Feature 9: toss_arm_timing_proxy (Tier 3)
    new_features[:, 8] = df[non_dom_w_y]

    # Fill any NaNs that may have been created (e.g., from angles)
    return np.nan_to_num(new_features)

def torso_scale_normalize(raw_keypoints_xy, keypoints_to_keep):
    """
    Normalizes raw pixel keypoints using torso-scaling (hip-centric, torso-length-scaled).
    Input is a (17, 2) or (13, 2) raw keypoint array.
    We need the indices for: L-Shoulder, R-Shoulder, L-Hip, R-Hip
    In the 17-point COCO set: 5, 6, 11, 12
    In our 13-point `keypoints_to_keep` set:
    - 5 is at index 1
    - 6 is at index 2
    - 11 is at index 7
    - 12 is at index 8
    """
    
    # Check if we got the full 17-point array or the 13-point
    if raw_keypoints_xy.shape[0] == len(keypoints_to_keep):
        # This is the 13-point array. Use mapped indices.
        l_shoulder = raw_keypoints_xy[1]
        r_shoulder = raw_keypoints_xy[2]
        l_hip = raw_keypoints_xy[7]
        r_hip = raw_keypoints_xy[8]
    elif raw_keypoints_xy.shape[0] >= 13:
        # This is the full array. Use direct COCO indices.
        l_shoulder = raw_keypoints_xy[5]
        r_shoulder = raw_keypoints_xy[6]
        l_hip = raw_keypoints_xy[11]
        r_hip = raw_keypoints_xy[12]
    else:
        # Not enough keypoints, return zeros
        return np.zeros_like(raw_keypoints_xy)

    # 1. Calculate centers
    hip_center = (l_hip + r_hip) / 2
    shoulder_center = (l_shoulder + r_shoulder) / 2
    
    # 2. Calculate scaling factor (torso length)
    torso_length = np.linalg.norm(shoulder_center - hip_center)
    
    # Handle division by zero if torso_length is 0
    if torso_length < 1e-6:
        # Cannot normalize, return None to signal failure
        return None

    # 3. Normalize all keypoints
    normalized_keypoints = (raw_keypoints_xy - hip_center) / torso_length
    
    return normalized_keypoints

# --- 5. Core Logic Classes ---
class PoseEstimator:
    """Wraps the YOLO model to perform tracking on a batch of frames."""
    def __init__(self, model_path):
        print(f"Loading YOLO pose model from: {model_path}")
        self.model = YOLO(model_path)
        self.keypoints_to_keep = [0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
        self.skeleton = [
            (0, 1), (0, 2), (1, 2), (1, 3), (2, 4), (3, 5), (4, 6),
            (1, 7), (2, 8), (7, 8), (7, 9), (8, 10), (9, 11), (10, 12)
        ]

    def track_players_in_batch(self, frame_batch):
        """Runs YOLO+BoT-SORT on a batch of frames in parallel."""
        results_list = self.model.track(
            frame_batch, persist=True, verbose=False, tracker="bytetrack.yaml", conf=TRACKING_CONF_THRESHOLD
        )
        batch_detections = []
        for results in results_list:
            frame_detections = []
            if results and results.boxes.id is not None:
                boxes = results.boxes.xyxyn.cpu().numpy()
                track_ids = results.boxes.id.int().cpu().tolist()
                confs = results.boxes.conf.cpu().numpy()
                keypoints_data = results.keypoints.cpu().numpy()
                for i, track_id in enumerate(track_ids):
                    # Get 0-1 normalized keypoints (for RNN)
                    kpts_xy_normalized = keypoints_data.xyn[i]
                    kpts_conf = keypoints_data.conf[i]
                    all_kpts_normalized = np.concatenate([kpts_xy_normalized, kpts_conf.reshape(-1, 1)], axis=1)
                    
                    # Get raw pixel keypoints (for DTW)
                    kpts_xy_raw = keypoints_data.xy[i]

                    player_data = {
                        "id": track_id, "box": boxes[i], "conf": confs[i],
                        "keypoints_norm_conf": all_kpts_normalized[self.keypoints_to_keep, :],
                        "keypoints_raw": kpts_xy_raw[self.keypoints_to_keep, :] # Get raw (x,y) for the 13 kpts
                    }
                    frame_detections.append(player_data)
            batch_detections.append(frame_detections)
        return batch_detections

class ShotClassifier:
    """(Refactored) Classifies shots using an ONNX RNN model without managing a feature buffer."""
    def __init__(self, model_path):
        print(f"Loading ONNX RNN classifier model from: {model_path}")
        self.session = onnxruntime.InferenceSession(
            model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        print(f"ONNX Runtime using provider: {self.session.get_providers()}")
        self.input_name = self.session.get_inputs()[0].name
        self.current_probs = np.zeros(4)

    def predict(self, features_sequence):
        """Processes a full feature sequence and returns shot probabilities."""
        # Reshape for the model: (1, sequence_length, num_features)
        features_seq_np = features_sequence.reshape(1, SEQUENCE_LENGTH, NUM_RNN_FEATURES)
        result = self.session.run(None, {self.input_name: features_seq_np})
        self.current_probs = result[0][0]
        return self.current_probs

class DTWComparator:
    """
    (Re-engineered) Performs a biomechanically-weighted 9-feature DTW comparison.
    
    This class loads a 9-feature ground truth and a 9-feature scaler.
    It uses a custom-weighted Euclidean distance (FW-DTW) to prioritize
    biomechanically important features (e.g., leg drive, trunk flexion).
    """
    def __init__(self, ground_truth_path, scaler_path):
        self.ground_truth_features = None
        self.scaler = None
        self.is_ready = False
        
        # --- Biomechanical Weights (Sum = 1.0) ---
        # Tier 1: Engine (0.51)
        self.w_hip_vel = 0.21
        self.w_knee_angle = 0.15
        self.w_trunk_flex = 0.15
        # Tier 2: Whip & Toss Timing (0.47)
        self.w_shoulder_abd = 0.13
        self.w_elbow_angle = 0.13
        self.w_wrist_y = 0.13
        self.w_toss_arm = 0.08 # Moved from Tier 3
        # Tier 3: Proxies (0.02)
        self.w_pronation = 0.01 # Dropped from 0.04
        self.w_rotation = 0.01 # Dropped from 0.03
        
        # This order MUST match the `calculate_9_biomech_features` output
        self.feature_weights = np.array([
            self.w_hip_vel, self.w_knee_angle, self.w_trunk_flex,
            self.w_shoulder_abd, self.w_elbow_angle, self.w_wrist_y,
            self.w_pronation, self.w_rotation, self.w_toss_arm
        ])

        # --- Feature Names (for breakdown report) ---
        self.feature_names = [
            "Hip Velocity", "Knee Angle", "Trunk Flexion",
            "Shoulder Abduction", "Elbow Angle", "Wrist Height",
            "Forearm Swing Path", "Torso Rotation", "Toss Arm"
        ]

        # --- Per-Feature Max Cost Baselines (NEEDS EMPIRICAL CALIBRATION) ---
        # These values represent the 95th percentile of costs for each feature
        # from a dataset of "bad" serves. They are used to convert raw
        # DTW costs into an intuitive 0-100% similarity score.
        # PLACEHOLDER VALUES:
        self.max_cost_baselines = np.array([
            100.0, 150.0, 100.0, 120.0, 150.0, 100.0, 100.0, 50.0, 100.0
        ])
        
        try:
            # --- IMPORTANT ---
            # This will fail until we generate the new 9-feature files.
            # We are writing this code in preparation for those new files.
            print(f"Loading 9-feature DTW ground truth from: {ground_truth_path}")
            ground_truth_df = pd.read_csv(ground_truth_path)
            
            # Get the 9 feature columns (excluding 'shot' label if it exists)
            feature_cols = [col for col in ground_truth_df.columns if col != 'shot']
            if len(feature_cols) != 9:
                 print(f"Warning: Expected 9 features, found {len(feature_cols)}.")
                 
            self.ground_truth_features = ground_truth_df[feature_cols].values
            
            print(f"Loading 9-feature scaler from: {scaler_path}")
            self.scaler = joblib.load(scaler_path)
            
            print("Biomech-DTW assets loaded successfully for 9-feature comparison.")
            self.is_ready = True
            
        except FileNotFoundError as e:
            print(f"DTW asset loading failed (THIS IS EXPECTED until new files are generated): {e}.")
        except Exception as e:
            print(f"An error occurred during DTW asset loading: {e}.")

    def create_9_biomech_features(self, user_shot_window_26_features, is_left_handed):
        """
        Takes the 30-frame, 26-feature (y,x) torso-scaled keypoint sequence,
        calculates the 9 biomechanical features, and scales them.
        """
        # 1. Calculate the (30, 9) biomech features from the (30, 26) keypoints
        biomech_features_unscaled = calculate_9_biomech_features(
            user_shot_window_26_features, is_left_handed
        )
        
        # 2. Scale the 9 features using the pre-fitted 9-feature scaler
        biomech_features_scaled = self.scaler.transform(biomech_features_unscaled)
        
        return biomech_features_scaled

    def compare_shot(self, user_shot_window_26_features, is_left_handed):
        """
        Performs the full 9-feature, weighted DTW comparison and returns a
        per-feature breakdown of similarity.
        """
        if not self.is_ready:
            return 0.0, {}

        # 1. Create the (30, 9) scaled feature matrix for the user
        user_serve_9_features = self.create_9_biomech_features(
            user_shot_window_26_features, is_left_handed
        )
        
        # 2. Use fastdtw with a custom weighted Euclidean distance to find the optimal path
        def weighted_euclidean(u, v, w):
            return np.sqrt(np.sum(w * (u - v)**2))

        distance, path = fastdtw(
            user_serve_9_features,
            self.ground_truth_features,
            dist=lambda u, v: weighted_euclidean(u, v, w=self.feature_weights)
        )

        # --- 3. Calculate Per-Feature Costs along the Optimal Path ---
        per_feature_costs = np.zeros(9)
        for pro_idx, user_idx in path:
            # Get the two aligned, scaled (but unweighted) feature vectors
            pro_vec = self.ground_truth_features[pro_idx]
            user_vec = user_serve_9_features[user_idx]
            
            # Calculate squared difference for each feature and accumulate
            squared_diffs = (pro_vec - user_vec) ** 2
            per_feature_costs += squared_diffs

        # --- 4. Convert Per-Feature Costs to Similarity Percentages ---
        # Normalize costs by path length to make them comparable
        path_length = len(path)
        if path_length > 0:
            avg_per_feature_costs = per_feature_costs / path_length
        else:
            avg_per_feature_costs = np.zeros(9)

        # Convert average costs to similarity scores
        # similarity = 1 - (cost / max_cost)
        # Ensure max_cost_baselines don't have zeros to avoid division errors
        safe_max_costs = np.maximum(self.max_cost_baselines, 1e-6)
        
        feature_dissimilarity = avg_per_feature_costs / safe_max_costs
        feature_similarity = 1.0 - feature_dissimilarity
        
        # Clip scores to be within the 0-100 range
        clipped_similarity = np.clip(feature_similarity * 100, 0, 100)
        
        # Create the breakdown dictionary
        breakdown = dict(zip(self.feature_names, clipped_similarity))

        # --- 5. Calculate Overall Calibrated Similarity Score ---
        MIN_DIST = 50.0  # Placeholder!
        MAX_DIST = 150.0 # Placeholder!

        if distance <= MIN_DIST:
            overall_similarity = 100.0
        elif distance >= MAX_DIST:
            overall_similarity = 0.0
        else:
            normalized_dissimilarity = (distance - MIN_DIST) / (MAX_DIST - MIN_DIST)
            overall_similarity = 100 * (1 - normalized_dissimilarity)

        print(f"--- Biomech DTW (9 Features) --- Weighted Distance: {distance:.2f}, Similarity: {overall_similarity:.1f}% ---")

        return overall_similarity, breakdown

# --- 6. On-Screen Display and Counters ---
class ShotCounter:
    def __init__(self):
        self.nb_forehands, self.nb_backhands, self.nb_serves = 0, 0, 0
        self.last_shot = "neutral"
        self.frames_since_last_shot = MIN_FRAMES_BETWEEN_SHOTS
        self.results = []
        self.last_shot_detected_on_this_frame = False # New flag

    def update(self, probs, frame_id):
        self.last_shot_detected_on_this_frame = False # Reset at the start
        shot_detected, shot_type = False, "neutral"
        if self.frames_since_last_shot > MIN_FRAMES_BETWEEN_SHOTS:
            if probs[0] > 0.90: shot_type, shot_detected = "backhand", True; self.nb_backhands += 1
            elif probs[1] > 0.95: shot_type, shot_detected = "forehand", True; self.nb_forehands += 1
            elif len(probs) > 3 and probs[3] > 0.80: shot_type, shot_detected = "serve", True; self.nb_serves += 1

        if shot_detected:
            self.last_shot, self.frames_since_last_shot = shot_type, 0
            self.results.append({"FrameID": frame_id, "Shot": self.last_shot})
            self.last_shot_detected_on_this_frame = True # Set on detection

        self.frames_since_last_shot += 1

    def display(self, frame):
        counters = {"Backhands": (self.nb_backhands, "backhand"), "Forehands": (self.nb_forehands, "forehand"), "Serves": (self.nb_serves, "serve")}
        for i, (name, (count, shot_type)) in enumerate(counters.items()):
            y_pos = frame.shape[0] - 100 + (i * 40)
            color = (0, 255, 0) if (self.last_shot == shot_type and self.frames_since_last_shot < 30) else (0, 0, 255)
            cv2.putText(frame, f"{name} = {count}", (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        return frame

def draw_probs_bars(frame, probs):
    labels, prob_indices = ["S", "B", "N", "F"], [3, 0, 2, 1]
    for i, (label, prob_idx) in enumerate(zip(labels, prob_indices)):
        bar_x = BAR_ORIGIN_X + SPACE_BETWEEN_BARS * i
        if prob_idx < len(probs):
            cv2.putText(frame, label, (TEXT_ORIGIN_X + SPACE_BETWEEN_BARS * i, 230), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            bar_fill_height = int(BAR_HEIGHT * probs[prob_idx])
            cv2.rectangle(frame, (bar_x, BAR_HEIGHT + MARGIN_ABOVE_BAR - bar_fill_height), (bar_x + BAR_WIDTH, BAR_HEIGHT + MARGIN_ABOVE_BAR), (0, 0, 255), -1)
        cv2.rectangle(frame, (bar_x, MARGIN_ABOVE_BAR), (bar_x + BAR_WIDTH, BAR_HEIGHT + MARGIN_ABOVE_BAR), (255, 255, 255), 1)
    return frame

def draw_info_overlay(frame, fps, frame_id, tracking_id, dtw_score=0.0):
    cv2.putText(frame, f"{int(fps)} FPS", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    cv2.putText(frame, f"Frame {frame_id}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
    status_text = f"Tracking ID: {tracking_id}" if tracking_id is not None else "Tracking ID: SEARCHING..."
    color = (0, 255, 0) if tracking_id is not None else (0, 165, 255)
    cv2.putText(frame, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame

def draw_target_player_visuals(frame, player_data, estimator):
    h, w, _ = frame.shape
    box = player_data['box']
    x1, y1, x2, y2 = int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)
    color = (0, 255, 0)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
    display_text = f"Player ID: {player_data['id']} ({player_data['conf']:.2f})"
    cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 1)
    keypoints = player_data['keypoints_norm_conf']
    for p1_idx, p2_idx in estimator.skeleton:
        if keypoints[p1_idx, 2] > 0.3 and keypoints[p2_idx, 2] > 0.3:
            p1 = (int(keypoints[p1_idx, 0] * w), int(keypoints[p1_idx, 1] * h))
            p2 = (int(keypoints[p2_idx, 0] * w), int(keypoints[p2_idx, 1] * h))
            cv2.line(frame, p1, p2, (255, 0, 0), 1)
    for i in range(keypoints.shape[0]):
        if keypoints[i, 2] > 0.3:
            center = (int(keypoints[i, 0] * w), int(keypoints[i, 1] * h))
            cv2.circle(frame, center, 3, (0, 0, 255), -1)
    return frame

def draw_dtw_similarity_widget(frame, score, breakdown={}):
    """
    Draws a similarity widget with an overall score and a per-feature breakdown.
    """
    h, w, _ = frame.shape
    
    # --- Overall Score Bar ---
    widget_w, widget_h = 300, 50
    pos_x = 50
    pos_y = (h - widget_h) // 2 - 100 # Move it up

    # Draw background
    blue_color = (255, 100, 0)
    cv2.rectangle(frame, (pos_x, pos_y), (pos_x + widget_w, pos_y + widget_h), blue_color, -1)

    if score > 0:
        # Draw progress
        progress_w = int(widget_w * (score / 100.0))
        white_color = (255, 255, 255)
        cv2.rectangle(frame, (pos_x, pos_y), (pos_x + progress_w, pos_y + widget_h), white_color, -1)

        # Draw text
        text = f"Overall Similarity: {score:.1f}%"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_x = pos_x + (widget_w - text_size[0]) // 2
        text_y = pos_y + (widget_h + text_size[1]) // 2
        
        # Draw black shadow/outline for text inside the bar
        cv2.putText(frame, text, (text_x + 1, text_y + 1), font, font_scale, (0, 0, 0), font_thickness)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255,255,255) if score < 50 else (0,0,0), font_thickness)

    # --- Per-Feature Breakdown List ---
    if score > 0 and breakdown:
        start_y = pos_y + widget_h + 20
        font_scale = 0.5
        line_height = 25
        
        for i, (name, feature_score) in enumerate(breakdown.items()):
            
            # Background for the feature bar
            bar_pos_y = start_y + i * line_height
            cv2.rectangle(frame, (pos_x, bar_pos_y), (pos_x + widget_w, bar_pos_y + 20), (50, 50, 50), -1)
            
            # Progress for the feature bar
            feature_progress_w = int(widget_w * (feature_score / 100.0))
            # Color coding: Red (<40), Yellow (40-70), Green (>70)
            if feature_score < 40:
                bar_color = (0, 0, 255)
            elif feature_score < 70:
                bar_color = (0, 255, 255)
            else:
                bar_color = (0, 255, 0)
            cv2.rectangle(frame, (pos_x, bar_pos_y), (pos_x + feature_progress_w, bar_pos_y + 20), bar_color, -1)

            # Text for the feature
            feature_text = f"{name}: {feature_score:.0f}%"
            text_size, _ = cv2.getTextSize(feature_text, font, font_scale, font_thickness)
            text_y = bar_pos_y + 15
            
            cv2.putText(frame, feature_text, (pos_x + 5, text_y), font, font_scale, (255, 255, 255), 1)

    return frame

def generate_pro_feedback(overall_score, breakdown):
    """
    Generates a structured, actionable feedback report based on the DTW analysis,
    incorporating kinetic chain concepts and tiered tips.

    Args:
        overall_score (float): The overall similarity score (0-100).
        breakdown (dict): A dictionary mapping feature names to their scores (0-100).

    Returns:
        str: A formatted string containing the feedback.
    """
    # --- 1. Configuration ---
    PRO_TIPS_INDIVIDUAL = {
        "Hip Velocity": {
            "OKAY": "Your leg drive is a bit off. Focus on explosively pushing up with your legs, like you're jumping to reach the ball.",
            "BAD": "Your leg drive is a major power leak. You must focus on bending your knees and driving upwards forcefully. This is fundamental."
        },
        "Knee Angle": {
            "OKAY": "You're not bending your knees enough. Try to 'sit' into your serve more to load up power from your legs.",
            "BAD": "A deep knee bend is non-negotiable for power. You are staying too upright. Practice sitting into a chair and exploding up."
        },
        "Trunk Flexion": {
            "OKAY": "Your trunk tilt needs work. Focus on coiling and uncoiling your upper body to generate rotational power.",
            "BAD": "You are serving with only your arm. Your torso is not coiling and uncoiling. This is a critical break in the kinetic chain."
        },
        "Shoulder Abduction": {
            "OKAY": "Your arm isn't getting high enough. Think about reaching up and extending fully towards the ball at contact.",
            "BAD": "You are not reaching the 'trophy pose.' Your hitting arm must be raised higher to create a powerful swing path."
        },
        "Elbow Angle": {
            "OKAY": "Your elbow angle is limiting your power. Ensure you maintain a good 'L' shape (around 90 degrees) during the trophy pose.",
            "BAD": "Your elbow angle is collapsing, causing a 'pushing' motion. Focus on keeping a distinct 'L' shape in your arm before swinging."
        },
        "Wrist Height": {
            "OKAY": "Your wrist position is affecting the racket drop. Let the racket head drop down your back more to create a whip-like motion.",
            "BAD": "There is almost no racket drop, which is killing your racket head speed. You must let the racket head fall freely down your back."
        },
        "Forearm Swing Path": {
            "OKAY": "Your forearm rotation could be smoother. This is a complex motion, but focus on the feeling of 'snapping' your wrist through contact.",
            "BAD": "You are 'slapping' at the ball instead of pronating. This is a difficult but essential motion for advanced serves. Watch videos on forearm pronation."
        },
        "Torso Rotation": {
            "OKAY": "You're losing power by not rotating your torso enough. Think about turning your shoulders and hips away from the net and then uncoiling explosively.",
            "BAD": "Your body is facing the net throughout the serve. You must turn your side to the net to allow for proper torso rotation."
        },
        "Toss Arm": {
            "OKAY": "Your toss arm timing is disrupting your rhythm. Keep your non-dominant arm up longer to maintain balance and a consistent toss.",
            "BAD": "Your toss arm is dropping immediately, causing a loss of balance and power. Keep it up until you start your swing."
        }
    }

    ENGINE_GROUP = ["Hip Velocity", "Knee Angle", "Trunk Flexion"]
    WHIP_GROUP = ["Shoulder Abduction", "Elbow Angle", "Wrist Height"]
    REFINEMENT_GROUP = ["Forearm Swing Path", "Torso Rotation", "Toss Arm"]

    PRO_TIPS_GROUP = {
        "Engine": {
            "OKAY": "Your main focus should be your 'engine.' Our analysis shows you are losing significant power from your legs and torso. Focus on coiling your body and exploding up and into the court.",
            "BAD": "CRITICAL: Your power foundation is broken. You are not using your legs or torso. Before anything else, you must learn to generate power from the ground up."
        },
        "Whip": {
            "OKAY": "The 'whip' of your arm seems to be the main issue. You're not transferring energy effectively through your shoulder and elbow. Focus on a fluid, relaxed arm motion, letting the racket accelerate naturally.",
            "BAD": "Your arm is acting like a stiff lever, not a whip. This is a major flaw. Focus on relaxing your arm and letting the racket head accelerate through a fluid, whip-like motion."
        },
        "Refinements": {
            "OKAY": "Your core motion is solid, but the 'refinements' need work. Focus on fine-tuning your toss, torso rotation, and forearm snap to add precision and extra pace.",
            "BAD": "While your core motion has some elements, the key refinements are missing, leading to an inconsistent and weak serve. You need to focus on the details of timing and control."
        }
    }

    WORKING_ON_THRESHOLD = 65.0
    BAD_THRESHOLD = 40.0
    WHATS_WORKING_THRESHOLD = 80.0

    # --- 2. Analysis ---
    if not breakdown:
        return "No breakdown available to generate feedback."

    working_on = []
    whats_working = []
    sorted_features = sorted(breakdown.items(), key=lambda item: item[1])
    
    for name, score in sorted_features:
        if score < WORKING_ON_THRESHOLD:
            working_on.append((name, score))
        elif score >= WHATS_WORKING_THRESHOLD:
            whats_working.append((name, score))

    engine_scores = [breakdown[f] for f in ENGINE_GROUP if f in breakdown]
    whip_scores = [breakdown[f] for f in WHIP_GROUP if f in breakdown]
    refinement_scores = [breakdown[f] for f in REFINEMENT_GROUP if f in breakdown]

    avg_engine = np.mean(engine_scores) if engine_scores else 0
    avg_whip = np.mean(whip_scores) if whip_scores else 0
    avg_refinement = np.mean(refinement_scores) if refinement_scores else 0

    group_scores = {
        "Engine": avg_engine,
        "Whip": avg_whip,
        "Refinements": avg_refinement
    }
    
    lowest_group_name = min(group_scores, key=group_scores.get)

    # --- 3. Formatting ---
    feedback_lines = [
        "\n--- COACH'S FEEDBACK ---",
        f"Overall Similarity to Pro: {overall_score:.0f}%\n"
    ]

    feedback_lines.append("--- Individual Feature Performance ---")
    if whats_working:
        feedback_lines.append("What's Working Well:")
        for name, score in sorted(whats_working, key=lambda item: item[1], reverse=True):
            feedback_lines.append(f"  ✅ {name}: {score:.0f}%")
    
    if working_on:
        if whats_working: feedback_lines.append("")
        feedback_lines.append("What to Work On:")
        for name, score in working_on:
            feedback_lines.append(f"  ❌ {name}: {score:.0f}%")
    feedback_lines.append("\n")

    feedback_lines.append("--- Kinetic Chain Analysis ---")
    feedback_lines.append(f"  Engine (Power Generation): {avg_engine:.0f}%")
    feedback_lines.append(f"  Whip (Arm Action): {avg_whip:.0f}%")
    feedback_lines.append(f"  Refinements (Timing & Control): {avg_refinement:.0f}%\n")

    feedback_lines.append("--- Pro-Tips ---")
    if lowest_group_name in PRO_TIPS_GROUP:
        group_score = group_scores[lowest_group_name]
        tip_level = 'BAD' if group_score < BAD_THRESHOLD else 'OKAY'
        feedback_lines.append("Focus Area:")
        feedback_lines.append(f"  \"{PRO_TIPS_GROUP[lowest_group_name][tip_level]}\"")

    if working_on:
        lowest_feature_name, lowest_feature_score = working_on[0]
        if lowest_feature_name in PRO_TIPS_INDIVIDUAL:
            tip_level = 'BAD' if lowest_feature_score < BAD_THRESHOLD else 'OKAY'
            if lowest_group_name in PRO_TIPS_GROUP: feedback_lines.append("")
            feedback_lines.append("Specific Tip:")
            feedback_lines.append(f"  To improve your '{lowest_group_name}', specifically work on your {lowest_feature_name.lower()}.")
            feedback_lines.append(f"  Try this: \"{PRO_TIPS_INDIVIDUAL[lowest_feature_name][tip_level]}\"")

    feedback_lines.append("------------------------\n")
    
    return "\n".join(feedback_lines)

# --- 7. Main Execution ---
def get_box_center(box, frame_dims):
    w, h = frame_dims
    x1, y1, x2, y2 = box
    return np.array([(x1 + x2) * w / 2, (y1 + y2) * h / 2])

def process_video(video_path, yolo_model_path, rnn_model_path, output_dir, start_frame):
    # --- Initializations ---
    pose_estimator = PoseEstimator(yolo_model_path)
    classifier = ShotClassifier(rnn_model_path)
    shot_counter = ShotCounter()
    dtw_comparator = DTWComparator(GROUND_TRUTH_PATH, ENGINEERED_FEATURES_SCALER_PATH)

    # --- Sliding Window Buffers ---
    rnn_feature_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)
    dtw_data_buffer = collections.deque(maxlen=SEQUENCE_LENGTH)

    # --- DTW Score UI Management ---
    current_dtw_score = 0.0
    current_dtw_breakdown = {}
    dtw_score_display_frames = 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}"); return

    frame_width, frame_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps, frame_count = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    downsample_factor = max(1, int(round(video_fps / 30)))
    output_fps = video_fps / downsample_factor

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"processed_{os.path.basename(video_path)}_{timestamp}.mp4"
    output_path = os.path.join(output_dir, output_filename)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), output_fps, (frame_width, frame_height))
    print(f"Processing video with BATCH SIZE={BATCH_SIZE}. Output will be saved to: {output_path}")

    frame_id, prev_time = 0, time.time()
    target_player_id = None
    last_known_position = None
    frames_since_target_lost = 0
    frame_batch = []
    original_frames_in_batch = []

    with tqdm(total=frame_count, desc=f"Analyzing {os.path.basename(video_path)}") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_id += 1
                pbar.update(1)

                if start_frame and frame_id < start_frame: continue
                if downsample_factor > 1 and frame_id % downsample_factor != 0: continue

                original_height, original_width = frame.shape[:2]
                scale_ratio = PROCESSING_WIDTH / original_width
                processing_height = int(original_height * scale_ratio)
                resized_frame = cv2.resize(frame, (PROCESSING_WIDTH, processing_height))
                
                frame_batch.append(resized_frame)
                original_frames_in_batch.append(frame)

            if len(frame_batch) == BATCH_SIZE or (not ret and len(frame_batch) > 0):
                batch_detections = pose_estimator.track_players_in_batch(frame_batch)
                
                for i in range(len(frame_batch)):
                    current_frame_id = frame_id - (len(frame_batch) - 1) + i
                    detected_players = batch_detections[i]
                    annotated_resized_frame = frame_batch[i].copy()
                    target_player_data = None

                    if target_player_id is None:
                        if detected_players:
                            best_player = max(detected_players, key=lambda p: p['conf'])
                            target_player_id = best_player['id']
                            last_known_position = get_box_center(best_player['box'], (PROCESSING_WIDTH, processing_height))
                            target_player_data = best_player
                            print(f"\n--- Player Lock-On ---\nLocked onto Player ID: {target_player_id} at Frame: {current_frame_id}\n----------------------\n")
                    else:
                        found_player_this_frame = False
                        for player in detected_players:
                            if player['id'] == target_player_id:
                                target_player_data = player
                                last_known_position = get_box_center(player['box'], (PROCESSING_WIDTH, processing_height))
                                frames_since_target_lost = 0
                                found_player_this_frame = True
                                break
                        
                        if not found_player_this_frame and frames_since_target_lost < REACQUISITION_GRACE_PERIOD:
                            frames_since_target_lost += 1
                            min_dist = float('inf'); best_candidate = None
                            for player in detected_players:
                                dist = np.linalg.norm(get_box_center(player['box'], (PROCESSING_WIDTH, processing_height)) - last_known_position)
                                if dist < PROXIMITY_THRESHOLD and dist < min_dist:
                                    min_dist = dist; best_candidate = player
                            
                            if best_candidate:
                                print(f"--- Player Re-acquired by Proximity at Frame {current_frame_id} ---")
                                print(f"Old ID: {target_player_id} -> New ID: {best_candidate['id']}\n-------------------------------------------------")
                                target_player_id = best_candidate['id']
                                target_player_data = best_candidate
                                last_known_position = get_box_center(best_candidate['box'], (PROCESSING_WIDTH, processing_height))
                                frames_since_target_lost = 0

                    if target_player_data:
                        draw_target_player_visuals(annotated_resized_frame, target_player_data, pose_estimator)

                    # --- Create Two Separate Feature Streams ---
                    
                    # 1. RNN Stream (0-1 Frame Normalized)
                    # This stream uses the 0-1 normalized data, as the RNN was trained on it.
                    rnn_features = np.zeros(NUM_RNN_FEATURES)
                    if target_player_data:
                        # Use the 0-1 normalized keypoints + confidence
                        kpts_norm_conf = target_player_data['keypoints_norm_conf'].copy()
                        if IS_PLAYER_LEFT_HANDED: kpts_norm_conf[:, 0] = 1 - kpts_norm_conf[:, 0]
                        kpts_norm_conf[kpts_norm_conf[:, 2] < 0.2, :2] = 0 # Zero out low-conf points
                        
                        kps_xy_norm = kpts_norm_conf[:, :2]
                        rnn_features = kps_xy_norm.flatten() # (x,y) order, 26 features
                    
                    rnn_feature_buffer.append(rnn_features)

                    # 2. DTW Stream (Torso-Scaled Normalized)
                    # This stream uses the new, robust normalization for an "apples-to-apples" comparison.
                    kpts_torso_scaled = None
                    if target_player_data:
                        kpts_raw = target_player_data['keypoints_raw'].copy() # (13, 2) raw pixel data
                        kpts_torso_scaled = torso_scale_normalize(kpts_raw, pose_estimator.keypoints_to_keep)

                    if kpts_torso_scaled is not None:
                        if IS_PLAYER_LEFT_HANDED:
                            # Flip the normalized x-coordinates
                            kpts_torso_scaled[:, 0] = -kpts_torso_scaled[:, 0]
                        
                        # DTW Ground truth expects (y,x) order
                        kps_yx_torso_scaled = kpts_torso_scaled[:, [1, 0]]
                        dtw_features = kps_yx_torso_scaled.flatten() # (y,x) order, 26 features
                        last_valid_dtw_features = dtw_features
                    else:
                        # On a bad frame (occlusion, etc.), use the last known good features
                        dtw_features = last_valid_dtw_features
                    
                    dtw_data_buffer.append(dtw_features)

                    if len(rnn_feature_buffer) == SEQUENCE_LENGTH:
                        # Branch A: RNN Prediction
                        rnn_input_sequence = np.array(rnn_feature_buffer, dtype=np.float32)
                        probs = classifier.predict(rnn_input_sequence)
                        shot_counter.update(probs, current_frame_id)

                        # Branch B: DTW Comparison (Triggered by RNN)
                        if shot_counter.last_shot_detected_on_this_frame and shot_counter.last_shot == 'serve':
                            user_serve_window = np.array(dtw_data_buffer)
                            current_dtw_score, current_dtw_breakdown = dtw_comparator.compare_shot(user_serve_window, IS_PLAYER_LEFT_HANDED)
                            dtw_score_display_frames = int(output_fps * 3) # Display for 3 seconds

                            # --- Generate and Print Coach's Feedback ---
                            if current_dtw_breakdown:
                                feedback_text = generate_pro_feedback(current_dtw_score, current_dtw_breakdown)
                                print(feedback_text)

                    # --- UI Updates ---
                    if dtw_score_display_frames > 0:
                        dtw_score_display_frames -= 1
                    else:
                        current_dtw_score = 0.0
                        current_dtw_breakdown = {}

                    final_annotated_frame = cv2.resize(annotated_resized_frame, (original_width, original_height))
                    final_frame = draw_probs_bars(final_annotated_frame, classifier.current_probs)
                    final_frame = shot_counter.display(final_frame)

                    current_time = time.time()
                    fps = BATCH_SIZE / (current_time - prev_time) if (current_time - prev_time) > 0 else 0
                    prev_time = current_time
                    
                    final_frame = draw_info_overlay(final_frame, fps, current_frame_id, target_player_id, current_dtw_score)
                    final_frame = draw_dtw_similarity_widget(final_frame, current_dtw_score, current_dtw_breakdown)
                    out.write(final_frame)

                frame_batch.clear()
                original_frames_in_batch.clear()

            if not ret: break

    cap.release(); out.release(); cv2.destroyAllWindows()
    print("Video processing complete.")
    print("\n--- Detected Shots ---"); print(pd.DataFrame(shot_counter.results))

if __name__ == "__main__":
    setup_tf_gpu()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    process_video(
        video_path=VIDEO_PATH, yolo_model_path=YOLO_MODEL_PATH,
        rnn_model_path=RNN_MODEL_PATH, output_dir=OUTPUT_DIR, start_frame=START_FRAME
    )
