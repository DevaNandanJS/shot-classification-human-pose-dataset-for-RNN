import pandas as pd
from pathlib import Path
import json
from argparse import ArgumentParser

"""
This script normalizes a dataset of biomechanical features using Z-score normalization.

--- PURPOSE ---
To scale the 13-feature dataset so that each feature has a mean of 0 and a standard
deviation of 1. This is a critical preprocessing step for training neural networks,
as it ensures that all features contribute equally to the model's learning process,
preventing features with larger numeric ranges from dominating.

--- LOGIC & WORKING ---
1.  **Find Data:** The script first scans the input directory for all `.csv` files,
    which are assumed to be the output of `features_pose_centric.py`.

2.  **Aggregate Data:** It reads all individual CSV files and concatenates them into a
    single, large pandas DataFrame. This gives a global view of the entire dataset.

3.  **Calculate Global Statistics:** From this aggregated DataFrame, it calculates the
    mean (μ) and standard deviation (σ) for each of the 13 feature columns.

4.  **Save Scaler:** These calculated `mean` and `std` values are saved to a JSON file.
    This "scaler" file is crucial because the same values must be used to normalize
    any future data (e.g., validation, test, or live inference data) to ensure
    consistency.

5.  **Normalize and Save:** The script iterates through the original CSV files one more
    time. For each file, it applies the Z-score formula using the global `mean` and
    `std`: normalized_value = (original_value - mean) / std.

6.  **Export:** Each normalized DataFrame is saved as a new CSV file in the specified
    output directory, preserving the original filename.

--- USAGE ---
Run from the command line, providing paths to the input data, the desired output
folder, and the path to save the scaler file.

`python RNN/normalize_dataset.py --input_dir <path_to_unnormalized_data> \
                             --output_dir <path_to_save_normalized_data> \
                             --scaler_path <path_to_save_scaler.json>`
"""

def normalize_dataset(input_dir, output_dir, scaler_path):
    """
    Calculates global statistics, normalizes the dataset, and saves the results.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    scaler_path = Path(scaler_path)

    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    # Also ensure parent of scaler file exists
    scaler_path.parent.mkdir(parents=True, exist_ok=True)

    csv_files = list(input_path.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_path}. Aborting.")
        return

    print(f"Found {len(csv_files)} CSV files to process.")

    # --- Step 1: Aggregate all data to calculate global stats ---
    all_data = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

    if all_data.empty:
        print("All CSV files are empty. Aborting.")
        return

    # --- Step 2: Calculate global mean and std ---
    global_mean = all_data.mean()
    global_std = all_data.std()

    # Avoid division by zero for features with no variance
    global_std[global_std == 0] = 1.0

    print("Calculated Global Mean:\n", global_mean)
    print("\nCalculated Global Std Dev:\n", global_std)

    # --- Step 3: Save the scaler statistics to a JSON file ---
    scaler_data = {
        "mean": global_mean.to_dict(),
        "std": global_std.to_dict()
    }
    with open(scaler_path, 'w') as f:
        json.dump(scaler_data, f, indent=4)
    print(f"\nScaler (mean/std) saved to {scaler_path}")

    # --- Step 4: Normalize and save each file ---
    print("\nNormalizing and saving individual files...")
    for f in csv_files:
        df = pd.read_csv(f)
        
        # Apply Z-score normalization
        normalized_df = (df - global_mean) / global_std
        
        # Define output path, preserving the original filename
        output_file_path = output_path / f.name
        
        # Save the normalized dataframe
        normalized_df.to_csv(output_file_path, index=False)

    print(f"\nSuccessfully normalized {len(csv_files)} files.")
    print(f"Normalized data saved to: {output_path.resolve()}")


if __name__ == "__main__":
    # Hardcoded paths as per project convention
    # TODO: Manually update these paths before running.
    INPUT_DIR = r"C:\RNN dataset creation\dataset\f1"
    OUTPUT_DIR = r"C:\RNN dataset creation\dataset_normalized\f1"
    SCALER_PATH = r"C:\RNN dataset creation\dataset_normalized\scaler_f1.json"

    # Using ArgumentParser for clarity, though paths are hardcoded
    parser = ArgumentParser(description="Normalize a dataset of tennis shot features.")
    parser.add_argument('--input_dir', type=str, default=INPUT_DIR, help='Directory with unnormalized CSVs.')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Directory to save normalized CSVs.')
    parser.add_argument('--scaler_path', type=str, default=SCALER_PATH, help='Path to save the scaler JSON file.')
    args = parser.parse_args()

    normalize_dataset(args.input_dir, args.output_dir, args.scaler_path)
