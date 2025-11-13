import os
import re
import subprocess
import sys
from tqdm import tqdm

def convert_video_fps(input_path, output_path, target_fps):
    """
    Converts a video file to a specified frame rate using ffmpeg, displaying a progress bar.
    """
    # Command to get video duration using ffprobe
    ffprobe_command = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        input_path
    ]

    try:
        # Get total duration
        duration_str = subprocess.check_output(ffprobe_command).decode('utf-8').strip()
        total_duration = float(duration_str)
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print("Error: ffprobe not found or failed to get video duration.")
        print("Please ensure ffmpeg (which includes ffprobe) is installed and in your PATH.")
        sys.exit(1)

    # Overwrite output file if it exists, and run quietly but report progress
    command = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-i', input_path,
        '-filter:v', f'fps={target_fps}',
        '-progress', 'pipe:1', # Pipe progress to stdout
        '-nostats', # Disable default stats output
        output_path
    ]
    
    print(f"Converting '{os.path.basename(input_path)}' to {target_fps}fps...")

    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True) as process:
        with tqdm(total=total_duration, unit='s', unit_scale=True, desc="Converting") as pbar:
            for line in process.stdout:
                if "out_time_ms" in line:
                    time_in_us = int(line.strip().split("=")[1])
                    current_time = time_in_us / 1_000_000
                    pbar.update(current_time - pbar.n) # Update to current time
    
    if process.returncode != 0:
        print(f"Error during conversion. FFmpeg returned code {process.returncode}")
    else:
        print(f"Successfully converted to '{output_path}'")

def main():
    # --- USER CONFIGURATION ---
    # Please update these paths to your actual input and output files.
    input_file = r"C:\RNN dataset creation\input\Yoshihito Nishioka Japan's Current ATP No.1 2023 Practice + Slow-mo Technique   Court Level 4K 60FPS.mp4"  # Raw 60fps video
    output_file = r"C:\RNN dataset creation\input\left-handed\f5.mp4" # Converted 30fps video
    target_fps = 30
    # --------------------------

    if not os.path.isfile(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)

    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    convert_video_fps(input_file, output_file, target_fps)

if __name__ == "__main__":
    main()
