import os
import re
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

def analyze_dataset(dataset_path):
    """
    Analyzes the dataset to count the number of samples for each shot type and
    visualizes the distribution as a pie chart.

    Args:
        dataset_path (str): The path to the dataset directory.
    """
    shot_types = []
    # Regex to extract shot type from filename, e.g., 'l_backhand' from 'l_backhand_001.csv'
    shot_type_pattern = re.compile(r'([a-zA-Z_]+)_\d+\.csv')

    # Walk through the dataset directory
    for subdir, _, files in os.walk(dataset_path):
        for filename in files:
            if filename.endswith('.csv'):
                match = shot_type_pattern.match(filename)
                if match:
                    shot_type = match.group(1)
                    shot_types.append(shot_type)

    if not shot_types:
        print("No samples found in the specified directory.")
        return

    # Count the occurrences of each shot type
    shot_counts = Counter(shot_types)

    # Data for the pie chart
    labels = shot_counts.keys()
    sizes = shot_counts.values()

    # Create the pie chart
    fig, ax = plt.subplots(figsize=(10, 8))
    wedges, texts, autotexts = ax.pie(sizes, autopct='%1.1f%%', startangle=90, textprops=dict(color="w"))

    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title("Sample Size Distribution of Tennis Shots")

    # Add a legend with counts
    ax.legend(wedges, [f'{l} ({s})' for l, s in zip(labels, sizes)],
              title="Shot Types",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))

    plt.setp(autotexts, size=8, weight="bold")
    plt.show()


if __name__ == "__main__":
    DATASET_DIRECTORY = 'dataset-non'
    analyze_dataset(DATASET_DIRECTORY)
