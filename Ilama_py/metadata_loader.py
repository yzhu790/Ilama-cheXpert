# metadata_loader.py
"""Handles loading CheXpert metadata from a CSV file."""

import os
import pandas as pd

def load_metadata(metadata_csv_path, image_dir_root_name="CheXpert-v1.0-small"):
    """
    Loads CheXpert metadata from CSV into a dictionary keyed by a standardized image path.

    Args:
        metadata_csv_path: Path to the metadata CSV (e.g., train.csv).
        image_dir_root_name: The expected root directory name in the 'Path' column
                             (e.g., "CheXpert-v1.0-small"). Used for path standardization.

    Returns:
        dict: Metadata dictionary keyed by relative image path, or None if failed.
    """
    if not metadata_csv_path:
        print("Metadata CSV path not provided. Skipping metadata.")
        return None
    if not os.path.exists(metadata_csv_path):
        print(f"Warning: Metadata CSV not found at {metadata_csv_path}. Skipping metadata.")
        return None

    print(f"Loading metadata from: {metadata_csv_path}")
    try:
        df = pd.read_csv(metadata_csv_path)
        if 'Path' not in df.columns:
             print(f"Error: 'Path' column not found in metadata CSV: {metadata_csv_path}")
             return None

        # Standardize path format (use forward slashes)
        df['KeyPath'] = df['Path'].str.replace('\\', '/')

        # Optional: Add prefix if missing (adjust based on your CSV format)
        # if not df['KeyPath'].iloc[0].startswith(image_dir_root_name + '/'):
        #    print(f"Info: Assuming metadata paths need prefix '{image_dir_root_name}/'.")
        #    df['KeyPath'] = image_dir_root_name + '/' + df['KeyPath']

        metadata_dict = df.set_index('KeyPath').to_dict('index')
        print(f"Loaded metadata for {len(metadata_dict)} entries.")
        return metadata_dict
    except Exception as e:
        print(f"Error loading or processing metadata CSV '{metadata_csv_path}': {e}")
        return None