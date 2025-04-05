# main_pipeline.py
"""Main pipeline script to process CheXpert images using a LLaVA GGUF model."""

import os
import csv
import argparse
from tqdm import tqdm
import pandas as pd
import sys
import traceback

# Import functions and constants from other modules
from constants import SUPPORTED_EXTENSIONS, CSV_COLUMNS
from vlm_handler import initialize_llava_gguf_model, run_inference
from metadata_loader import load_metadata

def find_image_paths(root_dir):
    """Scans the directory structure and returns a list of image paths."""
    print(f"Scanning for images in: {root_dir}")
    image_paths_to_process = []
    parent_dir_name = os.path.basename(os.path.dirname(root_dir))
    base_folder_name = os.path.basename(root_dir)

    if not os.path.isdir(root_dir):
        print(f"Error: Root image directory not found: {root_dir}")
        return []

    for patient in os.listdir(root_dir):
        patient_path = os.path.join(root_dir, patient)
        if not os.path.isdir(patient_path): continue

        for study in os.listdir(patient_path):
            study_path = os.path.join(patient_path, study)
            if not os.path.isdir(study_path): continue

            for img_file in os.listdir(study_path):
                if img_file.lower().endswith(SUPPORTED_EXTENSIONS):
                    full_img_path = os.path.join(study_path, img_file)
                    # Construct the relative path key
                    relative_img_path_parts = [part for part in [parent_dir_name, base_folder_name, patient, study, img_file] if part]
                    relative_img_path = "/".join(relative_img_path_parts)
                    image_paths_to_process.append((full_img_path, relative_img_path))

    print(f"Found {len(image_paths_to_process)} potential image files.")
    return image_paths_to_process

def run_pipeline(args):
    """Runs the main image processing pipeline."""

    # --- 1. Load Metadata (Optional) ---
    image_dir_parent_name = os.path.basename(os.path.dirname(args.image_dir))
    metadata = load_metadata(args.metadata_csv, image_dir_parent_name)

    # --- 2. Initialize Model ---
    llm_model = initialize_llava_gguf_model(
        args.gguf_model_path,
        args.mmproj_path,
        args.n_gpu_layers,
        args.n_ctx,
        args.verbose
    )
    if not llm_model:
        print("Failed to initialize the model. Exiting.")
        sys.exit(1)

    # --- 3. Find Image Paths ---
    image_paths = find_image_paths(args.image_dir)
    if not image_paths:
        print("No images found to process.")
        sys.exit(0)

    # --- 4. Handle Skipping Processed Files ---
    processed_keys = set()
    images_to_run = []
    if args.skip_processed and os.path.exists(args.output_csv):
        try:
            df_existing = pd.read_csv(args.output_csv, keep_default_na=False, usecols=['Path'])
            processed_keys = set(df_existing['Path'])
            print(f"Found {len(processed_keys)} already processed image paths in {args.output_csv}.")
        except Exception as e:
            print(f"Warning: Could not read existing CSV ('{args.output_csv}') to skip processed files: {e}")
            processed_keys = set() # Reset on failure

    if args.skip_processed:
        images_to_run = [(full, rel) for full, rel in image_paths if rel not in processed_keys]
        skipped_count = len(image_paths) - len(images_to_run)
        if skipped_count > 0:
            print(f"Skipping {skipped_count} images found in the output CSV.")
    else:
        images_to_run = image_paths

    if not images_to_run:
        print("No new images to process after checking skipped files.")
        sys.exit(0)

    # --- 5. Process Images and Write CSV ---
    file_mode = 'a' if args.skip_processed and processed_keys else 'w'
    header_needed = file_mode == 'w' or not os.path.exists(args.output_csv) or os.path.getsize(args.output_csv) == 0

    print(f"Processing {len(images_to_run)} images...")
    try:
        with open(args.output_csv, file_mode, newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS, extrasaction='ignore')
            if header_needed:
                writer.writeheader()

            for full_img_path, relative_img_path in tqdm(images_to_run, desc="Processing Images"):
                # Run inference
                diagnosis_dict, raw_output, error_msg = run_inference(
                    llm_model, full_img_path, args.prompt, args.max_tokens
                )

                # Prepare row data
                row = {"Path": relative_img_path, "Raw_VLM_Output": "", "Error_Message": ""} # Defaults

                if error_msg:
                    row["Error_Message"] = error_msg
                    # Fill labels with blank on error
                    row.update({label: "" for label in CSV_COLUMNS if label not in row})
                else:
                    row["Raw_VLM_Output"] = raw_output
                    row.update(diagnosis_dict) # Add the 0/1 labels

                # Add metadata if available
                row["Sex"], row["Age"], row["Frontal/Lateral"], row["AP/PA"] = "", "", "", ""
                if metadata and relative_img_path in metadata:
                    meta = metadata[relative_img_path]
                    row["Sex"] = meta.get("Sex", "")
                    row["Age"] = meta.get("Age", "")
                    row["Frontal/Lateral"] = meta.get("Frontal/Lateral", "")
                    row["AP/PA"] = meta.get("AP/PA", "")

                writer.writerow(row)
                csvfile.flush() # Optional: write periodically

        print(f"\nProcessing finished. Results saved to {args.output_csv}")

    except IOError as e:
        print(f"\nError writing to CSV file {args.output_csv}: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred during CSV writing or processing loop: {e}")
        traceback.print_exc()

# --- Main Execution Guard ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Main pipeline to process CheXpert images using a LLaVA GGUF model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
    # Required arguments
    parser.add_argument("--image_dir", type=str, required=True, help="Directory containing CheXpert patient folders (e.g., '/path/to/CheXpert-v1.0-small/train').")
    parser.add_argument("--gguf_model_path", type=str, required=True, help="Path to the main LLaVA GGUF model file.")
    parser.add_argument("--mmproj_path", type=str, required=True, help="Path to the corresponding LLaVA multimodal projector (mmproj) file.")

    # Optional arguments
    parser.add_argument("--output_csv", type=str, default="chexpert_llava_gguf_reports.csv", help="Path to the output CSV file.")
    parser.add_argument("--metadata_csv", type=str, default=None, help="Path to the CheXpert metadata CSV (e.g., train.csv).")
    parser.add_argument("--prompt", type=str, default="Analyze the provided chest X-ray image. For each of the following 14 findings, state 1 if present and 0 if absent. Output ONLY the list below:\nNo Finding: [0 or 1]\nEnlarged Cardiomediastinum: [0 or 1]\nCardiomegaly: [0 or 1]\nLung Opacity: [0 or 1]\nLung Lesion: [0 or 1]\nEdema: [0 or 1]\nConsolidation: [0 or 1]\nPneumonia: [0 or 1]\nAtelectasis: [0 or 1]\nPneumothorax: [0 or 1]\nPleural Effusion: [0 or 1]\nPleural Other: [0 or 1]\nFracture: [0 or 1]\nSupport Devices: [0 or 1]", help="Text prompt for the VLM. Use \\n for newlines.")
    parser.add_argument("--max_tokens", type=int, default=150, help="Max new tokens for VLM generation.")
    parser.add_argument("--n_gpu_layers", type=int, default=-1, help="Layers to offload to GPU (-1=all, 0=CPU).")
    parser.add_argument("--n_ctx", type=int, default=2048, help="Model context window size.")
    parser.add_argument("--skip_processed", action='store_true', help="Skip images already in output CSV.")
    parser.add_argument("--verbose", action='store_true', help="Enable verbose logging from llama.cpp.")

    cli_args = parser.parse_args()

    # Run the main pipeline function
    run_pipeline(cli_args)

    print("Pipeline execution complete.")