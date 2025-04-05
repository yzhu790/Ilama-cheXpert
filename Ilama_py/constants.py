# constants.py
"""Shared constants for the CheXpert VLM processing pipeline."""

# List of the 14 CheXpert competition labels
CHEXPERT_LABELS = [
    "No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly",
    "Lung Opacity", "Lung Lesion", "Edema", "Consolidation", "Pneumonia",
    "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other",
    "Fracture", "Support Devices"
]

# Supported image file extensions
SUPPORTED_EXTENSIONS = ('.jpg', '.jpeg', '.png')

# Column names for the output CSV
CSV_COLUMNS = ["Path", "Sex", "Age", "Frontal/Lateral", "AP/PA"] + CHEXPERT_LABELS + ["Raw_VLM_Output", "Error_Message"]