#!/usr/bin/env python3

import os
import shutil

# Base directory containing FIELD# folders
source_base = "/Users/tomsayada/spectral_analysis_project/BLOeM_DR4.0_Combined"

# Destination directory for the consolidated FITS files
destination_base = "/Users/tomsayada/spectral_analysis_project/data/BLOEM_ALL"

# Create the destination folder if it doesn't exist
os.makedirs(destination_base, exist_ok=True)

# Loop over FIELD1 through FIELD8
for field_num in range(1, 9):
    # Example: /Users/.../BLOeM_DR4.0_Combined/FIELD1/FITS
    fits_dir = os.path.join(source_base, f"FIELD{field_num}", "FITS")

    if not os.path.isdir(fits_dir):
        print(f"Skipping {fits_dir} (not a directory).")
        continue

    # List all files in that directory
    for filename in os.listdir(fits_dir):
        # Check for .fits files only
        if not filename.endswith(".fits"):
            continue

        # Full path of the source file
        src_path = os.path.join(fits_dir, filename)

        # Example filename: "BLOeM_1-003_01_Combined.fits"
        # Remove extension, then split on underscores
        base_name = os.path.splitext(filename)[0]  # e.g. "BLOeM_1-003_01_Combined"
        parts = base_name.split("_")  # e.g. ["BLOeM", "1-003", "01", "Combined"]

        if len(parts) < 2:
            print(f"Skipping {filename} (unexpected name format).")
            continue

        # The second part should look like "1-003"
        field_star_part = parts[1]  # "1-003"

        # Split that on dash
        if "-" not in field_star_part:
            print(f"Skipping {filename} (no dash in {field_star_part}).")
            continue

        field_str, star_str = field_star_part.split("-", 1)  # e.g. ["1", "003"]

        # Construct output folder name, e.g. "1_003"
        out_folder_name = f"{field_str}_{star_str}"
        out_folder_path = os.path.join(destination_base, out_folder_name)

        # Create the folder if needed
        os.makedirs(out_folder_path, exist_ok=True)

        # Destination path
        dst_path = os.path.join(out_folder_path, filename)

        # Copy the file
        shutil.copy2(src_path, dst_path)
        print(f"Copied {src_path} -> {dst_path}")

print("Done organizing FITS files.")
