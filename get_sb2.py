import os
import shutil

# Source and target directories
source_dir = "/Users/tomsayada/PycharmProjects/BLOeM/source/Bdwarfs/extracted_from_fields"
target_dir = "/Users/tomsayada/spectral_analysis_project/src/SB2_data"

os.makedirs(target_dir, exist_ok=True)

# The list of folder names (codes) to find and copy
codes = [
    "6-062", "5-063", "6-038", "8-080", "5-037", "5-062", "6-055", "6-037",
    "4-062", "2-064", "2-052", "1-037", "5-111", "1-055", "1-083", "3-079",
    "5-013", "2-103", "6-049", "7-057", "4-056", "4-005", "4-059", "4-095",
    "8-025", "6-115", "3-067", "3-072", "3-086", "3-110", "1-088", "5-100",
    "6-042", "6-016", "5-080", "5-027", "6-026"
]

for code in codes:
    source_folder = os.path.join(source_dir, code)
    if os.path.isdir(source_folder):
        # Destination path
        dest_folder = os.path.join(target_dir, code)
        # Copy the entire folder tree
        shutil.copytree(source_folder, dest_folder, dirs_exist_ok=True)
        print(f"Copied {source_folder} to {dest_folder}")
    else:
        print(f"{code} not found in {source_dir}")
