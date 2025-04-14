#!/usr/bin/env python3
# run_analysis_weighted.py

"""
run_analysis_weighted.py

Purpose:
    - Mirrors your original run_analysis.py pipeline:
      1) Optionally use tkinter to pick the parent data folder
      2) Find subfolders that match e.g. 4-003 or simulation_*
      3) Run in parallel fits for each subfolder
      4) Use whichever lines, profile_type, baseline, etc. you specify
      5) BUT calls the new "weighted" fit scripts:
         - run_standard_fit_weighted.py
         - run_ratio_constrained_fit_weighted.py
        instead of the original ones.

Usage:
    python run_analysis_weighted.py
    (or run it inside an IDE; you'll get the same UI prompt for the folder)
"""

import os
import sys
import subprocess
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

# Attempt tkinter import for file browser
USE_FILE_BROWSER_FOR_FOLDER = True
try:
    import tkinter as tk
    from tkinter import filedialog
except ImportError:
    USE_FILE_BROWSER_FOR_FOLDER = False
    print("Warning: tkinter not available, can't show folder dialog.")


###############################################################################
# User-configurable settings (just like your original run_analysis.py)
###############################################################################
DATA_FOLDER = "/Users/tomsayada/spectral_analysis_project/data/obs"

PROFILE_TYPE  = 'sym'      # e.g. 'sym' or 'asym'
LINE_PROFILE  = 'voigt'    # 'voigt' or 'gaussian'
FIT_TYPE      = 'standard' # 'standard' or 'ratio'
FIT_BASELINE  = False
USE_WEIGHTED  = True
USE_MCMC      = True

# Your default lines (change or remove if you want them from command line)
LINES_TO_FIT  = ['He4026', 'He4388', 'He4471', 'H4340']

# Separation-based weighting parameters
MIN_WEIGHT = 0.2      # Minimum weight for near-blended epochs (0.0-1.0)
MAX_SEP = 400.0       # RV separation in km/s at which weight reaches 1.0
MAX_ITERATIONS = 3    # Maximum number of weighting iterations

###############################################################################
def build_output_subfolder_name():
    """
    Builds a string like: 'standard_sym_voigt_sepWeighted_nobaseline_fit_results'
    """
    fit_str  = FIT_TYPE.lower()
    prof_str = PROFILE_TYPE.lower()
    line_str = LINE_PROFILE.lower()

    # Use "sepWeighted" to indicate the new approach
    w_str    = "sepWeighted" if USE_WEIGHTED else "unweighted"
    b_str    = "baseline" if FIT_BASELINE else "nobaseline"

    return f"{fit_str}_{prof_str}_{line_str}_{w_str}_{b_str}_fit_results"


def run_fit_on_folder(folder_path, output_dir,
                      fit_script, # e.g. run_standard_fit_weighted.py
                      profile_type, line_profile, lines_arg,
                      use_weighted, fit_baseline, use_mcmc,
                      min_weight, max_sep, max_iterations):
    """
    Runs the chosen fit script in a subprocess for a single subfolder.
    Includes new parameters for separation-based weighting.
    """
    os.makedirs(output_dir, exist_ok=True)

    script_path = os.path.join(os.path.dirname(__file__), fit_script)
    cmd = [
        sys.executable,
        script_path,
        "--data_dir", folder_path,
        "--output_dir", output_dir,
        "--profile_type", profile_type,
        "--line_profile", line_profile,
        "--lines", lines_arg,
        "--min_weight", str(min_weight),
        "--max_sep", str(max_sep),
        "--max_iterations", str(max_iterations)
    ]
    if not use_weighted:
        cmd.append("--unweighted")
    if fit_baseline:
        cmd.append("--fit_baseline")
    if use_mcmc:
        cmd.append("--mcmc")

    print(f"\n[Weighted Analysis] Running {fit_script} for '{folder_path}'\nCommand: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True)
        print(f"Finished: {folder_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running fit for {folder_path}: {e}")


def main():
    # Step A: Possibly use a file browser to choose data_dir
    if USE_FILE_BROWSER_FOR_FOLDER:
        root = tk.Tk()
        root.withdraw()
        print("Select the parent data folder:")
        chosen_dir = filedialog.askdirectory(title="Select Data Folder")
        root.destroy()
        if not chosen_dir:
            print("No folder selected. Exiting.")
            return
        data_dir = chosen_dir
    else:
        data_dir = DATA_FOLDER

    if not os.path.isdir(data_dir):
        print(f"Data folder not found: {data_dir}")
        return

    # Step B: Decide which fit script to call (the new weighted versions)
    if FIT_TYPE.lower() == 'standard':
        fit_script = "run_standard_fit_weighted.py"
    else:
        fit_script = "run_ratio_constrained_fit_weighted.py"

    # Build lines arg
    lines_arg = ",".join(LINES_TO_FIT)

    # Build output subfolder name
    output_sub = build_output_subfolder_name()

    # Step C: Identify subfolders matching pattern or no subfolders => single folder
    subfolders = []
    for entry in os.listdir(data_dir):
        fullpath = os.path.join(data_dir, entry)
        if not os.path.isdir(fullpath):
            continue

        # (1) Check if it starts with simulation_
        # (2) or matches ^(\d+)-(\d{3})$
        if entry.startswith("simulation_"):
            subfolders.append(entry)
        else:
            if re.match(r'^\d+-\d{3}$', entry):
                subfolders.append(entry)

    if subfolders:
        print(f"Found {len(subfolders)} matching subfolders (simulation_* or digit-digit pattern).")
        # parallel with ProcessPoolExecutor
        tasks = []
        max_workers = min(4, len(subfolders))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for subf in subfolders:
                folder_path = os.path.join(data_dir, subf)
                output_dir  = os.path.join(folder_path, output_sub)
                future = executor.submit(
                    run_fit_on_folder,
                    folder_path,
                    output_dir,
                    fit_script,
                    PROFILE_TYPE,
                    LINE_PROFILE,
                    lines_arg,
                    USE_WEIGHTED,
                    FIT_BASELINE,
                    USE_MCMC,
                    MIN_WEIGHT,
                    MAX_SEP,
                    MAX_ITERATIONS
                )
                tasks.append(future)

            for future in as_completed(tasks):
                try:
                    future.result()
                except Exception as exc:
                    print(f"Parallel fit job failed: {exc}")

        print("\nAll parallel runs completed.\n")

    else:
        # No matching subfolders => single folder
        print("Running single-folder fit (no subfolders found).")
        output_dir = os.path.join(data_dir, output_sub)

        run_fit_on_folder(
            data_dir,
            output_dir,
            fit_script,
            PROFILE_TYPE,
            LINE_PROFILE,
            lines_arg,
            USE_WEIGHTED,
            FIT_BASELINE,
            USE_MCMC,
            MIN_WEIGHT,
            MAX_SEP,
            MAX_ITERATIONS
        )

        print("\nAll done with single-folder fit.\n")


if __name__ == "__main__":
    main()