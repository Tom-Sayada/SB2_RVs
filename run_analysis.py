#!/usr/bin/env python3
# run_analysis.py

import os
import sys
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

USE_FILE_BROWSER_FOR_FOLDER = True
DATA_FOLDER                = "/Users/tomsayada/spectral_analysis_project/data/obs"

PROFILE_TYPE  = 'sym'         # e.g. 'sym' or 'asym'
LINE_PROFILE  = 'voigt'       # 'voigt' or 'gaussian'
FIT_TYPE      = 'ratio'    # 'standard' or 'ratio'
FIT_BASELINE  = False
USE_WEIGHTED  = True
USE_MCMC      = False         # <-- Toggle for MCMC. If True, pass --mcmc to the child script.

LINES_TO_FIT  = ['He4026', 'He4388', 'He4471', 'H4340']

# choose from 'He4471', 'He4026', 'He4388', 'H4340', 'H4101'
# adding more lines requires addition to Example lines in run_standard_fit (line 119) and (line 180) for run_ratio_constrained_fit
# if no lines are chosen, the fit is run on all the lines in the example lines

if USE_FILE_BROWSER_FOR_FOLDER:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("tkinter not available.")
        sys.exit(1)


def build_output_subfolder_name():
    fit_str = FIT_TYPE.lower()
    prof_str = PROFILE_TYPE.lower()
    line_str = LINE_PROFILE.lower()
    w_str = "weighted" if USE_WEIGHTED else "unweighted"
    b_str = "baseline" if FIT_BASELINE else "nobaseline"
    return f"{fit_str}_{prof_str}_{line_str}_{w_str}_{b_str}_fit_results"


def run_fit_on_folder(folder_path, output_dir, fit_script,
                      profile_type, line_profile, lines_arg,
                      use_weighted, fit_baseline, use_mcmc):
    """Run fitting script on a single folder"""
    os.makedirs(output_dir, exist_ok=True)
    script_path = os.path.join(os.path.dirname(__file__), fit_script)

    cmd = [
        sys.executable,
        script_path,
        "--data_dir", folder_path,
        "--output_dir", output_dir,
        "--profile_type", profile_type,
        "--line_profile", line_profile,
        "--lines", lines_arg
    ]
    if not use_weighted:
        cmd.append("--unweighted")
    if fit_baseline:
        cmd.append("--fit_baseline")
    if use_mcmc:
        cmd.append("--mcmc")

    print(f"\nRunning {fit_script} for '{folder_path}' with profile={profile_type}, "
          f"line_profile={line_profile}, "
          f"{'weighted' if use_weighted else 'unweighted'}, "
          f"baseline={fit_baseline}, mcmc={use_mcmc}\n"
          f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
        print(f"Finished: {folder_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running fit for {folder_path}: {e}")
def main():
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

    if FIT_TYPE.lower() == 'standard':
        fit_script = "run_standard_fit.py"
    else:
        fit_script = "run_ratio_constrained_fit.py"

    lines_arg = ",".join(LINES_TO_FIT)

    # We'll look for subfolders named either "simulation_*" or "F-..."
    subfolders = []
    for entry in os.listdir(data_dir):
        fullpath = os.path.join(data_dir, entry)
        if os.path.isdir(fullpath) and (entry.startswith("simulation_") or entry.startswith("F-")):
            subfolders.append(entry)

    if subfolders:
        print(f"Found {len(subfolders)} matching subfolders (simulation_* or F-*).")

        def extract_num(s):
            import re
            m = re.search(r'(\d+)$', s)
            return int(m.group(1)) if m else 999999

        subfolders.sort(key=extract_num)
        print("Will process in parallel:", subfolders)

        tasks = []
        max_workers = min(4, len(subfolders))  # up to 4 or #subfolders, whichever is smaller
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            for subf in subfolders:
                folder_path = os.path.join(data_dir, subf)
                output_sub = build_output_subfolder_name()
                output_dir = os.path.join(folder_path, output_sub)

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
                    USE_MCMC
                )
                tasks.append(future)

            for future in as_completed(tasks):
                try:
                    future.result()
                except Exception as exc:
                    print(f"Parallel fit job failed: {exc}")

        print("\nAll parallel runs completed.\n")

    else:
        # no matching subfolders => single folder fit
        print("Running single-folder fit (no subfolders found).")
        output_sub = build_output_subfolder_name()
        output_dir = os.path.join(data_dir, output_sub)

        run_fit_on_folder(
            folder_path=data_dir,
            output_dir=output_dir,
            fit_script=fit_script,
            profile_type=PROFILE_TYPE,
            line_profile=LINE_PROFILE,
            lines_arg=lines_arg,
            use_weighted=USE_WEIGHTED,
            fit_baseline=FIT_BASELINE,
            use_mcmc=USE_MCMC
        )

        print("\nAll done with single-folder fit.\n")


if __name__ == "__main__":
    main()