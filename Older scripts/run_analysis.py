#!/usr/bin/env python3
# run_analysis.py

import os
import subprocess
import sys

USE_SIMULATED_DATA         = False
USE_FILE_BROWSER_FOR_FOLDER = False
DATA_FOLDER                = "/Users/tomsayada/spectral_analysis_project/data/obs"

PROFILE_TYPE       = 'sym'
FIT_TYPE           = 'standard' # 'standard' or 'ratio'
FIT_BASELINE       = False
USE_WEIGHTED       = True
LINES_TO_FIT       = ['He4026', 'He4388', 'He4471' ]

# lines to fit: ,'He4471' ,'He4026' ,'He4388' ,'H4340' ,'H4101' ,'H3970':


if USE_FILE_BROWSER_FOR_FOLDER:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        print("tkinter not available.")
        sys.exit(1)

def build_output_subfolder_name():
    fit_str = FIT_TYPE.lower()
    prof_str= PROFILE_TYPE.lower()
    w_str   = "weighted" if USE_WEIGHTED else "unweighted"
    b_str   = "baseline" if FIT_BASELINE else "nobaseline"
    return f"{fit_str}_{prof_str}_{w_str}_{b_str}_fit_results"

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

    if USE_SIMULATED_DATA:
        subfolders = []
        for d in os.listdir(data_dir):
            fullpath = os.path.join(data_dir, d)
            if os.path.isdir(fullpath) and d.startswith("simulation_"):
                subfolders.append(d)
        subfolders.sort(key=lambda x: int(x.split("_")[-1]))
        if not subfolders:
            print(f"No simulation_* subfolders found in {data_dir}. Exiting.")
            return

        print(f"Found {len(subfolders)} simulation folders. Running fits on each...")

        for sim_dir in subfolders:
            sim_path = os.path.join(data_dir, sim_dir)
            output_sub = build_output_subfolder_name()
            output_dir = os.path.join(sim_path, output_sub)
            os.makedirs(output_dir, exist_ok=True)

            script_path = os.path.join(os.path.dirname(__file__), fit_script)
            cmd = [
                sys.executable,
                script_path,
                "--data_dir", sim_path,
                "--output_dir", output_dir,
                "--profile_type", PROFILE_TYPE,
                "--lines", lines_arg
            ]
            if not USE_WEIGHTED:
                cmd.append("--unweighted")
            if FIT_BASELINE:
                cmd.append("--fit_baseline")

            print(f"\nRunning {FIT_TYPE} fit for {sim_dir} with profile={PROFILE_TYPE}, "
                  f"{'weighted' if USE_WEIGHTED else 'unweighted'}, baseline={FIT_BASELINE}\n"
                  f"Command: {' '.join(cmd)}")

            try:
                subprocess.run(cmd, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error running fit for {sim_dir}: {e}")

    else:
        print("Running on real data (single folder).")
        output_sub = build_output_subfolder_name()
        output_dir = os.path.join(data_dir, output_sub)
        os.makedirs(output_dir, exist_ok=True)

        script_path = os.path.join(os.path.dirname(__file__), fit_script)
        cmd = [
            sys.executable,
            script_path,
            "--data_dir", data_dir,
            "--output_dir", output_dir,
            "--profile_type", PROFILE_TYPE,
            "--lines", lines_arg
        ]
        if not USE_WEIGHTED:
            cmd.append("--unweighted")
        if FIT_BASELINE:
            cmd.append("--fit_baseline")

        print(f"\nRunning {FIT_TYPE} fit for real data folder: {data_dir}\n"
              f"Command: {' '.join(cmd)}")

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running {FIT_TYPE} fit on real data: {e}")

    print("\nAll done.")

if __name__ == "__main__":
    main()
