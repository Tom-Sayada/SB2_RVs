#!/usr/bin/env python3
# run_analysis.py

import os
import sys
import json
import logging
import argparse
import tkinter as tk
from tkinter import filedialog
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
from datetime import datetime
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed

from tqdm import tqdm  # for progress bars

# -------------------------------------------------------------------------
# USER-CONFIGURABLE SETTINGS
# -------------------------------------------------------------------------
USE_SIMULATED_DATA = True
USE_FILE_BROWSER_FOR_FOLDER = True
DATA_FOLDER = "/Users/tomsayada/spectral_analysis_project/data/obs"
PARALLEL_PROCESSING = True

PROFILE_TYPE = 'sym'       # 'sym' or 'asym'
FIT_TYPE = 'standard'      # 'standard' or 'ratio'
FIT_BASELINE = False
USE_WEIGHTED = True
LINES_TO_FIT = ['He4026', 'He4388', 'He4471', 'H4340']

#'He4471', 'He4026', 'He4388', 'H4340'
# -------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('spectral_analysis.log')]
)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for spectral analysis"""
    data_folder: str
    use_simulated: bool = True
    profile_type: str = 'sym'
    fit_type: str = 'standard'
    fit_baseline: bool = False
    use_weighted: bool = True
    parallel_processing: bool = True
    max_workers: int = 4
    save_intermediates: bool = True

    # Lines to fit with optional weights
    lines_to_fit: Dict[str, float] = None

    # Retry logic if needed
    retry_failed: bool = True
    max_retries: int = 3

    def __post_init__(self):
        # Provide default line weights if user didn't specify them
        if self.lines_to_fit is None:
            self.lines_to_fit = {
                'He4026': 1.0,
                'He4388': 1.0,
                'He4471': 1.0,
                'H4340': 0.7
            }


class AnalysisError(Exception):
    """Custom exception for analysis errors"""
    pass


def get_data_folder() -> Optional[str]:
    """
    Prompt the user to select a folder with tkinter (if USE_FILE_BROWSER_FOR_FOLDER=True).
    Otherwise, return DATA_FOLDER directly.
    """
    if not USE_FILE_BROWSER_FOR_FOLDER:
        logger.info("File browser disabled => using DATA_FOLDER directly")
        return DATA_FOLDER

    try:
        root = tk.Tk()
        root.withdraw()
        logger.info("Select the parent data folder:")
        chosen_dir = filedialog.askdirectory(title="Select Data Folder")
        root.destroy()

        if not chosen_dir:
            logger.error("No folder selected. Exiting.")
            return None
        logger.info(f"User selected folder: {chosen_dir}")
        return chosen_dir

    except Exception as e:
        logger.error(f"Error with file browser: {e}")
        return None


def find_simulation_folders(parent_dir: str) -> List[str]:
    """
    Find subfolders named 'simulation_*' in parent_dir. Sort them by numeric suffix.
    """
    subfolders = []
    try:
        for d in os.listdir(parent_dir):
            fullpath = os.path.join(parent_dir, d)
            if os.path.isdir(fullpath) and d.startswith("simulation_"):
                subfolders.append(fullpath)
        # Sort by numeric suffix if it exists
        subfolders.sort(key=lambda x: int(os.path.basename(x).split("_")[-1]))
        return subfolders
    except Exception as e:
        logger.error(f"Error finding simulation folders in {parent_dir}: {e}")
        return []


def validate_data_folder(folder: str) -> bool:
    """Check that the folder has some observation files to process."""
    try:
        if not os.path.isdir(folder):
            logger.error(f"Folder not found: {folder}")
            return False

        files = os.listdir(folder)
        obs_files = [
            f for f in files
            if (f.startswith('obs_') or 'spec_' in f or '_V1_' in f or '_V2_' in f)
        ]

        if not obs_files:
            logger.warning(f"No observation files found in {folder}")
            return False

        logger.info(f"Found {len(obs_files)} observation files in {folder}")
        return True

    except Exception as e:
        logger.error(f"Error validating folder {folder}: {e}")
        return False


def run_in_parallel(folders: List[str], config: 'AnalysisConfig') -> Dict[str, Tuple[bool, str]]:
    """
    Run the analysis in parallel for all folders.
    Show a TQDM progress bar that increments as each folder finishes.
    """
    results = {}
    pbar = tqdm(total=len(folders), desc="Fitting folders in parallel", position=0)

    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        future_map = {
            executor.submit(run_analysis_for_folder, f, config): f
            for f in folders
        }
        for future in as_completed(future_map):
            folder = future_map[future]
            try:
                ok, msg = future.result()
                results[folder] = (ok, msg)
            except Exception as e:
                results[folder] = (False, str(e))
            pbar.update(1)

    pbar.close()
    return results


def run_sequential(folders: List[str], config: 'AnalysisConfig') -> Dict[str, Tuple[bool, str]]:
    """
    Run the analysis sequentially for each folder, with a TQDM progress bar.
    """
    results = {}
    with tqdm(total=len(folders), desc="Fitting folders sequentially", position=0) as pbar:
        for folder in folders:
            ok, msg = run_analysis_for_folder(folder, config)
            results[folder] = (ok, msg)
            pbar.update(1)
    return results


def build_output_subfolder_name(
    fit_type: str,
    profile_type: str,
    use_weighted: bool,
    fit_baseline: bool
) -> str:
    """
    Build a standardized output subfolder name for saving fit results.
    E.g. 'standard_sym_weighted_nobaseline_fit_results'
    """
    w_str = "weighted" if use_weighted else "unweighted"
    b_str = "baseline" if fit_baseline else "nobaseline"
    return f"{fit_type}_{profile_type}_{w_str}_{b_str}_fit_results"


def run_analysis_for_folder(folder: str, config: 'AnalysisConfig', retry_count: int = 0) -> Tuple[bool, str]:
    """
    Actually call run_standard_fit.py or run_ratio_constrained_fit.py for the given folder.
    Return (success, message).
    """
    # Decide which script to run based on config
    if config.fit_type.lower() == 'standard':
        script_name = "run_standard_fit.py"
    else:
        script_name = "run_ratio_constrained_fit.py"

    # Quick check for obs files
    if not validate_data_folder(folder):
        return False, f"No valid obs files in {folder}"

    # Create subfolder for results inside 'folder'
    subfolder_name = build_output_subfolder_name(
        fit_type=config.fit_type.lower(),
        profile_type=config.profile_type.lower(),
        use_weighted=config.use_weighted,
        fit_baseline=config.fit_baseline
    )
    output_dir = os.path.join(folder, subfolder_name)
    os.makedirs(output_dir, exist_ok=True)

    # Build lines argument
    lines_arg = ",".join(config.lines_to_fit.keys()) if config.lines_to_fit else ""

    # Prepare the command
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), script_name),
        "--data_dir", folder,
        "--output_dir", output_dir,
        "--profile_type", config.profile_type,
        "--lines", lines_arg
    ]
    if not config.use_weighted:
        cmd.append("--unweighted")
    if config.fit_baseline:
        cmd.append("--fit_baseline")

    logger.info(f"Running command: {' '.join(cmd)}")

    try:
        # Removed capture_output=True so we see real-time logs
        completed = subprocess.run(
            cmd,
            text=True,
            check=False
        )
        # If you want to see stdout/stderr after it finishes:
        # logger.debug(f"STDOUT:\n{completed.stdout}")
        # logger.debug(f"STDERR:\n{completed.stderr}")

        if completed.returncode != 0:
            err_msg = f"{script_name} failed (rc={completed.returncode})"
            logger.error(err_msg)

            if config.retry_failed and (retry_count < config.max_retries):
                logger.info(f"Retrying {folder} (attempt {retry_count + 1})")
                return run_analysis_for_folder(folder, config, retry_count + 1)

            return False, err_msg
        else:
            logger.info(f"Analysis complete for {folder}. Results in: {output_dir}")
            return True, ""

    except Exception as e:
        logger.exception(f"Error running {script_name} on {folder}: {e}")
        return False, str(e)


def setup_output_folders_and_data(config: AnalysisConfig) -> Tuple[str, List[str], str]:
    """
    1) Open file browser to pick a folder (or just use config.data_folder if disabled).
    2) If use_simulated => gather simulation_* subfolders
    3) Else => treat the chosen folder as the only data folder
    4) Create a base output folder named analysis_results_<timestamp>
       under the chosen folder (for logs or overall summary).
    5) Return (base_output, list_of_folders, timestamp)
    """
    chosen_dir = get_data_folder()
    if not chosen_dir:
        raise AnalysisError("No folder chosen or error with file browser.")

    config.data_folder = os.path.abspath(chosen_dir)
    logger.info(f"Data folder set to: {config.data_folder}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output = os.path.join(config.data_folder, f"analysis_results_{timestamp}")
    os.makedirs(base_output, exist_ok=True)
    logger.info(f"Base output: {base_output}")

    if config.use_simulated:
        sim_folders = find_simulation_folders(config.data_folder)
        if not sim_folders:
            raise AnalysisError(f"No simulation_* folders found in {config.data_folder}")
        return base_output, sim_folders, timestamp
    else:
        # Real data => single folder
        return base_output, [config.data_folder], timestamp


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full integrated run_analysis for SB2 project.")
    parser.add_argument("--workers", type=int, default=4, help="Max worker processes")
    return parser.parse_args()


def main() -> int:
    try:
        args = parse_arguments()

        # Build config from user settings at top
        config = AnalysisConfig(
            data_folder=DATA_FOLDER,
            use_simulated=USE_SIMULATED_DATA,
            profile_type=PROFILE_TYPE,
            fit_type=FIT_TYPE,
            fit_baseline=FIT_BASELINE,
            use_weighted=USE_WEIGHTED,
            parallel_processing=PARALLEL_PROCESSING,
            max_workers=args.workers or 4,
            save_intermediates=True,
            lines_to_fit={ln: 1.0 for ln in LINES_TO_FIT}  # assign weight=1
        )

        base_output, folders, timestamp = setup_output_folders_and_data(config)

        if config.parallel_processing and len(folders) > 1:
            results = run_in_parallel(folders, config)
        else:
            results = run_sequential(folders, config)

        total = len(folders)
        success_count = sum(1 for (ok, _) in results.values() if ok)
        logger.info(f"Analysis complete => {success_count}/{total} successful")

        # Summarize
        summary_json = os.path.join(base_output, "analysis_summary.json")
        summary = {
            "config": {
                "fit_type": config.fit_type,
                "profile_type": config.profile_type,
                "baseline": config.fit_baseline,
                "weighted": config.use_weighted,
                "lines": list(config.lines_to_fit.keys()),
                "parallel": config.parallel_processing,
            },
            "timestamp": timestamp,
            "results": {}
        }
        for folder, (ok, msg) in results.items():
            summary["results"][folder] = {
                "success": ok,
                "error_message": msg
            }
        with open(summary_json, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary: {summary_json}")

        if success_count < total:
            logger.warning("Some folders failed; check logs or analysis_summary.json")
            return 1
        return 0

    except KeyboardInterrupt:
        logger.info("\nAnalysis interrupted by user")
        return 130
    except AnalysisError as e:
        logger.error(f"AnalysisError: {e}")
        return 1
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())