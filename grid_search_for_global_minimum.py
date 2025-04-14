#!/usr/bin/env python3

"""
grid_search_global_min.py

A fully working example that:
1) Prompts for a parent folder (containing e.g. F-059 or simulation_29).
2) Prompts for Weighted or Unweighted mode.
3) For each selected folder, locates the standard_fit_params.json in
   standard_sym_voigt_{weighted|unweighted}_nobaseline_auto_fit_results.
4) Loads data arrays for multiple lines/epochs.
5) For each epoch, does a coarse grid search.
6) Optionally does a fine grid search on promising results.
7) Plots the χ² landscape and checks for improvement.
8) If improvements exist, refines the fit using refine_fit(...).
9) Summarizes results, including improvement in reduced χ².
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import re
import logging
import time
import matplotlib.pyplot as plt
import argparse

from lmfit import Parameters, Minimizer
from matplotlib.colors import LogNorm

# ================================================================
# USER CONFIGURATION - EDIT THESE SETTINGS
# ================================================================

SKIP_FINE_GRID_SEARCH = True
DEBUG_MODE = False

# Spectral line definitions
SPECTRAL_LINES = {
        'He4471': {'rest_wave': 4471.5, 'window': 11.0},
        'He4026': {'rest_wave': 4026.0, 'window': 25.0},
        'He4388': {'rest_wave': 4388.0, 'window': 13.0},
        'H4340':  {'rest_wave': 4340.472, 'window': 30.0},
        'H4101':  {'rest_wave': 4101,     'window': 22.0},
        'He4144': {'rest_wave': 4144, 'window': 10.0},
        'He4120': {'rest_wave': 4120, 'window': 13.0}
    }

# ================================================================
# END OF USER CONFIGURATION
# ================================================================

# local imports
try:
    from src.utils import (
        find_line_center_smoothed,
        find_noise_regions,
        find_observation_files,
        load_data_for_epoch
    )
    from src.model_builder import (
        residuals as standard_residuals,
        compute_full_model,
        get_star_components
    )
    from src.plot_results import report_fit_results
except ImportError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)
    sys.path.append(parent_dir)

    from src.utils import (
        find_line_center_smoothed,
        find_noise_regions,
        find_observation_files,
        load_data_for_epoch
    )
    from src.model_builder import (
        residuals as standard_residuals,
        compute_full_model,
        get_star_components
    )
    from src.plot_results import report_fit_results

logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def select_parent_folder():
    """Open a file dialog to select the parent folder."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    folder_path = filedialog.askdirectory(
        title="Select Parent Folder with subfolders (e.g. F-059 or simulation_29)"
    )
    root.destroy()
    if not folder_path:
        print("No folder selected. Exiting.")
        sys.exit(1)
    return folder_path


def ask_for_weighting():
    """
    Prompt the user to choose Weighted or Unweighted mode.
    Returns: 'weighted' or 'unweighted'
    """
    while True:
        choice = input("Select fitting mode: [W]eighted or [U]nweighted? ").strip().lower()
        if choice.startswith('w'):
            return 'weighted'
        elif choice.startswith('u'):
            return 'unweighted'
        print("Invalid input. Please enter 'w' or 'u'.")


def find_candidate_folders(parent_folder):
    """
    Returns a list of full paths to subfolders that match one of these patterns:
    - <digits>-<digits> (e.g. 4-059)
    - F-<digits> (e.g. F-123)
    - simulation_<digits> (e.g. simulation_29)
    """
    subfolders = []
    pattern = re.compile(r'^(?:\d+-\d+|F-\d+|simulation_\d+)$', re.IGNORECASE)
    for entry in os.listdir(parent_folder):
        fullp = os.path.join(parent_folder, entry)
        if os.path.isdir(fullp) and pattern.match(entry):
            subfolders.append(fullp)
    subfolders.sort()
    return subfolders


def find_standard_params(folder_path, weighting_mode):
    """
    Look for 'standard_fit_params.json' in:
        standard_sym_voigt_{weighting_mode}_nobaseline_auto_fit_results
    Return the JSON if found, else None.
    """
    sub_dir = f"standard_sym_voigt_{weighting_mode}_nobaseline_auto_fit_results"
    candidate_dir = os.path.join(folder_path, sub_dir)
    if os.path.isdir(candidate_dir):
        candidate_json = os.path.join(candidate_dir, "standard_fit_params.json")
        if os.path.isfile(candidate_json):
            return candidate_json
    return None


def load_params(json_file):
    """Load and return an lmfit Parameters object from JSON."""
    with open(json_file, 'r') as f:
        data = json.load(f)
    params = Parameters()
    for name, details in data.items():
        val = details.get("value", 0.0)
        pmin = details.get("min", None)
        pmax = details.get("max", None)
        params.add(name, value=val, min=pmin, max=pmax)
    return params


def calculate_chi_square(params, wv_dict, fl_dict, ep_dict, un_dict, central_wv, **kwargs):
    """Calculate chi-square from residuals."""
    resids = standard_residuals(params, wv_dict, fl_dict, ep_dict, un_dict, central_wv, **kwargs)
    return np.sum(resids ** 2)


def calculate_reduced_chi_square(params, wv_dict, fl_dict, ep_dict, un_dict, central_wv, **kwargs):
    """Calculate reduced chi-square."""
    resids = standard_residuals(params, wv_dict, fl_dict, ep_dict, un_dict, central_wv, **kwargs)
    n_data = len(np.concatenate([resids]))
    n_params = len(params)
    chi2 = np.sum(resids ** 2)
    return chi2 / (n_data - n_params)


def grid_search(params, wv_dict, fl_dict, ep_dict, un_dict, central_wv,
                epoch, rv_min, rv_max, grid_step=5.0, **kwargs):
    """
    Coarse grid search over RV1, RV2 from rv_min to rv_max.
    """
    rv_values = np.arange(rv_min, rv_max + grid_step, grid_step)
    rv1_grid, rv2_grid = np.meshgrid(rv_values, rv_values)

    n_points = len(rv_values)
    chi2_grid = np.zeros((n_points, n_points))

    current_rv1 = params[f'rv1_epoch{epoch}'].value
    current_rv2 = params[f'rv2_epoch{epoch}'].value
    current_chi2 = calculate_chi_square(
        params.copy(), wv_dict, fl_dict, ep_dict, un_dict, central_wv, **kwargs
    )

    total_points = n_points * n_points
    start_time = time.time()

    print(f"\nGrid search with {total_points} points ({n_points}x{n_points} grid)")
    print(f"Grid range: RV=[{rv_min:.1f}, {rv_max:.1f}], step={grid_step:.1f}")

    with tqdm(total=total_points, desc=f"Grid search for epoch {epoch}") as pbar:
        for i in range(n_points):
            for j in range(n_points):
                # Optionally skip if two RVs are too close, etc.
                test_params = params.copy()
                test_params[f'rv1_epoch{epoch}'].value = rv1_grid[i, j]
                test_params[f'rv2_epoch{epoch}'].value = rv2_grid[i, j]
                chi2_grid[i, j] = calculate_chi_square(
                    test_params, wv_dict, fl_dict, ep_dict, un_dict, central_wv, **kwargs
                )
                pbar.update(1)

    min_idx = np.unravel_index(np.argmin(chi2_grid), chi2_grid.shape)
    best_grid_rv1 = rv1_grid[min_idx]
    best_grid_rv2 = rv2_grid[min_idx]
    best_grid_chi2 = chi2_grid[min_idx]

    elapsed = time.time() - start_time

    print(f"Grid search completed in {elapsed:.1f} seconds")
    print(f"Grid search results for Epoch {epoch}:")
    print(f"  Current RVs: ({current_rv1:.2f}, {current_rv2:.2f}), Chi2: {current_chi2:.4f}")
    print(f"  Best grid:   ({best_grid_rv1:.2f}, {best_grid_rv2:.2f}), Chi2: {best_grid_chi2:.4f}")

    raw_improvement = current_chi2 - best_grid_chi2
    print(f"  Raw chi² difference: {raw_improvement:.6f}")
    if raw_improvement > 0.001:
        improvement = raw_improvement / current_chi2 * 100
        print(f"  Improvement: {improvement:.2f}%")
    else:
        print("  No significant improvement found in grid search.")
    return rv1_grid, rv2_grid, chi2_grid, best_grid_rv1, best_grid_rv2, best_grid_chi2


def plot_chi2_landscape(rv1_grid, rv2_grid, chi2_grid, current_rvs, best_grid_rvs,
                        output_dir, epoch, sim_name):
    """
    Plot the chi-square landscape for the coarse/fine grid.
    """
    os.makedirs(output_dir, exist_ok=True)

    mask_valid = np.isfinite(chi2_grid)
    if not np.any(mask_valid):
        logger.warning(f"No valid chi² values for epoch {epoch}, skipping plot")
        return None

    chi2_finite = chi2_grid[mask_valid]
    chi2_max_finite = np.max(chi2_finite)
    chi2_plot = chi2_grid.copy()
    chi2_plot[~mask_valid] = chi2_max_finite * 1.1

    chi2_min = np.min(chi2_finite)
    norm_chi2 = chi2_plot - chi2_min + 1  # shift up for LogNorm

    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        max_val = np.max(norm_chi2)
        min_val = np.min(norm_chi2[norm_chi2 > 0])

        if max_val > min_val * 10:
            levels = np.logspace(np.log10(min_val), np.log10(max_val), 50)
            norm = LogNorm(vmin=min_val, vmax=max_val)
        else:
            levels = np.linspace(min_val, max_val, 50)
            norm = None

        contour = ax.contourf(rv1_grid, rv2_grid, norm_chi2, levels=levels,
                              cmap='viridis', norm=norm)

        if max_val > min_val * 10:
            cbar = fig.colorbar(contour, ax=ax, label='χ² - χ²$_{min}$ + 1')
            cbar.ax.set_yscale('log')
        else:
            cbar = fig.colorbar(contour, ax=ax, label='χ² - χ²$_{min}$ + 1')

        ax.plot(current_rvs[0], current_rvs[1], 'r*', markersize=10,
                label=f'Current: ({current_rvs[0]:.1f}, {current_rvs[1]:.1f})')
        ax.plot(best_grid_rvs[0], best_grid_rvs[1], 'wx', markersize=8, mew=2,
                label=f'Best: ({best_grid_rvs[0]:.1f}, {best_grid_rvs[1]:.1f})')

        ax.set_xlabel('RV1 (km/s)')
        ax.set_ylabel('RV2 (km/s)')
        ax.set_title(f'{sim_name} - χ² Landscape, Epoch {epoch}')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()

        outfile = os.path.join(output_dir, f'chi2_landscape_epoch{epoch}.png')
        plt.savefig(outfile, dpi=300)
        plt.close()
        return outfile
    except Exception as e:
        logger.error(f"Error creating plot for epoch {epoch}: {e}")
        plt.close()
        return None


def refine_fit(params, wv_dict, fl_dict, ep_dict, un_dict, central_wv,
               **kwargs):
    """
    Refine the fit using an LMFit Minimizer.
    1) Basin hopping
    2) Leastsq
    """
    minimizer = Minimizer(
        standard_residuals,
        params,
        fcn_args=(wv_dict, fl_dict, ep_dict, un_dict, central_wv),
        fcn_kws=kwargs
    )

    logger.info("Performing basin hopping optimization...")
    bh_callback = tqdm(total=10, desc="Basin hopping iterations")

    def bh_update_callback(x, f, accept):
        bh_callback.update(1)
        return False

    try:
        bh_result = minimizer.minimize(
            method='basinhopping',
            niter=10,
            T=5.0,
            stepsize=0.3,
            callback=bh_update_callback,
            minimizer_kwargs={'method': 'L-BFGS-B'}
        )
    finally:
        bh_callback.close()

    logger.info("Refining with least squares...")
    with tqdm(total=100, desc="Least squares refinement") as pbar:
        def ls_update_callback(params_, iter_, resid, *args, **kwargs_):
            pbar.update(1)
            return False

        try:
            result = minimizer.minimize(
                method='leastsq',
                params=bh_result.params,
                iter_cb=ls_update_callback,
                max_nfev=20000
            )
        except Exception as e:
            logger.warning(f"Callback failed: {e}, continuing without progress updates")
            result = minimizer.minimize(
                method='leastsq',
                params=bh_result.params,
                max_nfev=20000
            )

    logger.info("Refinement completed.")
    return result


def prepare_data_arrays(sim_folder, spectral_lines):
    """
    Prepare data arrays for fitting by loading observation files,
    building wv_dict, fl_dict, un_dict, ep_dict, etc.
    """
    from collections import OrderedDict

    try:
        epoch_files = find_observation_files(sim_folder)
        if not epoch_files:
            logger.error(f"No files found in {sim_folder}")
            return None
    except Exception as e:
        logger.error(f"Error finding observation files: {e}")
        return None

    logger.info(f"Found {len(epoch_files)} observation files.")

    fit_wv = OrderedDict()
    fit_fl = OrderedDict()
    fit_un = OrderedDict()
    fit_ep = OrderedDict()
    fit_noise = {}

    plot_wv = OrderedDict()
    plot_fl = OrderedDict()
    plot_un = OrderedDict()
    plot_ep = OrderedDict()
    plot_noise = {}

    for ln_name, ln_info in spectral_lines.items():
        line_id = f"line_{int(ln_info['rest_wave'] * 10)}"
        for d in [fit_wv, fit_fl, fit_un, fit_ep, plot_wv, plot_fl, plot_un, plot_ep]:
            d[line_id] = []
        fit_noise[line_id] = {}
        plot_noise[line_id] = {}

    logger.info(f"Building arrays for {len(spectral_lines)} lines...")

    for (ep, filepath) in tqdm(epoch_files, desc="Loading data"):
        try:
            df = load_data_for_epoch(filepath)
            if df.empty:
                continue

            for ln_name, ln_info in spectral_lines.items():
                line_id = f"line_{int(ln_info['rest_wave'] * 10)}"
                rest_wave = ln_info['rest_wave']
                half_window = ln_info['window'] / 2

                # find approximate line center
                search_extra = 25.0
                search_min = rest_wave - search_extra
                search_max = rest_wave + search_extra
                mask_search = (df['wavelength'] >= search_min) & (df['wavelength'] <= search_max)
                wv_search = df['wavelength'][mask_search].values
                fl_search = df['flux'][mask_search].values
                if len(wv_search) < 5:
                    continue

                found_center = find_line_center_smoothed(wv_search, fl_search, sigma=1.0)
                if found_center is None:
                    found_center = rest_wave

                line_min = found_center - half_window
                line_max = found_center + half_window
                mask_fit = (df['wavelength'] >= line_min) & (df['wavelength'] <= line_max)
                wv_fit = df['wavelength'][mask_fit].values
                fl_fit = df['flux'][mask_fit].values
                if len(wv_fit) == 0:
                    continue

                # estimate noise
                try:
                    nrs = find_noise_regions(
                        df, line_min, line_max,
                        noise_window=5,
                        min_noise_separation=5,
                        max_offset=30
                    )
                except TypeError:
                    nrs = find_noise_regions(
                        df, line_min, line_max,
                        noise_window=5,
                        min_noise_separation=5
                    )
                sigma_list = []
                for nm, nM, _ in nrs:
                    mm_noise = (df['wavelength'] >= nm) & (df['wavelength'] <= nM)
                    fl_n = df['flux'][mm_noise].values
                    if len(fl_n):
                        sigma_list.append(fl_n.std())
                if sigma_list:
                    epoch_sigma = np.mean(sigma_list)
                else:
                    epoch_sigma = max(0.02, fl_fit.std())

                # store
                fit_wv[line_id].append(wv_fit)
                fit_fl[line_id].append(fl_fit)
                fit_un[line_id].append(np.full_like(fl_fit, epoch_sigma))
                fit_ep[line_id].append(np.full_like(fl_fit, ep, dtype=int))
                fit_noise[line_id][ep] = nrs

                # store "plotting" arrays (slightly wider region)
                edges_min = [line_min] + [nm for nm, _, _ in nrs]
                edges_max = [line_max] + [nM for _, nM, _ in nrs]
                plot_min = min(edges_min) - 2
                plot_max = max(edges_max) + 2
                mask_plot = (df['wavelength'] >= plot_min) & (df['wavelength'] <= plot_max)
                wv_plot = df['wavelength'][mask_plot].values
                fl_plot = df['flux'][mask_plot].values
                if len(wv_plot) > 0:
                    plot_wv[line_id].append(wv_plot)
                    plot_fl[line_id].append(fl_plot)
                    plot_un[line_id].append(np.full_like(fl_plot, epoch_sigma))
                    plot_ep[line_id].append(np.full_like(fl_plot, ep, dtype=int))
                    plot_noise[line_id][ep] = nrs

        except Exception as e:
            logger.error(f"Error processing file {filepath} for epoch {ep}: {e}")
            continue

    # concatenate
    all_epochs = set()
    for lid in tqdm(fit_wv.keys(), desc="Concatenating arrays"):
        if len(fit_wv[lid]) > 0:
            fit_wv[lid] = np.concatenate(fit_wv[lid])
            fit_fl[lid] = np.concatenate(fit_fl[lid])
            fit_un[lid] = np.concatenate(fit_un[lid])
            fit_ep[lid] = np.concatenate(fit_ep[lid])
            all_epochs.update(fit_ep[lid])
        else:
            fit_wv[lid] = np.array([])
            fit_fl[lid] = np.array([])
            fit_un[lid] = np.array([])
            fit_ep[lid] = np.array([])

        if len(plot_wv[lid]) > 0:
            plot_wv[lid] = np.concatenate(plot_wv[lid])
            plot_fl[lid] = np.concatenate(plot_fl[lid])
            plot_un[lid] = np.concatenate(plot_un[lid])
            plot_ep[lid] = np.concatenate(plot_ep[lid])
        else:
            plot_wv[lid] = np.array([])
            plot_fl[lid] = np.array([])
            plot_un[lid] = np.array([])
            plot_ep[lid] = np.array([])

    if not all_epochs:
        logger.error("No valid data found in any of the processed files.")
        return None

    logger.info(f"Successfully built data arrays for {len(all_epochs)} epochs.")

    central_map = {}
    for ln_name, ln_info in spectral_lines.items():
        lid = f"line_{int(ln_info['rest_wave'] * 10)}"
        central_map[lid] = ln_info['rest_wave']

    return (fit_wv, fit_fl, fit_un, fit_ep, fit_noise,
            plot_wv, plot_fl, plot_un, plot_ep, plot_noise, central_map)


def do_full_grid_search_and_refine(folder_path,
                                   grid_step, fine_grid_step,
                                   rv_min, rv_max,
                                   skip_fine_grid,
                                   weighting_mode):
    """
    1) Locate standard_fit_params.json in standard_sym_voigt_{weighting_mode}_nobaseline_auto_fit_results
    2) Load them into params
    3) Prepare data arrays
    4) For each epoch, do coarse (and possibly fine) grid search
    5) Summarize improvement, generate plots
    6) If improved, refine_fit(...) and save new params
    """
    weighted_flag = (weighting_mode == 'weighted')
    json_path = find_standard_params(folder_path, weighting_mode)
    if not json_path:
        logger.error(f"No 'standard_fit_params.json' in subfolder for {weighting_mode}. Skipping {folder_path}.")
        return False

    logger.info(f"Found standard_fit_params.json: {json_path}")
    params = load_params(json_path)

    spectral_lines = SPECTRAL_LINES
    data_result = prepare_data_arrays(folder_path, spectral_lines)
    if data_result is None:
        logger.error(f"No data arrays for {folder_path}")
        return False

    (fit_wv, fit_fl, fit_un, fit_ep, fit_noise,
     plot_wv, plot_fl, plot_un, plot_ep, plot_noise,
     central_map) = data_result

    # ----- New snippet start -----
    # Remove any lines from the fit if their parameters are not present in the loaded params.
    keys_to_remove = []
    for key in list(central_map.keys()):
        if f"a1_{key}" not in params:
            print(f"Skipping line {key} because parameter 'a1_{key}' is missing in JSON.")
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del central_map[key]
        for d in [fit_wv, fit_fl, fit_un, fit_ep, fit_noise, plot_wv, plot_fl, plot_un, plot_ep, plot_noise]:
            if key in d:
                del d[key]
    # ----- New snippet end -----

    # gather epochs
    all_epochs = set()
    for lid in fit_ep:
        if len(fit_ep[lid]) > 0:
            all_epochs.update(fit_ep[lid])
    all_epochs = sorted(list(all_epochs))
    if not all_epochs:
        logger.error(f"No epochs found in {folder_path}")
        return False

    current_global_chi2 = np.sum(
        standard_residuals(params, fit_wv, fit_fl, fit_ep, fit_un,
                           central_map, profile_type='sym', weighted=weighted_flag) ** 2
    )
    logger.info(f"Initial global chi²: {current_global_chi2:.2f}")

    improved_epochs = {}
    epoch_details = []

    for ep in all_epochs:
        rv1_param = f"rv1_epoch{ep}"
        rv2_param = f"rv2_epoch{ep}"

        if rv1_param not in params or rv2_param not in params:
            logger.warning(f"Skipping epoch {ep} - parameters not found.")
            continue

        rv1_val = params[rv1_param].value
        rv2_val = params[rv2_param].value

        rv1_grid, rv2_grid, chi2_grid, best_rv1, best_rv2, best_chi2 = grid_search(
            params,
            fit_wv, fit_fl, fit_ep, fit_un, central_map,
            epoch=ep, rv_min=rv_min, rv_max=rv_max, grid_step=grid_step,
            profile_type='sym', weighted=weighted_flag
        )

        original_params = params.copy()

        test_params = params.copy()
        test_params[rv1_param].value = best_rv1
        test_params[rv2_param].value = best_rv2

        original_epoch_chi2 = calculate_chi_square(
            original_params, fit_wv, fit_fl, fit_ep, fit_un, central_map,
            profile_type='sym', weighted=weighted_flag
        )
        best_epoch_chi2 = calculate_chi_square(
            test_params, fit_wv, fit_fl, fit_ep, fit_un, central_map,
            profile_type='sym', weighted=weighted_flag
        )

        raw_improvement = original_epoch_chi2 - best_epoch_chi2
        improvement_pct = (raw_improvement / original_epoch_chi2 * 100) if original_epoch_chi2 else 0

        epoch_details.append({
            "epoch": ep,
            "original_rv1": rv1_val,
            "original_rv2": rv2_val,
            "best_rv1": best_rv1,
            "best_rv2": best_rv2,
            "original_chi2": original_epoch_chi2,
            "best_chi2": best_epoch_chi2,
            "raw_improvement": raw_improvement,
            "improvement_pct": improvement_pct
        })

        if raw_improvement > 0.001:
            logger.info(f"  Verified improvement for epoch {ep}: {raw_improvement:.6f}")
            improved_epochs[ep] = (best_rv1, best_rv2)
            params[rv1_param].value = best_rv1
            params[rv2_param].value = best_rv2

            # Fine grid if not skipped
            if fine_grid_step and fine_grid_step < grid_step and not skip_fine_grid:
                small_range = 5.0 * grid_step
                fine_rv1_min = max(rv_min, best_rv1 - small_range)
                fine_rv1_max = min(rv_max, best_rv1 + small_range)
                fine_rv2_min = max(rv_min, best_rv2 - small_range)
                fine_rv2_max = min(rv_max, best_rv2 + small_range)

                logger.info(f"  Running fine grid around ({best_rv1:.2f}, {best_rv2:.2f})...")
                fine_rv1_grid, fine_rv2_grid, fine_chi2_grid, fine_best_rv1, fine_best_rv2, fine_best_chi2 = grid_search(
                    params,
                    fit_wv, fit_fl, fit_ep, fit_un, central_map,
                    epoch=ep,
                    rv_min=fine_rv1_min, rv_max=fine_rv1_max,
                    grid_step=fine_grid_step,
                    profile_type='sym', weighted=weighted_flag
                )

                fine_test_params = params.copy()
                fine_test_params[rv1_param].value = fine_best_rv1
                fine_test_params[rv2_param].value = fine_best_rv2
                fine_epoch_chi2 = calculate_chi_square(
                    fine_test_params, fit_wv, fit_fl, fit_ep, fit_un, central_map,
                    profile_type='sym', weighted=weighted_flag
                )
                fine_improvement = original_epoch_chi2 - fine_epoch_chi2

                out_dir = os.path.join(folder_path, f"grid_search_plots_{weighting_mode}")
                os.makedirs(out_dir, exist_ok=True)
                plot_chi2_landscape(
                    fine_rv1_grid, fine_rv2_grid, fine_chi2_grid,
                    (rv1_val, rv2_val), (fine_best_rv1, fine_best_rv2),
                    output_dir=out_dir,
                    epoch=f"{ep}_fine",
                    sim_name=os.path.basename(folder_path)
                )

                if fine_improvement > raw_improvement:
                    logger.info(f"  Fine grid improved epoch {ep}: {fine_improvement:.6f} (was {raw_improvement:.6f})")
                    improved_epochs[ep] = (fine_best_rv1, fine_best_rv2)
                    params[rv1_param].value = fine_best_rv1
                    params[rv2_param].value = fine_best_rv2
                else:
                    logger.info("  Fine grid did not further improve results.")

        # Always produce a plot (coarse)
        out_dir = os.path.join(folder_path, f"grid_search_plots_{weighting_mode}")
        os.makedirs(out_dir, exist_ok=True)
        plot_chi2_landscape(
            rv1_grid, rv2_grid, chi2_grid,
            (rv1_val, rv2_val), (best_rv1, best_rv2),
            output_dir=out_dir,
            epoch=ep,
            sim_name=os.path.basename(folder_path)
        )

    # Summaries
    out_dir = os.path.join(folder_path, f"grid_search_results_{weighting_mode}")
    os.makedirs(out_dir, exist_ok=True)

    all_epochs_summary = os.path.join(out_dir, "all_epochs_summary.txt")
    with open(all_epochs_summary, 'w') as f:
        f.write(f"Grid Search Detailed Results for {os.path.basename(folder_path)} ({weighting_mode})\n")
        f.write("=====================================================\n\n")
        f.write(f"Grid parameters: rv_min={rv_min}, rv_max={rv_max}, step={grid_step}, fine_step={fine_grid_step}, "
                f"skip_fine_grid={skip_fine_grid}, weighting={weighting_mode}\n\n")
        f.write(f"Initial global chi²: {current_global_chi2:.6f}\n\n")
        f.write("Per-epoch results:\n-------------------\n\n")
        for info in epoch_details:
            f.write(f"Epoch {info['epoch']}:\n")
            f.write(f"  Current: RV1={info['original_rv1']:.2f}, RV2={info['original_rv2']:.2f}, "
                    f"Chi²={info['original_chi2']:.6f}\n")
            f.write(f"  Best:    RV1={info['best_rv1']:.2f}, RV2={info['best_rv2']:.2f}, "
                    f"Chi²={info['best_chi2']:.6f}\n")
            f.write(f"  Raw improvement: {info['raw_improvement']:.6f}  "
                    f"({info['improvement_pct']:.2f}%)\n\n")

    if improved_epochs:
        logger.info(f"Improvements found for epochs: {sorted(improved_epochs.keys())}")
        new_chi2 = np.sum(
            standard_residuals(params, fit_wv, fit_fl, fit_ep, fit_un,
                               central_map, profile_type='sym', weighted=weighted_flag) ** 2
        )
        improvement = (current_global_chi2 - new_chi2) / current_global_chi2 * 100
        logger.info(f"Chi² improvement from grid search: {improvement:.2f}%")

        logger.info("Refining the fit with the improved parameters...")
        result_final = refine_fit(
            params, fit_wv, fit_fl, fit_ep, fit_un,
            central_map, profile_type='sym', weighted=weighted_flag
        )

        # --- Begin MCMC uncertainty assessment ---
        def any_rv_stderr_zero(result):
            for pname, par in result.params.items():
                if pname.startswith("rv1_epoch") or pname.startswith("rv2_epoch"):
                    if par.stderr is None or par.stderr == 0.0:
                        return True
            return False

        if any_rv_stderr_zero(result_final):
            print("Some RV uncertainties are zero; running MCMC post-fit for uncertainty assessment...")
            mcmc_result = Minimizer(
                standard_residuals, result_final.params,
                fcn_args=(fit_wv, fit_fl, fit_ep, fit_un, central_map),
                fcn_kws={'profile_type': 'sym', 'weighted': weighted_flag}
            ).minimize(
                method='emcee', steps=2000, nwalkers=200, burn=500, thin=5, seed=123
            )
            mcmc_chain = mcmc_result.flatchain
            for pname, par in result_final.params.items():
                if pname.startswith("rv1_epoch") or pname.startswith("rv2_epoch"):
                    if par.stderr is None or par.stderr == 0.0:
                        if pname in mcmc_chain.columns:
                            new_std = mcmc_chain[pname].std()
                            par.stderr = new_std
                            print(f"[MCMC override] {pname}: new stderr = {new_std:.4f}")
        # --- End MCMC uncertainty assessment ---

        final_chi2 = np.sum(
            standard_residuals(result_final.params, fit_wv, fit_fl, fit_ep, fit_un,
                               central_map, profile_type='sym', weighted=weighted_flag) ** 2
        )
        total_improvement = (current_global_chi2 - final_chi2) / current_global_chi2 * 100
        logger.info(f"Final chi² after refinement: {final_chi2:.2f}")
        logger.info(f"Total improvement: {total_improvement:.2f}%")

        # Generate final refined-fit plots
        out_dir = os.path.join(folder_path, f"grid_refined_fit_results_{weighting_mode}")
        os.makedirs(out_dir, exist_ok=True)

        windows_map = {}
        for ln_name, ln_info in spectral_lines.items():
            lid = f"line_{int(ln_info['rest_wave'] * 10)}"
            windows_map[lid] = ln_info['window']

        df_res, df_chi = report_fit_results(
            result=result_final,
            wavelengths_line=fit_wv,
            fluxes_line=fit_fl,
            uncertainties_line=fit_un,
            epochs_line=fit_ep,
            central_wavelengths=central_map,
            noise_regions=fit_noise,
            windows=windows_map,
            output_directory=out_dir,
            profile_type='sym',
            plot_wavelengths_line=plot_wv,
            plot_fluxes_line=plot_fl,
            plot_uncertainties_line=plot_un,
            plot_epochs_line=plot_ep,
            plot_noise_dict=plot_noise
        )

        improved_json = os.path.join(out_dir, "improved_params.json")
        data_out = {}
        for k, v in result_final.params.items():
            data_out[k] = {"value": v.value, "min": v.min, "max": v.max}
        with open(improved_json, 'w') as f:
            json.dump(data_out, f, indent=2)
        logger.info(f"Saved improved parameters to {improved_json}")

        summary_file = os.path.join(out_dir, "improvement_summary.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Grid Search Results for {os.path.basename(folder_path)} ({weighting_mode})\n")
            f.write("======================================================\n\n")
            f.write(f"Original chi²: {current_global_chi2:.6f}\n")
            f.write(f"After grid search chi²: {new_chi2:.6f}\n")
            f.write(f"Final chi² after refinement: {final_chi2:.6f}\n")
            f.write(f"Total improvement: {total_improvement:.2f}%\n\n")
            f.write("Updated epochs:\n")
            for epoch, (rv1, rv2) in sorted(improved_epochs.items()):
                epoch_info = next((x for x in epoch_details if x["epoch"] == epoch), None)
                orig_rv1 = epoch_info["original_rv1"] if epoch_info else 0
                orig_rv2 = epoch_info["original_rv2"] if epoch_info else 0
                f.write(f"  Epoch {epoch}:\n")
                f.write(f"    Original:  RV1={orig_rv1:.2f}, RV2={orig_rv2:.2f}\n")
                f.write(f"    Grid Best: RV1={rv1:.2f},   RV2={rv2:.2f}\n")
                f.write(f"    Final:      RV1={result_final.params[f'rv1_epoch{epoch}'].value:.2f}, "
                        f"RV2={result_final.params[f'rv2_epoch{epoch}'].value:.2f}\n\n")

        return True
    else:
        logger.info("No improvements found. Current fit appears near a global minimum.")
        out_dir = os.path.join(folder_path, f"grid_search_results_{weighting_mode}")
        os.makedirs(out_dir, exist_ok=True)
        summary_file = os.path.join(out_dir, "no_improvement.txt")
        with open(summary_file, 'w') as f:
            f.write(f"Grid Search Results for {os.path.basename(folder_path)} ({weighting_mode})\n")
            f.write("======================================================\n\n")
            f.write("No improvements found in grid search.\n")
            f.write(f"Current chi²: {current_global_chi2:.6f}\n\n")
            f.write("See all_epochs_summary.txt for detailed info.\n")
        return True


def main():
    """Main function to run the script."""
    print("\n" + "=" * 70)
    print(" SPECTRAL LINE FITTING GLOBAL MINIMA CHECKER ".center(70, "="))
    print("=" * 70 + "\n")

    parser = argparse.ArgumentParser(description="Grid search for global minimum")
    parser.add_argument("--parent_folder", help="Path to parent folder containing subfolders")
    parser.add_argument("--folders", help="Comma-separated list of folder indices to process (e.g., '1,3,5')")
    parser.add_argument("--grid_step", type=float, default=5.0, help="Grid step size in km/s")
    parser.add_argument("--fine_step", type=float, default=1.0, help="Fine grid step size in km/s")
    parser.add_argument("--rv_min", type=float, default=-350.0, help="Minimum RV to search in km/s")
    parser.add_argument("--rv_max", type=float, default=650.0, help="Maximum RV to search in km/s")
    parser.add_argument("--skip_fine_grid", action="store_true", help="Skip fine grid search")

    args = parser.parse_args()

    parent_folder = args.parent_folder
    if not parent_folder:
        parent_folder = select_parent_folder()
    print(f"Selected parent folder: {parent_folder}")

    weighting_mode = ask_for_weighting()
    print(f"You selected: {weighting_mode.upper()}")

    candidate_folders = find_candidate_folders(parent_folder)
    if not candidate_folders:
        print(f"No subfolders found matching known patterns in {parent_folder}")
        return 1

    print("\nAvailable subfolders:")
    for i, cfold in enumerate(candidate_folders, start=1):
        print(f"  {i}. {os.path.basename(cfold)}")

    if args.folders:
        try:
            nums = [int(x.strip()) for x in args.folders.split(',')]
            folders_to_process = []
            for n in nums:
                if 1 <= n <= len(candidate_folders):
                    folders_to_process.append(candidate_folders[n - 1])
        except:
            print("Invalid folder selection.")
            return 1
    else:
        choice = input("\nProcess specific folders (comma-separated) or 'all': ").strip().lower()
        if choice == 'all':
            folders_to_process = candidate_folders
        else:
            try:
                nums = [int(x.strip()) for x in choice.split(',')]
                folders_to_process = []
                for n in nums:
                    if 1 <= n <= len(candidate_folders):
                        folders_to_process.append(candidate_folders[n - 1])
            except:
                print("Invalid input.")
                return 1

    if not folders_to_process:
        print("No valid selection => exiting.")
        return 1

    grid_step = args.grid_step
    fine_grid_step = args.fine_step
    rv_min = args.rv_min
    rv_max = args.rv_max
    skip_fine_grid = args.skip_fine_grid or SKIP_FINE_GRID_SEARCH

    print("\nGrid search parameters:")
    print(f"  Mode:             {weighting_mode}")
    print(f"  Grid step size:   {grid_step:.1f} km/s")
    print(f"  Fine grid step:   {fine_grid_step:.1f} km/s (Skip: {skip_fine_grid})")
    print(f"  RV range:         [{rv_min:.1f}, {rv_max:.1f}] km/s")

    results = {}
    start_t = time.time()
    for i, folder in enumerate(folders_to_process, start=1):
        folder_name = os.path.basename(folder)
        print(f"\n[{i}/{len(folders_to_process)}] Processing {folder_name}")
        print("-" * 50)
        try:
            t0 = time.time()
            success = do_full_grid_search_and_refine(
                folder, grid_step, fine_grid_step, rv_min, rv_max,
                skip_fine_grid, weighting_mode
            )
            t1 = time.time() - t0
            results[folder_name] = (success, t1)

            if i < len(folders_to_process):
                remaining = (len(folders_to_process) - i) * t1
                print(f"\nCompleted in {t1:.1f}s. Estimated time remaining: {remaining:.1f}s")
        except Exception as ex:
            logger.error(f"Error: {ex}")
            import traceback
            logger.debug(traceback.format_exc())
            results[folder_name] = (False, 0)

    total_time = time.time() - start_t

    print("\n" + "=" * 70)
    print(" PROCESSING SUMMARY ".center(70, "="))
    print("=" * 70)
    success_count = 0
    for f, (ok, tsec) in results.items():
        stat = "SUCCESS" if ok else "FAILED"
        if ok:
            success_count += 1
        print(f"{f}: {stat} ({tsec:.1f}s)")

    print(f"\nSuccessfully processed {success_count}/{len(results)} folders.")
    print(f"Total time: {total_time:.1f} s")

    summary_file = os.path.join(parent_folder, f"grid_search_summary_{time.strftime('%Y%m%d_%H%M%S')}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Grid Search Summary ({weighting_mode})\n")
        f.write("=======================================\n\n")
        f.write(f"Parent folder: {parent_folder}\n")
        f.write(f"Weighted mode: {weighting_mode}\n")
        f.write(f"Grid parameters: rv_min={rv_min}, rv_max={rv_max}, step={grid_step}, fine_step={fine_grid_step}, "
                f"skip_fine_grid={skip_fine_grid}\n\n")
        for folder_name, (ok, tsec) in results.items():
            status_text = "SUCCESS" if ok else "FAILED"
            f.write(f"{folder_name}: {status_text} ({tsec:.1f}s)\n")
        f.write(f"\nTotal time: {total_time:.1f} s\n")

    print(f"\nSummary saved to: {summary_file}")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\nOperation interrupted by user.")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)
