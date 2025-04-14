#!/usr/bin/env python3

"""
visualize_improved_parameters.py

A script to visualize and analyze the improved parameters from grid search:
1) Loads improved parameters from a JSON file
2) Prepares data arrays for the spectral lines
3) Generates epoch-by-epoch plots showing the fits
4) Creates detailed fit result reports
5) Calculates improvement metrics versus original parameters
6) Optionally performs final refinement of the fit

Configuration options are set directly in this script below.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm
import argparse
import logging
import time
from lmfit import Parameters

# ================================================================
# USER CONFIGURATION - EDIT THESE SETTINGS
# ================================================================

# Set to True to enable interactive folder selection, False to use the paths below
USE_INTERACTIVE_SELECTION = True

# Path to simulation folder (e.g., '8-110')
SIM_FOLDER = ""  # Leave empty to use interactive selection when USE_INTERACTIVE_SELECTION is True

# Paths to parameter files (leave empty to auto-detect)
IMPROVED_JSON_PATH = ""  # Path to improved_params.json
ORIGINAL_JSON_PATH = ""  # Path to standard_fit_params.json

# Output directory (leave empty to use default: SIM_FOLDER/improved_parameters_visualization)
OUTPUT_DIR = ""

# Set to True to perform final refinement using improved parameters as initial guess
PERFORM_REFINEMENT = True

# Set to True to enable debug logging
DEBUG_MODE = False

# Spectral line definitions
SPECTRAL_LINES = {
    'He4026': {'rest_wave': 4026.0, 'window': 20.0},
    'He4388': {'rest_wave': 4388.0, 'window': 20.0},
    'He4471': {'rest_wave': 4471.5, 'window': 20.0},
    'H4340': {'rest_wave': 4340.472, 'window': 20.0}
}

# ================================================================
# END OF USER CONFIGURATION
# ================================================================

# local imports - these should match the imports in your grid search script
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
    # If modules not in path, try to add the parent directory
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

# Set up logging
logging.basicConfig(
    level=logging.DEBUG if DEBUG_MODE else logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def select_folder_with_file(title, file_pattern=None):
    """Open a file dialog to select a folder containing a specific file pattern."""
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    folder_path = filedialog.askdirectory(title=title)
    root.destroy()

    if not folder_path:
        print("No folder selected. Exiting.")
        sys.exit(1)

    # If file pattern is provided, verify it exists in the folder
    if file_pattern:
        file_path = os.path.join(folder_path, file_pattern)
        if not os.path.isfile(file_path):
            print(f"Required file '{file_pattern}' not found in {folder_path}. Exiting.")
            sys.exit(1)

    return folder_path


def load_params_from_json(json_file):
    """Load parameters from JSON file into lmfit Parameters object."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)

        params = Parameters()
        for name, details in data.items():
            val = details.get("value", 0.0)
            pmin = details.get("min", None)
            pmax = details.get("max", None)
            params.add(name, value=val, min=pmin, max=pmax)

        return params
    except Exception as e:
        logger.error(f"Error loading parameters from {json_file}: {e}")
        return None


def prepare_data_arrays(sim_folder, spectral_lines):
    """
    Prepare data arrays for fitting by loading and processing observation files.

    Args:
        sim_folder: Path to simulation folder
        spectral_lines: Dictionary of spectral lines info

    Returns:
        Data dictionaries needed for fitting and central wavelength map
    """
    # Find observation files
    try:
        epoch_files = find_observation_files(sim_folder)
        if not epoch_files:
            logger.error(f"No files found in {sim_folder}")
            return None
    except Exception as e:
        logger.error(f"Error finding observation files: {e}")
        return None

    logger.info(f"Found {len(epoch_files)} observation files")

    # Initialize data dictionaries
    from collections import OrderedDict
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

    # Initialize lines
    for ln_name, ln_info in spectral_lines.items():
        line_id = f"line_{int(ln_info['rest_wave'] * 10)}"
        for d in [fit_wv, fit_fl, fit_un, fit_ep, plot_wv, plot_fl, plot_un, plot_ep]:
            d[line_id] = []
        fit_noise[line_id] = {}
        plot_noise[line_id] = {}

    logger.info(f"Building arrays for {len(spectral_lines)} spectral lines...")

    # Process each file
    for (ep, filepath) in tqdm(epoch_files, desc="Loading data"):
        try:
            df = load_data_for_epoch(filepath)
            if df.empty:
                continue

            for ln_name, ln_info in spectral_lines.items():
                line_id = f"line_{int(ln_info['rest_wave'] * 10)}"
                rest_wave = ln_info['rest_wave']
                half_window = ln_info['window'] / 2

                # Find the line center
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

                # Define line window
                line_min = found_center - half_window
                line_max = found_center + half_window
                mask_fit = (df['wavelength'] >= line_min) & (df['wavelength'] <= line_max)
                wv_fit = df['wavelength'][mask_fit].values
                fl_fit = df['flux'][mask_fit].values

                if len(wv_fit) == 0:
                    continue

                # Find noise regions for uncertainty estimates
                # Check the signature of find_noise_regions in your implementation
                try:
                    # First try with max_offset
                    nrs = find_noise_regions(
                        df, line_min, line_max,
                        noise_window=5,
                        min_noise_separation=5,
                        max_offset=30
                    )
                except TypeError:
                    try:
                        # If that fails, try without max_offset
                        nrs = find_noise_regions(
                            df, line_min, line_max,
                            noise_window=5,
                            min_noise_separation=5
                        )
                    except Exception as e:
                        # Fall back to minimal version
                        logger.warning(f"Using minimal noise_regions call due to error: {e}")
                        nrs = find_noise_regions(df, line_min, line_max)

                # Estimate uncertainty
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

                # Store fit data
                fit_wv[line_id].append(wv_fit)
                fit_fl[line_id].append(fl_fit)
                fit_un[line_id].append(np.full_like(fl_fit, epoch_sigma))
                fit_ep[line_id].append(np.full_like(fl_fit, ep, dtype=int))
                fit_noise[line_id][ep] = nrs

                # For plotting arrays (wider range)
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

    # Concatenate arrays
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

    # Check if we have any data
    if not all_epochs:
        logger.error("No valid data found in any of the processed files")
        return None

    logger.info(f"Successfully built data arrays for {len(all_epochs)} epochs")

    # Create central wavelength map
    central_map = {}
    for ln_name, ln_info in spectral_lines.items():
        lid = f"line_{int(ln_info['rest_wave'] * 10)}"
        central_map[lid] = ln_info['rest_wave']

    return (fit_wv, fit_fl, fit_un, fit_ep, fit_noise,
            plot_wv, plot_fl, plot_un, plot_ep, plot_noise, central_map, sorted(list(all_epochs)))


def plot_epoch_comparison(epoch, original_params, improved_params,
                          sim_folder, output_dir, spectral_lines,
                          fit_wv, fit_fl, fit_un, fit_ep, central_map):
    """
    Generate a comparison plot of original vs improved parameters for a specific epoch.

    Args:
        epoch: Epoch number to plot
        original_params: Original Parameters object
        improved_params: Improved Parameters object from grid search
        sim_folder: Base folder for the simulation
        output_dir: Directory to save plots
        spectral_lines: Dictionary of spectral lines info
        fit_wv, fit_fl, fit_un, fit_ep: Data dictionaries
        central_map: Dictionary mapping line IDs to central wavelengths

    Returns:
        Path to saved plot file
    """
    # Create epoch mask to get only data for this epoch
    epoch_mask = {}
    for lid in fit_ep:
        epoch_mask[lid] = (fit_ep[lid] == epoch)

    # Get the RV values for this epoch
    rv1_param = f"rv1_epoch{epoch}"
    rv2_param = f"rv2_epoch{epoch}"

    original_rv1 = original_params[rv1_param].value
    original_rv2 = original_params[rv2_param].value
    improved_rv1 = improved_params[rv1_param].value
    improved_rv2 = improved_params[rv2_param].value

    # Calculate chi-square for original and improved parameters
    try:
        orig_resids = standard_residuals(
            original_params,
            {lid: fit_wv[lid][epoch_mask[lid]] for lid in fit_wv if np.any(epoch_mask[lid])},
            {lid: fit_fl[lid][epoch_mask[lid]] for lid in fit_fl if np.any(epoch_mask[lid])},
            {lid: fit_ep[lid][epoch_mask[lid]] for lid in fit_ep if np.any(epoch_mask[lid])},
            {lid: fit_un[lid][epoch_mask[lid]] for lid in fit_un if np.any(epoch_mask[lid])},
            central_map,
            profile_type='sym', weighted=False
        )
        original_chi2 = np.sum(orig_resids ** 2)

        imp_resids = standard_residuals(
            improved_params,
            {lid: fit_wv[lid][epoch_mask[lid]] for lid in fit_wv if np.any(epoch_mask[lid])},
            {lid: fit_fl[lid][epoch_mask[lid]] for lid in fit_fl if np.any(epoch_mask[lid])},
            {lid: fit_ep[lid][epoch_mask[lid]] for lid in fit_ep if np.any(epoch_mask[lid])},
            {lid: fit_un[lid][epoch_mask[lid]] for lid in fit_un if np.any(epoch_mask[lid])},
            central_map,
            profile_type='sym', weighted=False
        )
        improved_chi2 = np.sum(imp_resids ** 2)
    except Exception as e:
        logger.error(f"Error calculating chi-square for epoch {epoch}: {e}")
        return None

    # Create a multi-panel figure for all spectral lines
    n_lines = len(spectral_lines)
    fig = plt.figure(figsize=(12, 3 * n_lines))
    gs = GridSpec(n_lines, 1, figure=fig)

    for i, (ln_name, ln_info) in enumerate(spectral_lines.items()):
        line_id = f"line_{int(ln_info['rest_wave'] * 10)}"

        # Skip if no data for this line and epoch
        if not np.any(epoch_mask[line_id]):
            continue

        # Get data for this line and epoch
        wv = fit_wv[line_id][epoch_mask[line_id]]
        fl = fit_fl[line_id][epoch_mask[line_id]]

        # Create a model from both parameter sets
        try:
            original_model = compute_full_model(
                wv, original_params, epoch, central_map[line_id],
                profile_type='sym', weighted=False
            )

            improved_model = compute_full_model(
                wv, improved_params, epoch, central_map[line_id],
                profile_type='sym', weighted=False
            )

            # Get individual star components for both fits
            original_components = get_star_components(
                wv, original_params, epoch, central_map[line_id],
                profile_type='sym'
            )

            improved_components = get_star_components(
                wv, improved_params, epoch, central_map[line_id],
                profile_type='sym'
            )
        except Exception as e:
            logger.error(f"Error computing model for epoch {epoch}, line {line_id}: {e}")
            continue

        # Add subplot
        ax = fig.add_subplot(gs[i, 0])

        # Plot data points
        ax.plot(wv, fl, 'ko', markersize=3, alpha=0.7, label='Data')

        # Plot original fit and components
        ax.plot(wv, original_model, 'r-', lw=1.5, alpha=0.7,
                label=f'Original (RV1={original_rv1:.1f}, RV2={original_rv2:.1f})')
        ax.plot(wv, original_components[0], 'r--', lw=1, alpha=0.5)
        ax.plot(wv, original_components[1], 'r--', lw=1, alpha=0.5)

        # Plot improved fit and components
        ax.plot(wv, improved_model, 'b-', lw=1.5, alpha=0.7,
                label=f'Improved (RV1={improved_rv1:.1f}, RV2={improved_rv2:.1f})')
        ax.plot(wv, improved_components[0], 'b--', lw=1, alpha=0.5)
        ax.plot(wv, improved_components[1], 'b--', lw=1, alpha=0.5)

        # Labels and title
        ax.set_xlabel('Wavelength (Å)')
        ax.set_ylabel('Flux')
        ax.set_title(f'{ln_name} - Epoch {epoch}')
        ax.grid(alpha=0.3)
        ax.legend(loc='best')

    # Add overall title with improvement info
    improvement_pct = (original_chi2 - improved_chi2) / original_chi2 * 100 if original_chi2 > 0 else 0
    plt.suptitle(
        f'Epoch {epoch} Comparison - Improvement: {original_chi2 - improved_chi2:.4f} ({improvement_pct:.2f}%)',
        fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, f'epoch_{epoch}_comparison.png')
    plt.savefig(outfile, dpi=150)
    plt.close()

    return outfile


def generate_rv_curve_plot(original_params, improved_params, all_epochs, output_dir):
    """
    Generate a radial velocity curve comparing original and improved parameters.

    Args:
        original_params: Original Parameters object
        improved_params: Improved Parameters object from grid search
        all_epochs: List of all epochs
        output_dir: Directory to save the plot

    Returns:
        Path to saved plot file
    """
    # Collect RV data for all epochs
    epochs = sorted(all_epochs)
    original_rv1 = [original_params[f'rv1_epoch{ep}'].value for ep in epochs]
    original_rv2 = [original_params[f'rv2_epoch{ep}'].value for ep in epochs]
    improved_rv1 = [improved_params[f'rv1_epoch{ep}'].value for ep in epochs]
    improved_rv2 = [improved_params[f'rv2_epoch{ep}'].value for ep in epochs]

    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot original RVs
    ax.plot(epochs, original_rv1, 'ro', label='Original RV1')
    ax.plot(epochs, original_rv2, 'ro', mfc='none', label='Original RV2')
    ax.plot(epochs, original_rv1, 'r-', alpha=0.5)
    ax.plot(epochs, original_rv2, 'r-', alpha=0.5)

    # Plot improved RVs
    ax.plot(epochs, improved_rv1, 'bo', label='Improved RV1')
    ax.plot(epochs, improved_rv2, 'bo', mfc='none', label='Improved RV2')
    ax.plot(epochs, improved_rv1, 'b-', alpha=0.5)
    ax.plot(epochs, improved_rv2, 'b-', alpha=0.5)

    # Highlight epochs with significant changes
    for i, ep in enumerate(epochs):
        dv1 = abs(improved_rv1[i] - original_rv1[i])
        dv2 = abs(improved_rv2[i] - original_rv2[i])
        if dv1 > 20 or dv2 > 20:  # Highlight epochs with changes > 20 km/s
            ax.annotate(f"Epoch {ep}", (ep, improved_rv1[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, color='blue')
            ax.annotate(f"Epoch {ep}", (ep, improved_rv2[i]),
                        xytext=(5, -15), textcoords='offset points',
                        fontsize=8, color='blue')

    # Labels and title
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Radial Velocity (km/s)')
    ax.set_title('Radial Velocity Curve: Original vs Improved Parameters')
    ax.grid(alpha=0.3)
    ax.legend(loc='best')

    # Save the figure
    os.makedirs(output_dir, exist_ok=True)
    outfile = os.path.join(output_dir, 'rv_curve_comparison.png')
    plt.savefig(outfile, dpi=300)
    plt.close()

    return outfile


def visualize_improvements(sim_folder, improved_json_path=None, original_json_path=None, output_dir=None,
                           spectral_lines=SPECTRAL_LINES):
    """
    Main function to visualize improvements from grid search.

    Args:
        sim_folder: Path to the simulation folder
        improved_json_path: Optional path to improved parameters JSON
        original_json_path: Optional path to original parameters JSON
        output_dir: Optional custom output directory
        spectral_lines: Dictionary of spectral lines info

    Returns:
        Tuple of (success, data_dict) where data_dict contains prepared data arrays
    """
    # Prepare output directory
    if not output_dir:
        output_dir = os.path.join(sim_folder, "improved_parameters_visualization")
    os.makedirs(output_dir, exist_ok=True)

    # Find parameter files if not provided
    if not improved_json_path:
        # Look in grid_refined_fit_results folder by default
        improved_json_path = os.path.join(sim_folder, "grid_refined_fit_results", "improved_params.json")

        if not os.path.isfile(improved_json_path):
            # Try to find it elsewhere
            for root, dirs, files in os.walk(sim_folder):
                if "improved_params.json" in files:
                    improved_json_path = os.path.join(root, "improved_params.json")
                    break

    if not original_json_path:
        # Look in standard fit results folders
        possible_dirs = [
            "standard_sym_voigt_unweighted_nobaseline_fit_results",
            "standard_sym_weighted_nobaseline_fit_results",
            "standard_sym_unweighted_nobaseline_fit_results",
            "standard_sym_voigt_weighted_nobaseline_fit_results"
        ]

        for d in possible_dirs:
            candidate_path = os.path.join(sim_folder, d, "standard_fit_params.json")
            if os.path.isfile(candidate_path):
                original_json_path = candidate_path
                break

    # Verify files exist
    if not improved_json_path or not os.path.isfile(improved_json_path):
        logger.error("Improved parameters JSON file not found.")
        return False, None

    if not original_json_path or not os.path.isfile(original_json_path):
        logger.error("Original parameters JSON file not found.")
        return False, None

    logger.info(f"Using improved parameters from: {improved_json_path}")
    logger.info(f"Using original parameters from: {original_json_path}")

    # Load parameters
    improved_params = load_params_from_json(improved_json_path)
    original_params = load_params_from_json(original_json_path)

    if not improved_params or not original_params:
        logger.error("Error loading parameter files.")
        return False, None

    # Prepare data arrays
    logger.info("Preparing data arrays...")
    data_result = prepare_data_arrays(sim_folder, spectral_lines)
    if data_result is None:
        logger.error("Error preparing data arrays.")
        return False, None

    (fit_wv, fit_fl, fit_un, fit_ep, fit_noise,
     plot_wv, plot_fl, plot_un, plot_ep, plot_noise,
     central_map, all_epochs) = data_result

    # Generate RV curve comparison
    logger.info("Generating RV curve comparison...")
    rv_curve_plot = generate_rv_curve_plot(original_params, improved_params, all_epochs, output_dir)
    logger.info(f"RV curve saved to: {rv_curve_plot}")

    # Calculate global improvement metrics
    original_global_chi2 = np.sum(standard_residuals(
        original_params, fit_wv, fit_fl, fit_ep, fit_un, central_map,
        profile_type='sym', weighted=False) ** 2)

    improved_global_chi2 = np.sum(standard_residuals(
        improved_params, fit_wv, fit_fl, fit_ep, fit_un, central_map,
        profile_type='sym', weighted=False) ** 2)

    total_improvement = (original_global_chi2 - improved_global_chi2) / original_global_chi2 * 100

    # Generate epoch-by-epoch comparison plots
    logger.info(f"Generating epoch comparison plots for {len(all_epochs)} epochs...")
    epoch_plots = []
    for ep in tqdm(all_epochs):
        plot_path = plot_epoch_comparison(
            ep, original_params, improved_params,
            sim_folder, output_dir, spectral_lines,
            fit_wv, fit_fl, fit_un, fit_ep, central_map
        )
        if plot_path:
            epoch_plots.append((ep, plot_path))

    # Generate detailed fit reports
    windows_map = {lid: spectral_lines[name]['window']
                   for name, info in spectral_lines.items()
                   for lid in [f"line_{int(info['rest_wave'] * 10)}"]}

    logger.info("Generating detailed fit reports for improved parameters...")
    df_res, df_chi = report_fit_results(
        result={"params": improved_params, "success": True},  # Mimic lmfit result structure
        wavelengths_line=fit_wv,
        fluxes_line=fit_fl,
        uncertainties_line=fit_un,
        epochs_line=fit_ep,
        central_wavelengths=central_map,
        noise_regions=fit_noise,
        windows=windows_map,
        output_directory=os.path.join(output_dir, "fit_reports"),
        profile_type='sym',
        plot_wavelengths_line=plot_wv,
        plot_fluxes_line=plot_fl,
        plot_uncertainties_line=plot_un,
        plot_epochs_line=plot_ep,
        plot_noise_dict=plot_noise
    )

    # Create a summary file
    summary_file = os.path.join(output_dir, "improvement_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Parameter Improvement Summary for {os.path.basename(sim_folder)}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Original parameters: {original_json_path}\n")
        f.write(f"Improved parameters: {improved_json_path}\n\n")

        f.write(f"Global Chi² Metrics:\n")
        f.write(f"  Original Chi²: {original_global_chi2:.6f}\n")
        f.write(f"  Improved Chi²: {improved_global_chi2:.6f}\n")
        f.write(f"  Absolute improvement: {original_global_chi2 - improved_global_chi2:.6f}\n")
        f.write(f"  Relative improvement: {total_improvement:.2f}%\n\n")

        f.write("Epochs with significant changes (|ΔRV| > 20 km/s):\n")
        for ep in all_epochs:
            rv1_param = f"rv1_epoch{ep}"
            rv2_param = f"rv2_epoch{ep}"

            orig_rv1 = original_params[rv1_param].value
            orig_rv2 = original_params[rv2_param].value
            impr_rv1 = improved_params[rv1_param].value
            impr_rv2 = improved_params[rv2_param].value

            dv1 = abs(impr_rv1 - orig_rv1)
            dv2 = abs(impr_rv2 - orig_rv2)

            if dv1 > 20 or dv2 > 20:
                f.write(f"  Epoch {ep}:\n")
                f.write(f"    Original: RV1={orig_rv1:.2f}, RV2={orig_rv2:.2f}\n")
                f.write(f"    Improved: RV1={impr_rv1:.2f}, RV2={impr_rv2:.2f}\n")
                f.write(f"    Changes:  ΔRV1={impr_rv1 - orig_rv1:.2f}, ΔRV2={impr_rv2 - orig_rv2:.2f}\n\n")

    logger.info(f"Visualization complete. Results saved in: {output_dir}")
    logger.info(f"Summary file: {summary_file}")

    # Create data dictionary to return for potential refinement
    data_dict = {
        'fit_wv': fit_wv,
        'fit_fl': fit_fl,
        'fit_un': fit_un,
        'fit_ep': fit_ep,
        'fit_noise': fit_noise,
        'plot_wv': plot_wv,
        'plot_fl': plot_fl,
        'plot_un': plot_un,
        'plot_ep': plot_ep,
        'plot_noise': plot_noise,
        'central_map': central_map,
        'all_epochs': all_epochs,
        'windows_map': windows_map,
        'original_params': original_params,
        'improved_params': improved_params,
        'original_json_path': original_json_path,
        'improved_json_path': improved_json_path
    }

    return True, data_dict


def perform_final_refinement(sim_folder, improved_json_path, output_dir, data_dict=None):
    """
    Perform a final refinement using improved parameters as initial guess.

    Args:
        sim_folder: Path to simulation folder
        improved_json_path: Path to improved parameters JSON
        output_dir: Output directory for refinement results
        data_dict: Optional dictionary with prepared data (to avoid reloading)

    Returns:
        Boolean indicating success or failure
    """
    logger.info("\n" + "=" * 70)
    logger.info(" PERFORMING FINAL REFINEMENT ".center(70, "="))
    logger.info("=" * 70 + "\n")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Use provided data or prepare data arrays
    if data_dict:
        fit_wv = data_dict['fit_wv']
        fit_fl = data_dict['fit_fl']
        fit_un = data_dict['fit_un']
        fit_ep = data_dict['fit_ep']
        fit_noise = data_dict['fit_noise']
        plot_wv = data_dict['plot_wv']
        plot_fl = data_dict['plot_fl']
        plot_un = data_dict['plot_un']
        plot_ep = data_dict['plot_ep']
        plot_noise = data_dict['plot_noise']
        central_map = data_dict['central_map']
        all_epochs = data_dict['all_epochs']
        windows_map = data_dict['windows_map']
        initial_params = data_dict['improved_params']
    else:
        # Define spectral lines
        spectral_lines = SPECTRAL_LINES

        # Load improved parameters as initial guess
        if not os.path.isfile(improved_json_path):
            logger.error(f"Error: Improved parameters file not found: {improved_json_path}")
            return False

        logger.info(f"Loading improved parameters from: {improved_json_path}")
        initial_params = load_params_from_json(improved_json_path)
        if not initial_params:
            logger.error("Error loading improved parameters.")
            return False

        # Prepare data arrays
        logger.info("Preparing data arrays for refinement...")
        data_result = prepare_data_arrays(sim_folder, spectral_lines)
        if data_result is None:
            logger.error("Error preparing data arrays.")
            return False

        (fit_wv, fit_fl, fit_un, fit_ep, fit_noise,
         plot_wv, plot_fl, plot_un, plot_ep, plot_noise,
         central_map, all_epochs) = data_result

        # Create window map
        windows_map = {lid: spectral_lines[name]['window']
                       for name, info in spectral_lines.items()
                       for lid in [f"line_{int(info['rest_wave'] * 10)}"]}

    # Calculate initial chi-square for comparison
    initial_chi2 = np.sum(standard_residuals(
        initial_params, fit_wv, fit_fl, fit_ep, fit_un, central_map,
        profile_type='sym', weighted=False) ** 2)

    logger.info(f"Initial global chi² before refinement: {initial_chi2:.6f}")

    # Create minimizer with LMFIT
    logger.info("Setting up minimizer...")
    from lmfit import Minimizer

    minimizer = Minimizer(
        standard_residuals,
        initial_params,
        fcn_args=(fit_wv, fit_fl, fit_ep, fit_un, central_map),
        fcn_kws={'profile_type': 'sym', 'weighted': False}
    )

    # First do basin hopping to explore parameter space
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

    # Then refine with least squares
    logger.info("Refining with least squares...")

    with tqdm(total=100, desc="Least squares refinement") as pbar:

        def ls_update_callback(params, iter, resid, *args, **kwargs):
            pbar.update(1)
            return False

        try:
            final_result = minimizer.minimize(
                method='leastsq',
                params=bh_result.params,
                iter_cb=ls_update_callback,
                max_nfev=20000
            )
        except Exception as e:
            # If callback doesn't work, fall back to no callback
            logger.warning(f"Callback failed, continuing without progress updates: {e}")
            final_result = minimizer.minimize(
                method='leastsq',
                params=bh_result.params,
                max_nfev=20000
            )

    # Calculate final chi-square
    final_chi2 = np.sum(standard_residuals(
        final_result.params, fit_wv, fit_fl, fit_ep, fit_un, central_map,
        profile_type='sym', weighted=False) ** 2)

    improvement = (initial_chi2 - final_chi2) / initial_chi2 * 100

    # Print refinement results
    logger.info("\nRefinement Results:")
    logger.info(f"  Initial Chi²: {initial_chi2:.6f}")
    logger.info(f"  Final Chi²:   {final_chi2:.6f}")
    logger.info(f"  Improvement:  {initial_chi2 - final_chi2:.6f} ({improvement:.2f}%)")

    # Save refined parameters
    refined_json = os.path.join(output_dir, "refined_params.json")
    data_out = {}
    for k, v in final_result.params.items():
        data_out[k] = {
            "value": v.value,
            "min": v.min,
            "max": v.max
        }
    with open(refined_json, 'w') as f:
        json.dump(data_out, f, indent=2)
    logger.info(f"Saved refined parameters to: {refined_json}")

    # Create summary of key parameter changes
    summary_file = os.path.join(output_dir, "refinement_summary.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Final Refinement Summary for {os.path.basename(sim_folder)}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Initial parameters: {improved_json_path}\n")
        f.write(f"Refined parameters: {refined_json}\n\n")

        f.write(f"Chi² Metrics:\n")
        f.write(f"  Initial Chi²: {initial_chi2:.6f}\n")
        f.write(f"  Final Chi²:   {final_chi2:.6f}\n")
        f.write(f"  Absolute improvement: {initial_chi2 - final_chi2:.6f}\n")
        f.write(f"  Relative improvement: {improvement:.2f}%\n\n")

        f.write("Parameter Changes Summary:\n")

        # Report changes for global parameters
        global_params = ['f_r', 'sigma1', 'sigma2', 'gamma']
        for param in global_params:
            if param in initial_params and param in final_result.params:
                initial_val = initial_params[param].value
                final_val = final_result.params[param].value
                change = final_val - initial_val
                percent = (change / initial_val * 100) if initial_val != 0 else float('inf')

                f.write(f"  {param}: {initial_val:.6f} -> {final_val:.6f} ")
                f.write(f"(Δ = {change:.6f}, {percent:.2f}%)\n")

        f.write("\nRV Parameter Changes:\n")
        # Report changes for RV parameters by epoch
        for ep in sorted(all_epochs):
            rv1_param = f'rv1_epoch{ep}'
            rv2_param = f'rv2_epoch{ep}'

            if rv1_param in initial_params and rv2_param in initial_params:
                init_rv1 = initial_params[rv1_param].value
                init_rv2 = initial_params[rv2_param].value
                final_rv1 = final_result.params[rv1_param].value
                final_rv2 = final_result.params[rv2_param].value

                delta_rv1 = final_rv1 - init_rv1
                delta_rv2 = final_rv2 - init_rv2

                if abs(delta_rv1) > 5 or abs(delta_rv2) > 5:  # Report significant changes
                    f.write(f"  Epoch {ep}:\n")
                    f.write(f"    RV1: {init_rv1:.2f} -> {final_rv1:.2f} (Δ = {delta_rv1:.2f})\n")
                    f.write(f"    RV2: {init_rv2:.2f} -> {final_rv2:.2f} (Δ = {delta_rv2:.2f})\n")

    # Generate fit reports with final parameters
    logger.info("\nGenerating fit reports with final parameters...")

    df_res, df_chi = report_fit_results(
        result=final_result,
        wavelengths_line=fit_wv,
        fluxes_line=fit_fl,
        uncertainties_line=fit_un,
        epochs_line=fit_ep,
        central_wavelengths=central_map,
        noise_regions=fit_noise,
        windows=windows_map,
        output_directory=output_dir,
        profile_type='sym',
        plot_wavelengths_line=plot_wv,
        plot_fluxes_line=plot_fl,
        plot_uncertainties_line=plot_un,
        plot_epochs_line=plot_ep,
        plot_noise_dict=plot_noise
    )

    # Compare initial and final RV curves
    logger.info("Generating comparison RV curve...")
    fig, ax = plt.subplots(figsize=(12, 8))

    epochs = sorted(all_epochs)

    # Get RV values from both parameter sets
    initial_rv1 = [initial_params[f'rv1_epoch{ep}'].value for ep in epochs]
    initial_rv2 = [initial_params[f'rv2_epoch{ep}'].value for ep in epochs]
    final_rv1 = [final_result.params[f'rv1_epoch{ep}'].value for ep in epochs]
    final_rv2 = [final_result.params[f'rv2_epoch{ep}'].value for ep in epochs]

    # Plot initial curves
    ax.plot(epochs, initial_rv1, 'ro', label='Initial RV1')
    ax.plot(epochs, initial_rv2, 'ro', mfc='none', label='Initial RV2')
    ax.plot(epochs, initial_rv1, 'r-', alpha=0.5)
    ax.plot(epochs, initial_rv2, 'r-', alpha=0.5)

    # Plot final curves
    ax.plot(epochs, final_rv1, 'bo', label='Final RV1')
    ax.plot(epochs, final_rv2, 'bo', mfc='none', label='Final RV2')
    ax.plot(epochs, final_rv1, 'b-', alpha=0.5)
    ax.plot(epochs, final_rv2, 'b-', alpha=0.5)

    # Labels and formatting
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Radial Velocity (km/s)')
    ax.set_title(f'Radial Velocity Curve: Initial vs Final Refinement')
    ax.grid(alpha=0.3)
    ax.legend(loc='best')

    # Save the figure
    rv_curve_file = os.path.join(output_dir, 'rv_curve_refinement.png')
    plt.savefig(rv_curve_file, dpi=300)
    plt.close()

    logger.info(f"Refinement complete. Results saved to: {output_dir}")
    return True


def main():
    """Main function to run the script from command line or directly."""
    print("\n" + "=" * 70)
    print(" VISUALIZE AND REFINE IMPROVED PARAMETERS ".center(70, "="))
    print("=" * 70 + "\n")

    # Parse command line arguments if provided
    parser = argparse.ArgumentParser(description="Visualize improved parameters and perform final refinement.")
    parser.add_argument("--sim_folder", help="Path to simulation folder (e.g., 8-110)")
    parser.add_argument("--improved_json", help="Path to improved_params.json file")
    parser.add_argument("--original_json", help="Path to original standard_fit_params.json file")
    parser.add_argument("--no_refine", action="store_true", help="Skip final refinement")
    parser.add_argument("--output_dir", help="Custom output directory for results")

    args = parser.parse_args()

    # Override config settings with command line arguments if provided
    sim_folder = args.sim_folder or SIM_FOLDER
    improved_json_path = args.improved_json or IMPROVED_JSON_PATH
    original_json_path = args.original_json or ORIGINAL_JSON_PATH
    output_dir = args.output_dir or OUTPUT_DIR
    perform_refinement = not args.no_refine if args.no_refine is not None else PERFORM_REFINEMENT

    # If no sim_folder provided and interactive selection enabled, prompt user
    if not sim_folder and USE_INTERACTIVE_SELECTION:
        print("Please select the simulation folder (e.g., containing 4-059, 8-110):")
        sim_folder = select_folder_with_file("Select Simulation Folder")

    if not sim_folder:
        print("Error: No simulation folder specified. Use --sim_folder or set SIM_FOLDER in the script.")
        return 1

    # Determine output directory if not provided
    if not output_dir:
        output_dir = os.path.join(sim_folder, "improved_parameters_visualization")
    os.makedirs(output_dir, exist_ok=True)

    # Run visualization
    vis_success, data_dict = visualize_improvements(
        sim_folder=sim_folder,
        improved_json_path=improved_json_path,
        original_json_path=original_json_path,
        output_dir=output_dir
    )

    if not vis_success:
        print("Visualization failed. Exiting.")
        return 1

    # If refine flag set, perform final refinement
    if perform_refinement:
        refine_dir = os.path.join(output_dir, "final_refinement")
        refined_success = perform_final_refinement(
            sim_folder=sim_folder,
            improved_json_path=data_dict['improved_json_path'],
            output_dir=refine_dir,
            data_dict=data_dict
        )

        if not refined_success:
            print("Refinement failed.")
            return 2

    print(f"\nAll processing complete. Results saved to: {output_dir}")
    return 0
