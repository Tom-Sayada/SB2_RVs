#!/usr/bin/env python3

"""
Script: plot_improved_fits.py

Purpose:
  Generate detailed plots from improved parameters found after verify_global_minimum.
  Creates:
  - Epoch-by-epoch plots
  - RV vs time plots
  - RV2 vs RV1 plots
  Similar to the original fitting process.
"""

import os
import re
import sys
import json
import argparse
import numpy as np
import pandas as pd
from lmfit import Parameters
import matplotlib.pyplot as plt

try:
    import tkinter as tk
    from tkinter import filedialog

    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    print("Warning: tkinter not available. You must specify --results_folder")

# Local modules
from src.utils import (
    find_observation_files,
    load_data_for_epoch,
    find_noise_regions,
    find_line_center_smoothed
)
from src.plot_results import (
    report_fit_results,
    build_per_epoch_rvs,
    plot_rv_vs_time,
    plot_rv2_vs_rv1
)


def load_params_from_json(json_path):
    """
    Load parameters from a JSON file and convert to lmfit.Parameters
    """
    if not os.path.isfile(json_path):
        print(f"Error: File not found: {json_path}")
        return None

    try:
        with open(json_path, 'r') as f:
            param_data = json.load(f)

        # Convert to lmfit.Parameters
        params = Parameters()
        for name, details in param_data.items():
            value = details.get('value', 0.0)
            min_val = details.get('min', None)
            max_val = details.get('max', None)

            # Set all attributes at creation time
            params.add(name, value=value, min=min_val, max=max_val)

        return params

    except Exception as e:
        print(f"Error loading parameters from {json_path}: {str(e)}")
        return None


def detect_fit_type(params):
    """
    Determine if this is a standard fit or ratio-constrained fit
    based on parameter names.
    """
    if 'ratio' in params and 'v_sys' in params:
        return 'ratio'
    else:
        # Look for rv1_epochX and rv2_epochX pairs
        has_rv1 = any(name.startswith('rv1_epoch') for name in params)
        has_rv2 = any(name.startswith('rv2_epoch') for name in params)
        if has_rv1 and has_rv2:
            return 'standard'

    # Fallback
    return 'unknown'


def load_data_arrays(data_dir, spectral_lines):
    """
    Load spectral data from observation files.
    Returns combined and flattened arrays.
    """
    # Find observation files
    epoch_files = find_observation_files(data_dir)

    # Initialize data dictionaries
    fit_wv = {}
    fit_fl = {}
    fit_un = {}
    fit_ep = {}
    fit_noise = {}
    mjd_dict = {}

    # Initialize for each line
    for ln_name, ln_info in spectral_lines.items():
        line_id = f"line_{int(ln_info['rest_wave'] * 10)}"
        fit_wv[line_id] = []
        fit_fl[line_id] = []
        fit_un[line_id] = []
        fit_ep[line_id] = []
        fit_noise[line_id] = {}

    # Load data
    print(f"Loading data from {len(epoch_files)} observation files...")
    for (ep, filepath) in epoch_files:
        df = load_data_for_epoch(filepath)
        if df.empty:
            continue

        # Store MJD if available
        if hasattr(df, 'attrs') and 'MJD' in df.attrs:
            mjd_dict[ep] = df.attrs['MJD']
        else:
            mjd_dict[ep] = np.nan

        # Process each line
        for ln_name, ln_info in spectral_lines.items():
            line_id = f"line_{int(ln_info['rest_wave'] * 10)}"
            restw = ln_info['rest_wave']
            halfw = ln_info['window'] / 2.0

            # Find line in spectrum
            search_extra = 25.0
            smin = restw - search_extra
            smax = restw + search_extra
            mask_search = (df['wavelength'] >= smin) & (df['wavelength'] <= smax)
            wv_search = df['wavelength'][mask_search].values
            fl_search = df['flux'][mask_search].values
            if len(wv_search) < 5:
                continue

            # Find line center
            found_center = find_line_center_smoothed(wv_search, fl_search, sigma=1.0, absorption=True)
            if found_center is None:
                found_center = restw

            # Extract line region
            lw_min = found_center - halfw
            lw_max = found_center + halfw
            mask_fit = (df['wavelength'] >= lw_min) & (df['wavelength'] <= lw_max)
            wv_fit = df['wavelength'][mask_fit].values
            fl_fit = df['flux'][mask_fit].values
            if len(wv_fit) == 0:
                continue

            # Find noise regions
            nrs = find_noise_regions(df, lw_min, lw_max, noise_window=5, min_noise_separation=5)
            if nrs:
                sig_list = []
                for (nm, nM, _) in nrs:
                    mmn = (df['wavelength'] >= nm) & (df['wavelength'] <= nM)
                    fl_n = df['flux'][mmn].values
                    if len(fl_n) > 0:
                        sig_list.append(fl_n.std())
                if sig_list:
                    epoch_sigma = np.mean(sig_list)
                else:
                    epoch_sigma = max(0.02, fl_fit.std())
            else:
                epoch_sigma = max(0.02, fl_fit.std())

            # Store data
            fit_wv[line_id].append(wv_fit)
            fit_fl[line_id].append(fl_fit)
            fit_un[line_id].append(np.full_like(fl_fit, epoch_sigma))
            fit_ep[line_id].append(np.full_like(fl_fit, ep, dtype=int))
            fit_noise[line_id].setdefault(ep, [])
            fit_noise[line_id][ep] = nrs

    # Flatten arrays
    all_ep = set()
    for line_id in fit_wv:
        if fit_wv[line_id]:
            fit_wv[line_id] = np.concatenate(fit_wv[line_id])
            fit_fl[line_id] = np.concatenate(fit_fl[line_id])
            fit_un[line_id] = np.concatenate(fit_un[line_id])
            fit_ep[line_id] = np.concatenate(fit_ep[line_id])
            all_ep.update(fit_ep[line_id].tolist())
        else:
            fit_wv[line_id] = np.array([])
            fit_fl[line_id] = np.array([])
            fit_un[line_id] = np.array([])
            fit_ep[line_id] = np.array([])

    all_ep = np.unique(list(all_ep))

    # Central wavelengths map
    central_map = {}
    for ln_name, ln_info in spectral_lines.items():
        line_id = f"line_{int(ln_info['rest_wave'] * 10)}"
        central_map[line_id] = ln_info['rest_wave']

    return (fit_wv, fit_fl, fit_un, fit_ep, fit_noise,
            central_map, list(all_ep), mjd_dict)


def perform_mcmc_analysis(params, fit_type, wv_dict, fl_dict, ep_dict, un_dict, central_map,
                          profile_type='sym', line_profile='voigt', weighted=True, fit_baseline=False):
    """
    Perform MCMC analysis to estimate parameter uncertainties.
    """
    try:
        # Import the appropriate residual function based on fit type
        if fit_type == 'ratio':
            from src.ratio_fit_model import residuals
            fcn_kws = {
                'profile_type': profile_type,
                'line_profile': line_profile,
                'weighted': weighted,
                'fit_baseline': fit_baseline
            }
        else:
            from src.model_builder import residuals
            fcn_kws = {
                'profile_type': profile_type,
                'line_profile': line_profile,
                'weighted': weighted
            }

        # Create a Minimizer and run MCMC to get uncertainties
        from lmfit import Minimizer

        minimizer = Minimizer(
            residuals,
            params,
            fcn_args=(wv_dict, fl_dict, ep_dict, un_dict, central_map),
            fcn_kws=fcn_kws
        )

        # Run MCMC - we don't need to modify the parameters, just get uncertainties
        n_params = len([p for p in params if params[p].vary])
        mcmc_result = minimizer.minimize(
            method='emcee',
            params=params,
            steps=2000,
            # To:
            # Calculate number of parameters and set appropriate number of walkers
            nwalkers = max(100, 2 * n_params + 10),  # At least 2x params plus some buffer
            burn=300,
            thin=20,
            is_weighted=weighted,
            seed=123
        )

        # Get the MCMC chain for plotting
        mcmc_chain = mcmc_result.flatchain

        # Update parameter errors from MCMC
        for name in params:
            if name in mcmc_chain.columns:
                std_val = mcmc_chain[name].std()
                params[name].stderr = std_val
                print(f"Updated {name} uncertainty: {std_val:.6f}")

        print(f"MCMC complete, determined uncertainties for {len(mcmc_chain.columns)} parameters")

        return mcmc_chain

    except Exception as e:
        print(f"Warning: MCMC calculation failed: {str(e)}")
        return None


def show_folder_dialog(title="Select folder"):
    """
    Show a folder selection dialog using tkinter.
    Returns the selected folder path or None if cancelled.
    """
    if not HAS_TKINTER:
        print("Error: tkinter is not available. Cannot show folder dialog.")
        return None

    try:
        # Initialize tkinter and immediately withdraw the window
        root = tk.Tk()
        root.withdraw()

        # Make sure dialog appears on top
        root.attributes("-topmost", True)

        # Show the dialog
        print(f"Opening folder dialog: {title}")
        folder_path = filedialog.askdirectory(title=title)

        # Clean up
        root.destroy()

        if not folder_path:
            print("No folder selected.")
            return None

        return folder_path
    except Exception as e:
        print(f"Error showing folder dialog: {str(e)}")
        return None


def process_results_folder(results_folder):
    """
    Generate plots from the improved parameters in the results folder.
    """
    # Check if this is indeed a results folder
    folder_name = os.path.basename(results_folder)
    if not ("improved_rv_fit" in folder_name or "_results" in folder_name):
        print(f"Warning: Selected folder does not appear to be a results folder: {folder_name}")
        confirm = input("Continue anyway? (y/n): ")
        if confirm.lower() != 'y':
            return False

    # Find the parent folder (one level up) for data files
    parent_folder = os.path.dirname(results_folder)
    print(f"Using parent folder for data files: {parent_folder}")

    # Look for improved parameter file
    param_files = []
    for fname in os.listdir(results_folder):
        if "improved" in fname and fname.endswith(".json"):
            param_files.append(os.path.join(results_folder, fname))

    if not param_files:
        # Try any standard parameter file
        for fname in os.listdir(results_folder):
            if (fname.startswith("standard_fit_params") or fname.startswith("ratio_fit_params")) and fname.endswith(
                    ".json"):
                param_files.append(os.path.join(results_folder, fname))

    if not param_files:
        print(f"No parameter files found in {results_folder}")
        return False

    # Use the first parameter file found
    params_file = param_files[0]
    print(f"Using parameter file: {params_file}")

    # Load parameters
    params = load_params_from_json(params_file)
    if params is None:
        return False

    # Determine fit type
    fit_type = detect_fit_type(params)
    print(f"Detected fit type: {fit_type}")

    # Define spectral lines
    spectral_lines = {
        'He4471': {'rest_wave': 4471.5, 'window': 20.0},
        'He4026': {'rest_wave': 4026.0, 'window': 20.0},
        'He4388': {'rest_wave': 4388.0, 'window': 20.0},
        'H4340': {'rest_wave': 4340.472, 'window': 20.0}
    }

    # Load data arrays from parent folder
    data_arrays = load_data_arrays(parent_folder, spectral_lines)
    if not data_arrays or len(data_arrays[6]) == 0:  # Check all_epochs
        print(f"No valid data found in {parent_folder}")
        return False

    wv_dict, fl_dict, un_dict, ep_dict, noise_dict, central_map, all_epochs, mjd_dict = data_arrays

    # Create output directory
    output_dir = os.path.join(results_folder, "improved_plots")
    os.makedirs(output_dir, exist_ok=True)

    # Determine fit settings from folder name
    if "sym" in folder_name:
        profile_type = 'sym'
    elif "asym" in folder_name:
        profile_type = 'asym'
    else:
        profile_type = 'sym'  # Default

    if "voigt" in folder_name:
        line_profile = 'voigt'
    elif "gaussian" in folder_name:
        line_profile = 'gaussian'
    else:
        line_profile = 'voigt'  # Default

    if "unweighted" in folder_name:
        weighted = False
    else:
        weighted = True

    if "baseline" in folder_name and "nobaseline" not in folder_name:
        fit_baseline = True
    else:
        fit_baseline = False

    # Run MCMC analysis to get uncertainties
    print("Running MCMC to calculate parameter uncertainties...")
    mcmc_chain = perform_mcmc_analysis(
        params, fit_type, wv_dict, fl_dict, ep_dict, un_dict, central_map,
        profile_type, line_profile, weighted, fit_baseline
    )

    # If MCMC was successful, save the updated parameters
    if mcmc_chain is not None:
        updated_params_file = os.path.join(output_dir, "improved_params_with_uncertainties.json")
        param_data = {}
        for name, param in params.items():
            param_data[name] = {
                "value": param.value,
                "stderr": param.stderr if param.stderr is not None else None,
                "min": param.min,
                "max": param.max
            }
        with open(updated_params_file, 'w') as f:
            json.dump(param_data, f, indent=2)
        print(f"Saved parameters with uncertainties to: {updated_params_file}")

    # Create a Result-like object for report_fit_results
    class MockResult:
        def __init__(self, params, var_names=None):
            self.params = params
            self.var_names = list(params.keys()) if var_names is None else var_names

    mock_result = MockResult(params)

    # Create windows map
    windows_map = {}
    for ln_name, ln_info in spectral_lines.items():
        lid = f"line_{int(ln_info['rest_wave'] * 10)}"
        windows_map[lid] = ln_info['window']

    # Generate plots
    print(f"Generating plots in: {output_dir}")
    df_res, df_chi = report_fit_results(
        result=mock_result,
        wavelengths_line=wv_dict,
        fluxes_line=fl_dict,
        uncertainties_line=un_dict,
        epochs_line=ep_dict,
        central_wavelengths=central_map,
        noise_regions=noise_dict,
        windows=windows_map,
        output_directory=output_dir,
        profile_type=profile_type,
        line_profile=line_profile,
        plot_wavelengths_line=wv_dict,  # Use same arrays for plotting
        plot_fluxes_line=fl_dict,
        plot_uncertainties_line=un_dict,
        plot_epochs_line=ep_dict,
        plot_noise_dict=noise_dict,
        mcmc_chain=mcmc_chain,
        mjd_dict=mjd_dict
    )

    # RV vs time plots might be missing from results folder, create them directly
    if mjd_dict and df_res is not None and not df_res.empty and "MJD" in df_res.columns:
        # Check if we have any valid MJD values
        if not df_res["MJD"].isna().all():
            plot_rv_vs_time(df_res, output_dir)
            print("Generated RV vs Time plots")

    # RV2 vs RV1 plot might be missing, create it directly
    if df_res is not None and not df_res.empty:
        plot_rv2_vs_rv1(df_res, output_dir)
        print("Generated RV2 vs RV1 plot")

    print(f"All plots generated in: {output_dir}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Generate detailed plots from improved parameters found after verify_global_minimum."
    )
    parser.add_argument("--results_folder", type=str, default=None,
                        help="Folder containing improved parameters (from verify_global_minimum)")
    args = parser.parse_args()

    # Get results folder
    results_folder = args.results_folder

    # If not provided, show file dialog
    if results_folder is None:
        print("Please select the folder containing improved parameters...")
        results_folder = show_folder_dialog(
            title="Select folder with improved parameters (e.g. *_improved_rv_fit)"
        )

        if results_folder is None:
            print("No folder selected or dialog error. Exiting.")
            sys.exit(1)

    print(f"Selected folder: {results_folder}")

    # Check if folder exists
    if not os.path.isdir(results_folder):
        print(f"Folder not found: {results_folder}")
        sys.exit(1)

    # Process the folder
    result = process_results_folder(results_folder)

    if result:
        print("\nPlots generated successfully.")
    else:
        print("\nFailed to generate plots.")


if __name__ == "__main__":
    main()