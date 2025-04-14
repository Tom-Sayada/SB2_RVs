#!/usr/bin/env python3

"""
Script: verify_global_minimum.py

Purpose:
  1. Check if the global fit found the global minimum by individually
     re-optimizing RVs for each epoch while keeping other parameters fixed.
  2. If improvements are found, save the improved parameters and generate
     comparison plots showing before/after fit quality.
  3. Works with both standard and ratio-constrained fits.

Process:
  1. User selects parent folder containing F-NNN subfolders
  2. For each subfolder, look for standard_fit_params.json or ratio_fit_params.json
  3. Load parameters and observation data
  4. For each epoch, optimize RV1+RV2 or just RV2 (ratio case) independently
  5. Compare original vs optimized chi^2 values
  6. Generate summary reports and update parameters if improvements found
"""

import os
import re
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from lmfit import Parameters, Minimizer

# Tkinter import with error handling
try:
    import tkinter as tk
    from tkinter import filedialog

    HAS_TKINTER = True
except ImportError:
    HAS_TKINTER = False
    print("Warning: tkinter not available. You must specify --parent_folder")

# Local modules
from src.utils import (
    find_observation_files,
    load_data_for_epoch,
    find_noise_regions,
    find_line_center_smoothed
)
from src.model_builder import (
    compute_full_model as standard_compute_model,
    residuals as standard_residuals
)
from src.ratio_fit_model import (
    residuals as ratio_residuals
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
            # Get values with proper defaults
            value = details.get('value', 0.0)
            min_val = details.get('min', None)
            max_val = details.get('max', None)

            # Create the parameter correctly - need to set all attributes at creation time
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
    Very similar to the loading routine in the main fit scripts.
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
    for (ep, filepath) in tqdm(epoch_files, desc="Loading data"):
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


def compute_global_chi2(params, is_ratio_fit, wv_dict, fl_dict, ep_dict, un_dict,
                        central_map, profile_type='sym', line_profile='voigt',
                        weighted=True, fit_baseline=False):
    """
    Compute the global chi^2 value using the appropriate residual function
    based on the fit type.
    """
    if is_ratio_fit:
        # Ratio-constrained fit
        res = ratio_residuals(
            params,
            wv_dict, fl_dict, ep_dict, un_dict, central_map,
            profile_type=profile_type,
            line_profile=line_profile,
            weighted=weighted,
            fit_baseline=fit_baseline
        )
    else:
        # Standard fit
        res = standard_residuals(
            params,
            wv_dict, fl_dict, ep_dict, un_dict, central_map,
            profile_type=profile_type,
            line_profile=line_profile,
            weighted=weighted
        )

    chi2 = np.sum(res ** 2)
    n_points = sum(len(fl_dict[lid]) for lid in fl_dict)
    n_params = len([p for p in params if params[p].vary])
    dof = max(1, n_points - n_params)
    chi2_r = chi2 / dof

    return chi2, chi2_r, dof


def verify_global_minimum_ratio(params, wv_dict, fl_dict, ep_dict, un_dict,
                                central_map, all_epochs, profile_type='sym',
                                line_profile='voigt', weighted=True,
                                fit_baseline=False):
    """
    For ratio-constrained fits, independently optimize rv2_epochN for each epoch
    while keeping all other parameters fixed.
    """
    # Get initial chi^2
    original_chi2, original_chi2_r, dof = compute_global_chi2(
        params, True, wv_dict, fl_dict, ep_dict, un_dict, central_map,
        profile_type, line_profile, weighted, fit_baseline
    )

    # Copy parameters to store improvements
    improved_params = params.copy()

    # Track improvements
    improvements = []

    print(f"Original global chi^2 = {original_chi2:.2f}, chi^2_r = {original_chi2_r:.4f}")
    print("Optimizing RVs for each epoch independently...")

    # For each epoch, optimize only rv2_epochN
    for epoch in tqdm(all_epochs):
        # Create copy of parameters
        epoch_params = params.copy()

        # Free only rv2_epochN, fix all others
        for param_name in epoch_params:
            if param_name == f'rv2_epoch{epoch}':
                epoch_params[param_name].vary = True
            else:
                epoch_params[param_name].vary = False

        # Create minimizer for just this epoch
        epoch_minimizer = Minimizer(
            ratio_residuals,
            epoch_params,
            fcn_args=(wv_dict, fl_dict, ep_dict, un_dict, central_map),
            fcn_kws={
                'profile_type': profile_type,
                'line_profile': line_profile,
                'weighted': weighted,
                'fit_baseline': fit_baseline
            }
        )

        # Optimize
        try:
            epoch_result = epoch_minimizer.minimize(method='leastsq')

            # Compute chi^2 with new parameters
            new_chi2, new_chi2_r, _ = compute_global_chi2(
                epoch_result.params, True, wv_dict, fl_dict, ep_dict, un_dict, central_map,
                profile_type, line_profile, weighted, fit_baseline
            )

            # Check if improved
            if new_chi2 < original_chi2:
                # Get original and new values
                rv2_orig = params[f'rv2_epoch{epoch}'].value
                rv2_new = epoch_result.params[f'rv2_epoch{epoch}'].value

                # Calculate rv1 based on ratio and v_sys
                ratio_val = params['ratio'].value
                v_sys_val = params['v_sys'].value
                rv1_orig = -ratio_val * (rv2_orig + v_sys_val) - v_sys_val
                rv1_new = -ratio_val * (rv2_new + v_sys_val) - v_sys_val

                # Record improvement
                delta_chi2 = original_chi2 - new_chi2
                improvements.append({
                    'Epoch': epoch,
                    'RV1_orig': rv1_orig,
                    'RV1_new': rv1_new,
                    'RV2_orig': rv2_orig,
                    'RV2_new': rv2_new,
                    'Delta_chi2': delta_chi2,
                    'Pct_improvement': 100 * delta_chi2 / original_chi2
                })

                # Update improved parameters
                improved_params[f'rv2_epoch{epoch}'] = epoch_result.params[f'rv2_epoch{epoch}']

        except Exception as e:
            print(f"Error optimizing epoch {epoch}: {str(e)}")

    # Compute final chi^2 with all improvements
    if improvements:
        final_chi2, final_chi2_r, _ = compute_global_chi2(
            improved_params, True, wv_dict, fl_dict, ep_dict, un_dict, central_map,
            profile_type, line_profile, weighted, fit_baseline
        )

        print(f"\nFinal improved chi^2 = {final_chi2:.2f}, chi^2_r = {final_chi2_r:.4f}")
        print(f"Total improvement: {original_chi2 - final_chi2:.2f} "
              f"({100 * (original_chi2 - final_chi2) / original_chi2:.2f}%)")
    else:
        final_chi2 = original_chi2
        final_chi2_r = original_chi2_r
        print("\nNo improvements found. The fit appears to be at the global minimum.")

    # Return results
    return {
        'original_chi2': original_chi2,
        'original_chi2_r': original_chi2_r,
        'final_chi2': final_chi2,
        'final_chi2_r': final_chi2_r,
        'improved_params': improved_params,
        'improvements': improvements,
        'dof': dof
    }


def verify_global_minimum_standard(params, wv_dict, fl_dict, ep_dict, un_dict,
                                   central_map, all_epochs, profile_type='sym',
                                   line_profile='voigt', weighted=True,
                                   fit_baseline=False):
    """
    For standard fits, independently optimize rv1_epochN and rv2_epochN for each epoch
    while keeping all other parameters fixed.
    """
    # Get initial chi^2
    original_chi2, original_chi2_r, dof = compute_global_chi2(
        params, False, wv_dict, fl_dict, ep_dict, un_dict, central_map,
        profile_type, line_profile, weighted, fit_baseline
    )

    # Copy parameters to store improvements
    improved_params = params.copy()

    # Track improvements
    improvements = []

    print(f"Original global chi^2 = {original_chi2:.2f}, chi^2_r = {original_chi2_r:.4f}")
    print("Optimizing RVs for each epoch independently...")

    # For each epoch, optimize both rv1_epochN and rv2_epochN
    for epoch in tqdm(all_epochs):
        # Create copy of parameters
        epoch_params = params.copy()

        # Free only rv1_epochN and rv2_epochN, fix all others
        for param_name in epoch_params:
            if param_name in [f'rv1_epoch{epoch}', f'rv2_epoch{epoch}']:
                epoch_params[param_name].vary = True
            else:
                epoch_params[param_name].vary = False

        # Create minimizer for just this epoch
        epoch_minimizer = Minimizer(
            standard_residuals,
            epoch_params,
            fcn_args=(wv_dict, fl_dict, ep_dict, un_dict, central_map),
            fcn_kws={
                'profile_type': profile_type,
                'line_profile': line_profile,
                'weighted': weighted
            }
        )

        # Optimize
        try:
            epoch_result = epoch_minimizer.minimize(method='leastsq')

            # Compute chi^2 with new parameters
            new_chi2, new_chi2_r, _ = compute_global_chi2(
                epoch_result.params, False, wv_dict, fl_dict, ep_dict, un_dict, central_map,
                profile_type, line_profile, weighted, fit_baseline
            )

            # Check if improved
            if new_chi2 < original_chi2:
                # Get original and new values
                rv1_orig = params[f'rv1_epoch{epoch}'].value
                rv2_orig = params[f'rv2_epoch{epoch}'].value
                rv1_new = epoch_result.params[f'rv1_epoch{epoch}'].value
                rv2_new = epoch_result.params[f'rv2_epoch{epoch}'].value

                # Record improvement
                delta_chi2 = original_chi2 - new_chi2
                improvements.append({
                    'Epoch': epoch,
                    'RV1_orig': rv1_orig,
                    'RV1_new': rv1_new,
                    'RV2_orig': rv2_orig,
                    'RV2_new': rv2_new,
                    'Delta_chi2': delta_chi2,
                    'Pct_improvement': 100 * delta_chi2 / original_chi2
                })

                # Update improved parameters
                improved_params[f'rv1_epoch{epoch}'] = epoch_result.params[f'rv1_epoch{epoch}']
                improved_params[f'rv2_epoch{epoch}'] = epoch_result.params[f'rv2_epoch{epoch}']

        except Exception as e:
            print(f"Error optimizing epoch {epoch}: {str(e)}")

    # Compute final chi^2 with all improvements
    if improvements:
        final_chi2, final_chi2_r, _ = compute_global_chi2(
            improved_params, False, wv_dict, fl_dict, ep_dict, un_dict, central_map,
            profile_type, line_profile, weighted, fit_baseline
        )

        print(f"\nFinal improved chi^2 = {final_chi2:.2f}, chi^2_r = {final_chi2_r:.4f}")
        print(f"Total improvement: {original_chi2 - final_chi2:.2f} "
              f"({100 * (original_chi2 - final_chi2) / original_chi2:.2f}%)")
    else:
        final_chi2 = original_chi2
        final_chi2_r = original_chi2_r
        print("\nNo improvements found. The fit appears to be at the global minimum.")

    # Return results
    return {
        'original_chi2': original_chi2,
        'original_chi2_r': original_chi2_r,
        'final_chi2': final_chi2,
        'final_chi2_r': final_chi2_r,
        'improved_params': improved_params,
        'improvements': improvements,
        'dof': dof
    }


def save_improved_params_to_json(params, filepath):
    """
    Save improved parameters to a JSON file.
    """
    data = {}
    for name, param in params.items():
        data[name] = {
            "value": param.value,
            "min": param.min,
            "max": param.max
        }

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved improved parameters to: {filepath}")


def plot_improvements(improvements_df, output_dir):
    """
    Generate plots showing the improvements in RV values.
    """
    if improvements_df.empty:
        return

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Plot of RV1 original vs new
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.scatter(improvements_df['Epoch'], improvements_df['RV1_orig'],
                marker='o', color='blue', label='Original RV1')
    plt.scatter(improvements_df['Epoch'], improvements_df['RV1_new'],
                marker='x', color='red', label='Improved RV1')

    # Draw connecting lines
    for _, row in improvements_df.iterrows():
        plt.plot([row['Epoch'], row['Epoch']],
                 [row['RV1_orig'], row['RV1_new']],
                 'k--', alpha=0.5)

    plt.title('RV1 Values: Original vs. Improved')
    plt.ylabel('RV1 (km/s)')
    plt.legend()
    plt.grid(alpha=0.3)

    # Plot of RV2 original vs new
    plt.subplot(2, 1, 2)
    plt.scatter(improvements_df['Epoch'], improvements_df['RV2_orig'],
                marker='o', color='blue', label='Original RV2')
    plt.scatter(improvements_df['Epoch'], improvements_df['RV2_new'],
                marker='x', color='red', label='Improved RV2')

    # Draw connecting lines
    for _, row in improvements_df.iterrows():
        plt.plot([row['Epoch'], row['Epoch']],
                 [row['RV2_orig'], row['RV2_new']],
                 'k--', alpha=0.5)

    plt.title('RV2 Values: Original vs. Improved')
    plt.xlabel('Epoch')
    plt.ylabel('RV2 (km/s)')
    plt.legend()
    plt.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'rv_improvements.png'), dpi=300)
    plt.close()

    # Plot of chi^2 improvements
    plt.figure(figsize=(10, 6))
    plt.bar(improvements_df['Epoch'], improvements_df['Pct_improvement'], alpha=0.7)
    plt.title('Chi² Improvement by Epoch (%)')
    plt.xlabel('Epoch')
    plt.ylabel('Improvement (%)')
    plt.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chi2_improvements.png'), dpi=300)
    plt.close()


def process_folder(folder_path, output_suffix="_improved_rv_fit"):
    """
    Process a single F-NNN folder:
    1. Look for standard_fit_params.json in the expected location
    2. Determine whether it's a standard or ratio-constrained fit
    3. Load parameters and data
    4. Verify if we found the global minimum
    5. Save results and plots if improvements found
    """
    # Check for specific standard result folder pattern
    std_folder = os.path.join(folder_path, "standard_sym_voigt_weighted_nobaseline_fit_results")
    params_file = os.path.join(std_folder, "standard_fit_params.json")

    # Use specific file if found
    if os.path.isdir(std_folder) and os.path.isfile(params_file):
        folder_name = std_folder
        is_ratio_fit = False
    else:
        # Fallback to more flexible search if specific path not found
        # Look for result folders with standard naming pattern
        result_folders = []
        for entry in os.listdir(folder_path):
            full_path = os.path.join(folder_path, entry)
            if os.path.isdir(full_path) and any(x in entry for x in ["standard", "ratio"]):
                result_folders.append(full_path)

        folder_name = None
        params_file = None
        is_ratio_fit = False

        # Check standard fit first - look for any standard fit results
        for rf in result_folders:
            if "standard" in os.path.basename(rf):
                param_candidates = [
                    os.path.join(rf, "standard_fit_params.json"),
                    os.path.join(rf, "standard_fit_params_swapped.json")
                ]
                for pc in param_candidates:
                    if os.path.isfile(pc):
                        folder_name = rf
                        params_file = pc
                        is_ratio_fit = False
                        break
                if params_file:
                    break

        # Check ratio fit if standard not found
        if folder_name is None:
            for rf in result_folders:
                if "ratio" in os.path.basename(rf):
                    param_candidates = [
                        os.path.join(rf, "ratio_fit_params.json"),
                        os.path.join(rf, "ratio_fit_params_swapped.json")
                    ]
                    for pc in param_candidates:
                        if os.path.isfile(pc):
                            folder_name = rf
                            params_file = pc
                            is_ratio_fit = True
                            break
                    if params_file:
                        break

    if folder_name is None:
        print(f"No standard or ratio fit results found in {folder_path}")
        return False

    # Load parameters
    params = load_params_from_json(params_file)
    if params is None:
        return False

    # Double-check fit type
    detected_type = detect_fit_type(params)
    if (detected_type == 'ratio' and not is_ratio_fit) or (detected_type == 'standard' and is_ratio_fit):
        print(f"Warning: Detected {detected_type} fit but expected {'ratio' if is_ratio_fit else 'standard'}")
        is_ratio_fit = (detected_type == 'ratio')

    # Define spectral lines
    spectral_lines = {
        'He4471': {'rest_wave': 4471.5, 'window': 20.0},
        'He4026': {'rest_wave': 4026.0, 'window': 20.0},
        'He4388': {'rest_wave': 4388.0, 'window': 20.0},
        'H4340': {'rest_wave': 4340.472, 'window': 20.0}
    }

    # Load data
    data_arrays = load_data_arrays(folder_path, spectral_lines)
    if not data_arrays or len(data_arrays[6]) == 0:  # Check all_epochs
        print(f"No valid data found in {folder_path}")
        return False

    wv_dict, fl_dict, un_dict, ep_dict, noise_dict, central_map, all_epochs, mjd_dict = data_arrays

    # Determine fit settings
    profile_type = 'sym'  # Default, typically matches folder name
    line_profile = 'voigt'  # Default, typically matches folder name
    weighted = True  # Default, typically matches folder name
    fit_baseline = False  # Default, typically matches folder name

    # Check folder name to infer settings
    if "sym" in folder_name:
        profile_type = 'sym'
    elif "asym" in folder_name:
        profile_type = 'asym'

    if "voigt" in folder_name:
        line_profile = 'voigt'
    elif "gaussian" in folder_name:
        line_profile = 'gaussian'

    if "unweighted" in folder_name:
        weighted = False

    if "baseline" in folder_name and "nobaseline" not in folder_name:
        fit_baseline = True

    # Create output directory
    output_dir = os.path.join(folder_path, folder_name + output_suffix)
    os.makedirs(output_dir, exist_ok=True)

    # Verify global minimum
    if is_ratio_fit:
        results = verify_global_minimum_ratio(
            params, wv_dict, fl_dict, ep_dict, un_dict, central_map, all_epochs,
            profile_type, line_profile, weighted, fit_baseline
        )
    else:
        results = verify_global_minimum_standard(
            params, wv_dict, fl_dict, ep_dict, un_dict, central_map, all_epochs,
            profile_type, line_profile, weighted, fit_baseline
        )

    # Process results
    improvements = results['improvements']

    if improvements:
        # Save improved parameters
        param_file_base = os.path.basename(params_file)
        improved_params_file = os.path.join(output_dir, f"improved_{param_file_base}")
        save_improved_params_to_json(results['improved_params'], improved_params_file)

        # Create summary DataFrame
        improvements_df = pd.DataFrame(improvements)

        # Save to Excel
        excel_file = os.path.join(output_dir, "rv_improvements_summary.xlsx")
        with pd.ExcelWriter(excel_file) as writer:
            improvements_df.to_excel(writer, sheet_name="RV_Improvements", index=False)

            # Add summary sheet
            summary_data = {
                'Metric': [
                    'Original Chi²', 'Original Chi²_r', 'Final Chi²', 'Final Chi²_r',
                    'Absolute Improvement', 'Percentage Improvement',
                    'Number of Improved Epochs', 'Total Epochs'
                ],
                'Value': [
                    results['original_chi2'], results['original_chi2_r'],
                    results['final_chi2'], results['final_chi2_r'],
                    results['original_chi2'] - results['final_chi2'],
                    100 * (results['original_chi2'] - results['final_chi2']) / results['original_chi2'],
                    len(improvements), len(all_epochs)
                ]
            }
            pd.DataFrame(summary_data).to_excel(writer, sheet_name="Summary", index=False)

        # Generate plots
        plot_improvements(improvements_df, output_dir)

        print(f"Saved results to {output_dir}")
        return True
    else:
        # No improvements found
        print(f"No improvements found for {folder_path}")

        # Create a simple report file
        with open(os.path.join(output_dir, "verification_report.txt"), "w") as f:
            f.write(f"Global Minimum Verification Report\n")
            f.write(f"===============================\n\n")
            f.write(f"Folder: {folder_path}\n")
            f.write(f"Fit type: {'Ratio-constrained' if is_ratio_fit else 'Standard'}\n")
            f.write(f"Chi² = {results['original_chi2']:.2f}, Chi²_r = {results['original_chi2_r']:.4f}\n")
            f.write(f"Degrees of freedom: {results['dof']}\n\n")
            f.write(f"Verification result: No improvements found.\n")
            f.write(f"The original fit appears to have found the global minimum.\n")

        return False


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
        root.attributes("-topmost", True)  # Make sure dialog appears on top
        root.withdraw()

        # Show the dialog
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


def main():
    parser = argparse.ArgumentParser(
        description="Check if the global fit found the global minimum by individually re-optimizing RVs for each epoch while keeping other parameters fixed."
    )
    parser.add_argument("--parent_folder", type=str, default=None,
                        help="Parent folder containing F-NNN subfolders")
    parser.add_argument("--output_suffix", type=str, default="_improved_rv_fit",
                        help="Suffix for output directories")
    parser.add_argument("--single_folder", action="store_true",
                        help="Process parent_folder directly instead of looking for F-NNN subfolders")
    args = parser.parse_args()

    # Get parent folder
    parent_folder = args.parent_folder

    # If not provided, show file dialog
    if parent_folder is None:
        print("Please select a folder...")
        parent_folder = show_folder_dialog(
            title="Select folder (parent with F-NNN subfolders or direct F-NNN folder)"
        )

        if parent_folder is None:
            print("No folder selected or dialog error. Exiting.")
            sys.exit(1)

    print(f"Selected folder: {parent_folder}")

    # Check if folder exists
    if not os.path.isdir(parent_folder):
        print(f"Folder not found: {parent_folder}")
        sys.exit(1)

    # Determine if the selected folder itself is F-NNN pattern
    folder_name = os.path.basename(parent_folder)
    f_nnn_pattern = re.compile(r'^\d+-\d{3}')

    if f_nnn_pattern.match(folder_name) or args.single_folder:
        # If the folder itself matches F-NNN pattern or single_folder flag is set,
        # process it directly
        print(f"Processing single folder: {parent_folder}")
        result = process_folder(parent_folder, args.output_suffix)
        if result:
            print("\nImprovement found. See output directory for details.")
        else:
            print("\nNo improvement found. The fit appears to be at the global minimum.")
    else:
        # Find F-NNN subfolders
        subfolders = []
        for entry in os.listdir(parent_folder):
            fullpath = os.path.join(parent_folder, entry)
            if os.path.isdir(fullpath) and f_nnn_pattern.match(entry):
                subfolders.append(fullpath)

        if not subfolders:
            print(f"No F-NNN subfolders found in {parent_folder}")
            print("If this is a direct analysis folder, use --single_folder flag.")
            sys.exit(1)

        # Sort by name
        subfolders.sort()

        print(f"Found {len(subfolders)} F-NNN subfolders in {parent_folder}")

        # Process each subfolder
        results = []
        for sf in subfolders:
            print(f"\nProcessing {os.path.basename(sf)}")
            result = process_folder(sf, args.output_suffix)
            results.append((sf, result))

        # Print summary
        print("\n==== VERIFICATION SUMMARY ====")
        improved = [sf for sf, res in results if res]
        if improved:
            print(f"Improvements found in {len(improved)}/{len(subfolders)} folders:")
            for sf in improved:
                print(f"  - {os.path.basename(sf)}")
        else:
            print("No improvements found in any folder. All fits appear to be at global minimum.")

    print("\nDone!")

if __name__ == "__main__":
    main()
