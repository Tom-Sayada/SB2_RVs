#!/usr/bin/env python3

"""
Script: postfit_swap_and_refine.py

Purpose:
  1. Prompt user to select a parent folder containing subfolders (like 4-003).
  2. Each subfolder has:
       - Observations (obs_* files).
       - Two result folders:
         (a) ratio_sym_voigt_weighted_nobaseline_fit_results/
         (b) standard_sym_voigt_weighted_nobaseline_fit_results/
           -> inside here is standard_fit_params.json
  3. We load the standard_fit_params.json from (b), plus the subfolder's observation files.
  4. For each epoch, attempt to swap rv1_epochN <-> rv2_epochN; if total chi^2 improves, keep the swap.
  5. After all epoch swaps, do a local least-squares re-minimization.
  6. Generate final plots/Excel (like normal) into "swap_refined_fit_results".
  7. Save final parameters to "standard_fit_params_swapped.json" in that new folder.
"""

import os
import re
import json
import argparse
import tkinter as tk
from tkinter import filedialog

import numpy as np
from tqdm import tqdm
from lmfit import Parameters, Minimizer, report_fit

# Local modules (assuming your standard src/ structure):
from src.utils import (
    find_observation_files,
    load_data_for_epoch,
    find_noise_regions,
    find_line_center_smoothed,
    DataLoadError
)
from src.model_builder import setup_parameters, residuals
from src.plot_results import report_fit_results


###############################################################################
# (A) Helpers to load/save param JSON
###############################################################################
def load_params_json(json_path):
    if not os.path.isfile(json_path):
        return {}
    with open(json_path, 'r') as f:
        raw = json.load(f)
    out = {}
    for k, subd in raw.items():
        val = subd.get("value", 0.0)
        vmin = subd.get("min", None)
        vmax = subd.get("max", None)
        out[k] = (val, vmin, vmax)
    return out


def save_params_json(params, filepath):
    """
    Save an lmfit.Parameters object to a JSON file,
    matching the "standard_fit_params.json" format.
    """
    data = {}
    for k, par in params.items():
        data[k] = {
            "value": par.value,
            "min": par.min,
            "max": par.max
        }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved updated parameters to '{filepath}'")


###############################################################################
# (B) Compute total or reduced chi^2
###############################################################################
def compute_chi_squared(params, minimizer,
                        fit_wv, fit_fl, fit_ep, fit_un,
                        central_map,
                        profile_type='sym',
                        line_profile='voigt',
                        weighted=True):
    """
    Return total chi^2 = sum of [residual^2].
    """
    res = minimizer.userfcn(
        params,
        fit_wv, fit_fl, fit_ep, fit_un, central_map,
        profile_type=profile_type,
        line_profile=line_profile,
        weighted=weighted
    )
    chi2 = np.sum(res**2)
    return chi2


###############################################################################
# (C) Attempt a single epoch swap
###############################################################################
def apply_epoch_rv_swap(params, ep):
    """
    Swap rv1_epoch{ep} <-> rv2_epoch{ep} in-place.
    Return old (rv1, rv2) so we can revert if needed.
    If not found, returns None.
    """
    rv1_key = f"rv1_epoch{ep}"
    rv2_key = f"rv2_epoch{ep}"

    if rv1_key not in params or rv2_key not in params:
        return None

    old_rv1 = params[rv1_key].value
    old_rv2 = params[rv2_key].value
    # Swap
    params[rv1_key].set(value=old_rv2)
    params[rv2_key].set(value=old_rv1)
    return (old_rv1, old_rv2)


def revert_epoch_rv_swap(params, ep, old_vals):
    """Revert swap if it didn't help."""
    if not old_vals:
        return
    rv1_key = f"rv1_epoch{ep}"
    rv2_key = f"rv2_epoch{ep}"
    (old1, old2) = old_vals
    params[rv1_key].set(value=old1)
    params[rv2_key].set(value=old2)


###############################################################################
# (D) Main function
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Post-fit script: epoch-wise RV swap for standard approach, then final local refine."
    )
    parser.add_argument("--profile_type", type=str, default="sym",
                        help="Must match the original fit: e.g. sym or asym.")
    parser.add_argument("--line_profile", type=str, default="voigt",
                        choices=["voigt", "gaussian"],
                        help="Must match the original fit used.")
    parser.add_argument("--unweighted", action='store_true',
                        help="If original was unweighted, set this.")
    parser.add_argument("--fit_baseline", action='store_true',
                        help="If original used --fit_baseline, set this.")
    parser.add_argument("--json_filename", type=str,
                        default="standard_fit_params.json",
                        help="Name of the param file in standard_sym_voigt_weighted_nobaseline_fit_results.")
    args = parser.parse_args()

    profile_type = args.profile_type
    line_profile = args.line_profile
    use_weighted = not args.unweighted
    fit_baseline = args.fit_baseline
    json_fname = args.json_filename

    # 1. Let user pick the parent folder
    root = tk.Tk()
    root.withdraw()
    print("Select the parent folder containing subfolders (like 4-003).")
    parent_folder = filedialog.askdirectory(title="Select Parent Folder")
    root.destroy()
    if not parent_folder:
        print("No folder selected. Exiting.")
        return
    print(f"Selected parent folder: {parent_folder}")

    # 2. Identify subfolders matching pattern F-NNN (e.g. 4-003, 10-111, etc.)
    pattern = re.compile(r'^\d+-\d{3}$')
    subfolders = []
    for entry in os.listdir(parent_folder):
        fullpath = os.path.join(parent_folder, entry)
        if os.path.isdir(fullpath) and pattern.match(entry):
            subfolders.append(fullpath)

    if not subfolders:
        print("No subfolders matching 'F-NNN' found.")
        return

    # We'll do a reference set of lines that you typically use:
    # Adjust if your real lines differ
    spectral_lines_info = {
        'He4471': {'rest_wave': 4471.5, 'window': 20.0},
        'He4026': {'rest_wave': 4026.0, 'window': 20.0},
        'He4388': {'rest_wave': 4388.0, 'window': 20.0},
        'H4340':  {'rest_wave': 4340.472, 'window': 20.0}
    }

    for sf in subfolders:
        print("\n==========================================")
        print(f"Processing subfolder: {sf}")

        # 3. Inside each subfolder, we have:
        #    standard_sym_voigt_weighted_nobaseline_fit_results/ <--- has the JSON
        #    ratio_sym_voigt_weighted_nobaseline_fit_results/    <--- not used here
        standard_dir = os.path.join(sf, "standard_sym_voigt_weighted_nobaseline_fit_results")
        if not os.path.isdir(standard_dir):
            print("No standard result folder, skipping.")
            continue
        param_json_path = os.path.join(standard_dir, json_fname)
        if not os.path.isfile(param_json_path):
            print(f"No JSON found at {param_json_path}, skipping.")
            continue

        # 4. Load the parameters
        loaded_params = load_params_json(param_json_path)
        if not loaded_params:
            print("Could not load parameters.")
            continue

        # Build an lmfit.Parameters,
        # but skip any param that fails to add (returns None).
        params_in = Parameters()
        for k, (val, vmin, vmax) in loaded_params.items():
            # Attempt to create param
            p = params_in.add(str(k), value=val)  # ensure name is string
            if p is None:
                print(f"[Warning] Could not create Parameter '{k}' => skipping.")
                continue

            # If vmin is not None, set p.min
            if (vmin is not None) and (vmin != float('-inf')):
                try:
                    p.min = vmin
                except Exception as ee:
                    print(f"[Warning] Could not set min={vmin} for {k}: {ee}")

            # If vmax is not None, set p.max
            if (vmax is not None) and (vmax != float('inf')):
                try:
                    p.max = vmax
                except Exception as ee:
                    print(f"[Warning] Could not set max={vmax} for {k}: {ee}")

        # 5. Re-load the data from the subfolder
        try:
            epoch_files = find_observation_files(sf)
        except DataLoadError:
            print("No valid obs files or data load error. Skipping.")
            continue

        from collections import OrderedDict
        fit_wv = OrderedDict()
        fit_fl = OrderedDict()
        fit_un = OrderedDict()
        fit_ep = OrderedDict()
        plot_wv = OrderedDict()
        plot_fl = OrderedDict()
        plot_un = OrderedDict()
        plot_ep = OrderedDict()
        fit_noise = {}
        plot_noise = {}
        mjd_dict = {}

        for ln_name, ln_info in spectral_lines_info.items():
            lid = f"line_{int(ln_info['rest_wave']*10)}"
            fit_wv[lid] = []
            fit_fl[lid] = []
            fit_un[lid] = []
            fit_ep[lid] = []
            fit_noise[lid] = {}

            plot_wv[lid] = []
            plot_fl[lid] = []
            plot_un[lid] = []
            plot_ep[lid] = []
            plot_noise[lid] = {}

        for (ep, filepath) in tqdm(epoch_files, desc=f"Loading data in {os.path.basename(sf)}"):
            df = load_data_for_epoch(filepath)
            if df.empty:
                continue
            # MJD
            if hasattr(df, 'attrs') and 'MJD' in df.attrs:
                mjd_dict[ep] = df.attrs['MJD']
            else:
                mjd_dict[ep] = np.nan

            for ln_name, ln_info in spectral_lines_info.items():
                lid = f"line_{int(ln_info['rest_wave'] * 10)}"
                restw = ln_info['rest_wave']
                halfw = ln_info['window']/2.0
                s_extra = 25.0
                smin = restw - s_extra
                smax = restw + s_extra
                mask_search = (df['wavelength'] >= smin) & (df['wavelength'] <= smax)
                wv_s = df['wavelength'][mask_search].values
                fl_s = df['flux'][mask_search].values
                if len(wv_s) < 3:
                    continue
                found_center = find_line_center_smoothed(wv_s, fl_s, sigma=1.0, absorption=True)
                if found_center is None:
                    found_center = restw

                lw_min = found_center - halfw
                lw_max = found_center + halfw
                mask_fit = (df['wavelength'] >= lw_min) & (df['wavelength'] <= lw_max)
                wv_fit = df['wavelength'][mask_fit].values
                fl_fit = df['flux'][mask_fit].values
                if len(wv_fit) < 3:
                    continue

                nrs = find_noise_regions(df, lw_min, lw_max, noise_window=5, min_noise_separation=5)
                if nrs:
                    sig_list = []
                    for (nm, nM, _) in nrs:
                        mmn = (df['wavelength'] >= nm) & (df['wavelength'] <= nM)
                        subn = df['flux'][mmn].values
                        if len(subn) > 0:
                            sig_list.append(subn.std())
                    if sig_list:
                        epoch_sigma = np.mean(sig_list)
                    else:
                        epoch_sigma = max(0.02, fl_fit.std())
                else:
                    epoch_sigma = max(0.02, fl_fit.std())

                fit_wv[lid].append(wv_fit)
                fit_fl[lid].append(fl_fit)
                fit_un[lid].append(np.full_like(fl_fit, epoch_sigma))
                fit_ep[lid].append(np.full_like(fl_fit, ep, dtype=int))
                fit_noise[lid].setdefault(ep, [])
                fit_noise[lid][ep] = nrs

                # For bigger plotting range
                edges_min = [lw_min]
                edges_max = [lw_max]
                for (nm, nM, dr) in nrs:
                    edges_min.append(nm)
                    edges_max.append(nM)
                pm = min(edges_min)
                pM = max(edges_max)
                if pM <= pm:
                    pm = lw_min
                    pM = lw_max
                mask_plot = (df['wavelength'] >= pm) & (df['wavelength'] <= pM)
                wv_p = df['wavelength'][mask_plot].values
                fl_p = df['flux'][mask_plot].values
                if len(wv_p) < 3:
                    continue
                plot_wv[lid].append(wv_p)
                plot_fl[lid].append(fl_p)
                plot_un[lid].append(np.full_like(fl_p, epoch_sigma))
                plot_ep[lid].append(np.full_like(fl_p, ep, dtype=int))
                plot_noise[lid].setdefault(ep, [])
                plot_noise[lid][ep] = nrs

        # Flatten
        all_ep = set()
        for lid in fit_wv:
            if fit_wv[lid]:
                fit_wv[lid] = np.concatenate(fit_wv[lid])
                fit_fl[lid] = np.concatenate(fit_fl[lid])
                fit_un[lid] = np.concatenate(fit_un[lid])
                fit_ep[lid] = np.concatenate(fit_ep[lid])
                all_ep.update(fit_ep[lid].tolist())
            else:
                fit_wv[lid] = np.array([])
                fit_fl[lid] = np.array([])
                fit_un[lid] = np.array([])
                fit_ep[lid] = np.array([])

        for lid in plot_wv:
            if plot_wv[lid]:
                plot_wv[lid] = np.concatenate(plot_wv[lid])
                plot_fl[lid] = np.concatenate(plot_fl[lid])
                plot_un[lid] = np.concatenate(plot_un[lid])
                plot_ep[lid] = np.concatenate(plot_ep[lid])
            else:
                plot_wv[lid] = np.array([])
                plot_fl[lid] = np.array([])
                plot_un[lid] = np.array([])
                plot_ep[lid] = np.array([])

        all_ep = np.unique(list(all_ep))
        if not all_ep.size:
            print("No data found in this subfolder. Skipping.")
            continue

        # central map
        central_map = {}
        for ln_name, ln_info in spectral_lines_info.items():
            lid = f"line_{int(ln_info['rest_wave']*10)}"
            central_map[lid] = ln_info['rest_wave']

        # Build Minimizer
        minimizer = Minimizer(
            residuals,
            params_in,
            fcn_args=(fit_wv, fit_fl, fit_ep, fit_un, central_map),
            fcn_kws=dict(
                profile_type=profile_type,
                line_profile=line_profile,
                weighted=use_weighted
            )
        )

        # Evaluate initial chi^2
        old_chi2 = compute_chi_squared(params_in, minimizer,
                                       fit_wv, fit_fl, fit_ep, fit_un,
                                       central_map,
                                       profile_type=profile_type,
                                       line_profile=line_profile,
                                       weighted=use_weighted)
        print(f"Initial total chi^2 = {old_chi2:.3f}")

        # Attempt epoch swaps
        ep_list = np.unique(list(all_ep))
        n_improved = 0
        for e in ep_list:
            old_vals = apply_epoch_rv_swap(params_in, e)
            if old_vals is None:
                # no rv1_epoch / rv2_epoch for that ep => skip
                continue
            new_chi2 = compute_chi_squared(params_in, minimizer,
                                           fit_wv, fit_fl, fit_ep, fit_un,
                                           central_map,
                                           profile_type=profile_type,
                                           line_profile=line_profile,
                                           weighted=use_weighted)
            if new_chi2 < old_chi2:
                print(f"  Epoch {e}: swap improves chi^2 {old_chi2:.3f} -> {new_chi2:.3f}")
                old_chi2 = new_chi2
                n_improved += 1
            else:
                revert_epoch_rv_swap(params_in, e, old_vals)

        if n_improved > 0:
            print(f"Swaps improved in {n_improved} epochs. Now local refine with leastsq.")
        else:
            print("No epoch swaps improved chi^2. Doing a final local refine anyway...")

        # Final local refine
        result_final = minimizer.minimize(
            method='leastsq',
            params=params_in,
            max_nfev=20000
        )
        print("\n===== Post-Swap Fit Report =====\n")
        report_fit(result_final)

        # We'll create a new folder "swap_refined_fit_results" for final outputs
        swap_out_dir = os.path.join(sf, "swap_refined_fit_results")
        os.makedirs(swap_out_dir, exist_ok=True)

        # Summaries & final plots
        windows_map = {}
        for ln_name, ln_info in spectral_lines_info.items():
            lid = f"line_{int(ln_info['rest_wave']*10)}"
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
            output_directory=swap_out_dir,
            profile_type=profile_type,
            line_profile=line_profile,
            plot_wavelengths_line=plot_wv,
            plot_fluxes_line=plot_fl,
            plot_uncertainties_line=plot_un,
            plot_epochs_line=plot_ep,
            plot_noise_dict=plot_noise,
            mcmc_chain=None,
            mjd_dict=mjd_dict
        )

        # Save new param JSON
        out_json = os.path.join(swap_out_dir, "standard_fit_params_swapped.json")
        save_params_json(result_final.params, out_json)

        print(f"\nAll done with subfolder: {sf}")
        print(f"  => Results in {swap_out_dir}\n")


if __name__ == "__main__":
    main()
