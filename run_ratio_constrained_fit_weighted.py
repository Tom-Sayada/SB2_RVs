#!/usr/bin/env python3
# run_ratio_constrained_fit_weighted.py

"""
Implementation of ratio + v_sys approach with separation weighting:
  (rv1 + v_sys) = - ratio * (rv2 + v_sys)

Mirrors run_ratio_constrained_fit.py but uses the weighted model.
"""

import os
import argparse
import json
import numpy as np
from tqdm import tqdm
from lmfit import Minimizer, report_fit

# local modules
from src.utils import (
    find_observation_files,
    load_data_for_epoch,
    find_noise_regions,
    find_line_center_smoothed
)
# Import from ratio_fit_model_weighted
import src.ratio_fit_model_weighted as ratio_fit_model
from src.plot_results import report_fit_results


def main():
    parser = argparse.ArgumentParser(
        description="Ratio+v_sys SB2 fit with separation weighting."
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="output_ratio_fit_weighted")
    parser.add_argument("--profile_type", type=str, default='sym')
    parser.add_argument("--line_profile", type=str, default='voigt')
    parser.add_argument("--lines", type=str, default=None)
    parser.add_argument("--unweighted", action='store_true')
    parser.add_argument("--fit_baseline", action='store_true')
    parser.add_argument("--mcmc", action='store_true')
    parser.add_argument("--min_weight", type=float, default=0.2,
                        help="Minimum weight for near-blended epochs (0-1)")
    parser.add_argument("--max_sep", type=float, default=400.0,
                        help="RV separation in km/s where weight becomes 1.0")
    parser.add_argument("--max_iterations", type=int, default=3,
                        help="Maximum number of weight update iterations")
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.output_dir
    profile_type = args.profile_type.strip().lower()
    line_profile = args.line_profile.strip().lower()
    weighted = (not args.unweighted)
    fit_baseline = args.fit_baseline
    min_weight = args.min_weight
    max_sep = args.max_sep
    max_iterations = args.max_iterations

    os.makedirs(out_dir, exist_ok=True)

    # example lines
    all_lines_info = {
        'He4471': {'rest_wave': 4471.5, 'window': 20.0},
        'He4026': {'rest_wave': 4026.0, 'window': 20.0},
        'He4388': {'rest_wave': 4388.0, 'window': 20.0},
        'H4340': {'rest_wave': 4340.472, 'window': 20.0}
    }

    spectral_lines = all_lines_info
    if args.lines:
        requested = [ln.strip() for ln in args.lines.split(',') if ln.strip()]
        spectral_lines = {}
        for r in requested:
            if r not in all_lines_info:
                raise ValueError(f"Line '{r}' not recognized.")
            spectral_lines[r] = all_lines_info[r]

    epoch_files = find_observation_files(data_dir)
    if not epoch_files:
        print(f"No valid files in {data_dir}")
        return

    from collections import defaultdict
    fit_wv = {}
    fit_fl = {}
    fit_un = {}
    fit_ep = {}
    fit_noise = {}
    mjd_dict = {}

    for ln_name, ln_info in spectral_lines.items():
        lid = f"line_{int(ln_info['rest_wave'] * 10)}"
        fit_wv[lid] = []
        fit_fl[lid] = []
        fit_un[lid] = []
        fit_ep[lid] = []
        fit_noise[lid] = {}

    for (ep, filepath) in tqdm(epoch_files, desc="Loading data"):
        df = load_data_for_epoch(filepath)
        if df.empty:
            continue
        if hasattr(df, 'attrs') and 'MJD' in df.attrs:
            mjd_dict[ep] = df.attrs['MJD']
        else:
            mjd_dict[ep] = np.nan

        for ln_name, ln_info in spectral_lines.items():
            lid = f"line_{int(ln_info['rest_wave'] * 10)}"
            restw = ln_info['rest_wave']
            halfw = ln_info['window'] / 2.0

            # Use NumPy arrays directly to avoid endianness issues
            wavelength_array = np.asarray(df['wavelength'].values, dtype='<f8')
            flux_array = np.asarray(df['flux'].values, dtype='<f8')

            # same approach as your ratio fit script
            search_extra = 25.0
            smin = restw - search_extra
            smax = restw + search_extra
            mask = (wavelength_array >= smin) & (wavelength_array <= smax)
            wv_s = wavelength_array[mask]
            fl_s = flux_array[mask]

            if len(wv_s) < 5:
                continue

            center_est = find_line_center_smoothed(wv_s, fl_s, sigma=1.0, absorption=True)
            if center_est is None:
                center_est = restw
            lw_min = center_est - halfw
            lw_max = center_est + halfw

            # Use mask for fit region
            mask_fit = (wavelength_array >= lw_min) & (wavelength_array <= lw_max)
            wv_fit = wavelength_array[mask_fit]
            fl_fit = flux_array[mask_fit]

            if len(wv_fit) == 0:
                continue

            # noise
            nrs = find_noise_regions(df, lw_min, lw_max, noise_window=5, min_noise_separation=5)
            if nrs:
                sig_list = []
                for (nm, nM, _) in nrs:
                    # Use mask for noise regions
                    noise_mask = (wavelength_array >= nm) & (wavelength_array <= nM)
                    flux_noise = flux_array[noise_mask]
                    if len(flux_noise) > 0:
                        sig_list.append(np.std(flux_noise))
                if sig_list:
                    epoch_sigma = np.mean(sig_list)
                else:
                    epoch_sigma = max(0.02, np.std(fl_fit))
            else:
                epoch_sigma = max(0.02, np.std(fl_fit))

            fit_wv[lid].append(wv_fit)
            fit_fl[lid].append(fl_fit)
            fit_un[lid].append(np.full_like(fl_fit, epoch_sigma))
            fit_ep[lid].append(np.full_like(fl_fit, ep, dtype=int))
            fit_noise[lid].setdefault(ep, [])
            fit_noise[lid][ep] = nrs

    # flatten etc. same as normal
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

    all_ep = np.unique(list(all_ep))
    if len(all_ep) == 0:
        print("No data => exit.")
        return

    central_map = {}
    for ln_name, ln_info in spectral_lines.items():
        lid = f"line_{int(ln_info['rest_wave'] * 10)}"
        central_map[lid] = ln_info['rest_wave']

    # Build ratio-based initial params
    params = ratio_fit_model.setup_parameters(
        central_wavelengths=central_map,
        all_epochs=all_ep,
        profile_type=profile_type,
        fit_baseline=fit_baseline,
        line_profile=line_profile
    )

    # Minimizer
    minimizer = Minimizer(
        ratio_fit_model.residuals,
        params,
        fcn_args=(fit_wv, fit_fl, fit_ep, fit_un, central_map),
        fcn_kws={
            'profile_type': profile_type,
            'line_profile': line_profile,
            'weighted': weighted,
            'min_weight': min_weight,
            'max_sep': max_sep
        }
    )

    # Use the iterative weighting approach if requested
    if weighted:
        print(f"Running weighted fit with separation-based weighting (min_weight={min_weight}, max_sep={max_sep})")
        result_final = ratio_fit_model.iterate_with_weights(
            params=params,
            minimizer=minimizer,
            all_epochs=all_ep,
            min_weight=min_weight,
            max_sep=max_sep,
            max_iterations=max_iterations,
            verbose=True
        )
    else:
        # Standard unweighted fit
        print("Running standard unweighted fit")
        result_final = minimizer.minimize(method='leastsq', max_nfev=20000)

    report_fit(result_final)

    # optional MCMC
    if args.mcmc:
        # Add your MCMC implementation here if desired
        pass

    # final plotting
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
        profile_type=profile_type,
        line_profile=line_profile,
        mjd_dict=mjd_dict
    )

    # save final params
    out_json = os.path.join(out_dir, "ratio_fit_params_weighted.json")
    data = {}
    for k, v in result_final.params.items():
        data[k] = {
            "value": v.value,
            "min": v.min,
            "max": v.max
        }
    with open(out_json, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved improved parameters to: {out_json}")


if __name__ == "__main__":
    main()