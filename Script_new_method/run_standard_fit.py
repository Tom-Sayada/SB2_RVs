#!/usr/bin/env python

import os
import sys  # ADDED THIS so we can call sys.exit(...)
import argparse
import json
import numpy as np
from tqdm import tqdm
from lmfit import Minimizer, report_fit
import logging

# Local modules
from src.utils import (
    find_observation_files,
    load_data_for_epoch,
    find_noise_regions,
    find_line_center_smoothed
)
from src.model_builder import (
    setup_parameters,
    residuals
)
from src.plot_results import report_fit_results

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def save_bestfit_params_to_json(params, filepath):
    """Save the best-fit parameters to a JSON file"""
    data = {}
    for key, par in params.items():
        data[key] = {
            "value": par.value,
            "min": par.min,
            "max": par.max
        }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    logger.info(f"Saved best-fit parameters to '{filepath}'")


def perform_optimization(minimizer, description="Optimization"):
    """
    Perform two-stage optimization with progress reporting.
    First stage: basin-hopping with 5 iterations (TQDM bar).
    Second stage: leastsq refinement.
    """
    logger.info(f"Starting {description}")

    # Stage 1: quick basin-hopping with TQDM
    callback = tqdm(total=10, desc="Basin hopping")

    def update_progress(x, f, accept):
        callback.update(1)
        return False

    try:
        result_bh = minimizer.minimize(
            method='basinhopping',
            niter=10,
            T=5.0,
            stepsize=0.3,
            callback=update_progress,
            minimizer_kwargs={
                'method': 'L-BFGS-B',
                'options': {'maxiter': 200}
            }
        )
    finally:
        callback.close()

    logger.info("Refining solution")

    # Stage 2: leastsq
    result_final = minimizer.minimize(
        method='leastsq',
        params=result_bh.params,
        max_nfev=500,
        ftol=1e-4,
        xtol=1e-4
    )

    return result_final


def main():
    parser = argparse.ArgumentParser(
        description="Optimized SB2 spectral line fitting (Standard Fit)"
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="output_fit_results")
    parser.add_argument("--profile_type", type=str, default='sym')
    parser.add_argument("--lines", type=str, default=None)
    parser.add_argument("--unweighted", action='store_true')
    parser.add_argument("--fit_baseline", action='store_true')
    parser.add_argument("--save_params_json", type=str, default="standard_fit_params.json")

    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.output_dir
    profile_type = args.profile_type.strip().lower()
    use_weighted = not args.unweighted
    fit_baseline = bool(args.fit_baseline)

    os.makedirs(out_dir, exist_ok=True)

    # Define spectral lines
    all_lines_info = {
        'He4471': {'rest_wave': 4471.5, 'window': 20.0},
        'He4026': {'rest_wave': 4026.0, 'window': 20.0},
        'He4388': {'rest_wave': 4388.0, 'window': 20.0},
        'H4340':  {'rest_wave': 4340.472, 'window': 20.0}
    }

    if args.lines:
        requested = [ln.strip() for ln in args.lines.split(',') if ln.strip()]
        spectral_lines = {}
        for r in requested:
            if r not in all_lines_info:
                raise ValueError(f"Line '{r}' not recognized.")
            spectral_lines[r] = all_lines_info[r]
    else:
        spectral_lines = all_lines_info

    logger.info(f"\nData from: {data_dir}")
    logger.info(f"Output:   {out_dir}")
    logger.info(f"Fitting lines: {list(spectral_lines.keys())}")
    logger.info(f"Profile type: {profile_type}, Weighted={use_weighted}, Baseline={fit_baseline}\n")

    # Find observation files
    epoch_files = find_observation_files(data_dir)
    if not epoch_files:
        logger.error(f"No files found in {data_dir}")
        return 1

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

    logger.info(f"Found {len(epoch_files)} files. Building arrays now...")

    # Build data arrays
    for (ep, filepath) in tqdm(epoch_files, desc="Loading data"):
        df = load_data_for_epoch(filepath)
        if df.empty:
            continue

        for ln_name, ln_info in spectral_lines.items():
            line_id = f"line_{int(ln_info['rest_wave'] * 10)}"
            rest_wave = ln_info['rest_wave']
            half_window = ln_info['window'] / 2

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

            nrs = find_noise_regions(
                df, line_min, line_max,
                noise_window=5,
                min_separation=5,
                max_offset=30
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

            fit_wv[line_id].append(wv_fit)
            fit_fl[line_id].append(fl_fit)
            fit_un[line_id].append(np.full_like(fl_fit, epoch_sigma))
            fit_ep[line_id].append(np.full_like(fl_fit, ep, dtype=int))
            fit_noise[line_id][ep] = nrs

            # For plotting arrays
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

    logger.info("Processing arrays...")
    all_epochs = set()

    for lid in tqdm(fit_wv.keys(), desc="Concatenating arrays"):
        if len(fit_wv[lid]) > 0:
            fit_wv[lid] = np.concatenate(fit_wv[lid])
            fit_fl[lid] = np.concatenate(fit_fl[lid])
            fit_un[lid] = np.concatenate(fit_un[lid])
            fit_ep[lid] = np.concatenate(fit_ep[lid])
            all_epochs.update(fit_ep[lid].tolist())
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

    from numpy import unique
    all_epochs = unique(list(all_epochs))
    if len(all_epochs) == 0:
        logger.error("No valid data found")
        return 1

    central_map = {}
    for ln_name, ln_info in spectral_lines.items():
        lid = f"line_{int(ln_info['rest_wave'] * 10)}"
        central_map[lid] = ln_info['rest_wave']

    all_epochs = unique(list(all_epochs))
    if len(all_epochs) == 0:
        logger.error("No data found => exit.")
        return 1

    logger.info("\nInitializing model parameters...")
    params = setup_parameters(
        central_wavelengths=central_map,
        all_epochs=all_epochs,
        profile_type=profile_type,
        fit_baseline=fit_baseline
    )

    minimizer = Minimizer(
        residuals,
        params,
        fcn_args=(fit_wv, fit_fl, fit_ep, fit_un, central_map),
        fcn_kws={'profile_type': profile_type, 'weighted': use_weighted},
        nan_policy='omit'
    )

    result = perform_optimization(minimizer, description="Spectral line fitting")

    logger.info("\n===== Fit Report =====\n")
    report_fit(result)

    df_res, df_chi = report_fit_results(
        result=result,
        wavelengths_line=fit_wv,
        fluxes_line=fit_fl,
        uncertainties_line=fit_un,
        epochs_line=fit_ep,
        central_wavelengths=central_map,
        noise_regions=fit_noise,
        windows={f"line_{int(v['rest_wave']*10)}": v['window'] for v in spectral_lines.values()},
        output_directory=out_dir,
        profile_type=profile_type,
        plot_wavelengths_line=plot_wv,
        plot_fluxes_line=plot_fl,
        plot_uncertainties_line=plot_un,
        plot_epochs_line=plot_ep,
        plot_noise_dict=plot_noise
    )

    if args.save_params_json:
        save_bestfit_params_to_json(result.params,
                                    os.path.join(out_dir, args.save_params_json))

    logger.info(f"\nAll done. Results saved in: {out_dir}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())