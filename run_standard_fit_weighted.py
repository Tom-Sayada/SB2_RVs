#!/usr/bin/env python3
# run_standard_fit_weighted.py

import os
import argparse
import json
import numpy as np
from tqdm import tqdm
from lmfit import Minimizer, report_fit

# Local modules, but referencing the weighted versions
from src.utils import (
    find_observation_files,
    load_data_for_epoch,
    find_noise_regions,
    find_line_center_smoothed
)
# Import the weighted model builder functions
from src.model_builder_weighted import (
    compute_full_model,
    _compute_rv1_rv2,
    residuals,
    setup_parameters,
    iterate_with_weights,
    ModelError
)
from src.plot_results import report_fit_results


def main():
    parser = argparse.ArgumentParser(
        description="SB2 multi-line fit (standard), with separation-based weighting."
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="output_fit_results_weighted")
    parser.add_argument("--profile_type", type=str, default='sym')
    parser.add_argument("--line_profile", type=str, default='voigt')
    parser.add_argument("--lines", type=str, default=None)
    parser.add_argument("--unweighted", action='store_true')
    parser.add_argument("--fit_baseline", action='store_true')
    parser.add_argument("--save_params_json", type=str, default="standard_fit_params_weighted.json")
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
    use_weighted = (not args.unweighted)
    fit_baseline = args.fit_baseline
    save_path = args.save_params_json
    user_requested_mcmc = args.mcmc
    min_weight = args.min_weight
    max_sep = args.max_sep
    max_iterations = args.max_iterations

    os.makedirs(out_dir, exist_ok=True)

    # Example lines
    all_lines_info = {
        'He4471': {'rest_wave': 4471.5, 'window': 7.0},
        'He4026': {'rest_wave': 4026.0, 'window': 7.0},
        'He4388': {'rest_wave': 4388.0, 'window': 7.0},
        'H4340': {'rest_wave': 4340.472, 'window': 7.0}
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

    mjd_dict = {}

    for ln_name, ln_info in spectral_lines.items():
        line_id = f"line_{int(ln_info['rest_wave'] * 10)}"
        fit_wv[line_id] = []
        fit_fl[line_id] = []
        fit_un[line_id] = []
        fit_ep[line_id] = []
        fit_noise[line_id] = {}

        plot_wv[line_id] = []
        plot_fl[line_id] = []
        plot_un[line_id] = []
        plot_ep[line_id] = []
        plot_noise[line_id] = {}

    for (ep, filepath) in tqdm(epoch_files, desc="Loading data"):
        df = load_data_for_epoch(filepath)
        if df.empty:
            continue
        if hasattr(df, 'attrs') and 'MJD' in df.attrs:
            mjd_dict[ep] = df.attrs['MJD']
        else:
            mjd_dict[ep] = np.nan

        for ln_name, ln_info in spectral_lines.items():
            line_id = f"line_{int(ln_info['rest_wave'] * 10)}"
            restw = ln_info['rest_wave']
            halfw = ln_info['window'] / 2.0

            # Use NumPy arrays directly to avoid endianness issues
            wavelength_array = np.asarray(df['wavelength'].values, dtype='<f8')
            flux_array = np.asarray(df['flux'].values, dtype='<f8')

            search_extra = 25.0
            smin = restw - search_extra
            smax = restw + search_extra
            mask = (wavelength_array >= smin) & (wavelength_array <= smax)
            wv_s = wavelength_array[mask]
            fl_s = flux_array[mask]

            if len(wv_s) < 5:
                continue

            found_center = find_line_center_smoothed(wv_s, fl_s, sigma=1.0, absorption=True)
            if found_center is None:
                found_center = restw

            lw_min = found_center - halfw
            lw_max = found_center + halfw

            # Again use NumPy arrays for the actual fit region
            mask_fit = (wavelength_array >= lw_min) & (wavelength_array <= lw_max)
            wv_fit = wavelength_array[mask_fit]
            fl_fit = flux_array[mask_fit]

            if len(wv_fit) == 0:
                continue

            nrs = find_noise_regions(df, lw_min, lw_max, noise_window=5, min_noise_separation=5)
            if nrs:
                sig_list = []
                for (nm, nM, _) in nrs:
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

            fit_wv[line_id].append(wv_fit)
            fit_fl[line_id].append(fl_fit)
            fit_un[line_id].append(np.full_like(fl_fit, epoch_sigma))
            fit_ep[line_id].append(np.full_like(fl_fit, ep, dtype=int))
            fit_noise[line_id].setdefault(ep, [])
            fit_noise[line_id][ep] = nrs

            # for plotting
            edges_min = [lw_min] + [x[0] for x in nrs]
            edges_max = [lw_max] + [x[1] for x in nrs]
            pm = min(edges_min)
            pM = max(edges_max)
            if pM <= pm:
                pm, pM = lw_min, lw_max

            plot_mask = (wavelength_array >= pm) & (wavelength_array <= pM)
            wv_plot = wavelength_array[plot_mask]
            fl_plot = flux_array[plot_mask]

            if len(wv_plot) > 0:
                plot_wv[line_id].append(wv_plot)
                plot_fl[line_id].append(fl_plot)
                plot_un[line_id].append(np.full_like(fl_plot, epoch_sigma))
                plot_ep[line_id].append(np.full_like(fl_plot, ep, dtype=int))
                plot_noise[line_id].setdefault(ep, [])
                plot_noise[line_id][ep] = nrs

    all_ep = set()
    for lid in fit_wv:
        if len(fit_wv[lid]) > 0:
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

    all_ep = np.unique(list(all_ep))
    if len(all_ep) == 0:
        print("No data => exit.")
        return

    central_map = {}
    for ln_name, ln_info in spectral_lines.items():
        lid = f"line_{int(ln_info['rest_wave'] * 10)}"
        central_map[lid] = ln_info['rest_wave']

    # Use the setup_parameters function from model_builder_weighted
    params = setup_parameters(
        central_wavelengths=central_map,
        all_epochs=all_ep,
        profile_type=profile_type,
        fit_baseline=fit_baseline,
        line_profile=line_profile
    )

    # Create the minimizer with weighted residuals
    minimizer = Minimizer(
        residuals,
        params,
        fcn_args=(fit_wv, fit_fl, fit_ep, fit_un, central_map),
        fcn_kws={
            'profile_type': profile_type,
            'line_profile': line_profile,
            'weighted': use_weighted,
            'min_weight': min_weight,
            'max_sep': max_sep
        }
    )

    # Use the iterative weighting approach if requested
    if use_weighted:
        print(f"Running weighted fit with separation-based weighting (min_weight={min_weight}, max_sep={max_sep})")
        result_final = iterate_with_weights(
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

    if user_requested_mcmc:
        # If you want to implement MCMC, you would do it here
        # Example if you had an mcmc_fit function:
        # result_mcmc = mcmc_fit(result_final, minimizer, nwalkers=20, steps=1000)
        pass

    # Summaries & final plots
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
        plot_wavelengths_line=plot_wv,
        plot_fluxes_line=plot_fl,
        plot_uncertainties_line=plot_un,
        plot_epochs_line=plot_ep,
        plot_noise_dict=plot_noise,
        mjd_dict=mjd_dict
    )

    # Save final params
    param_data = {}
    for pname, pval in result_final.params.items():
        param_data[pname] = {
            "value": pval.value,
            "min": pval.min,
            "max": pval.max
        }
    with open(os.path.join(out_dir, save_path), 'w') as f:
        json.dump(param_data, f, indent=2)
    print(f"Saved best-fit parameters to '{save_path}' in {out_dir}")


if __name__ == "__main__":
    main()