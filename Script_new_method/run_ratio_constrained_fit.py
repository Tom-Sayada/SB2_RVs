#!/usr/bin/env python

"""
run_ratio_constrained_fit.py

Implementation of ratio + v_sys approach:
    (rv1 + v_sys) = -(ratio)*(rv2 + v_sys)

Auto-locates standard-fit JSON if not provided, uses linear regression of (rv1, rv2)
to guess ratio & v_sys, and sets rv2_epoch for each epoch.
"""

import os
import argparse
import json
import numpy as np
from tqdm import tqdm
from lmfit import Minimizer, report_fit

# Local imports
from src.utils import (
    find_observation_files,
    load_data_for_epoch,
    find_line_center_smoothed,
    find_noise_regions
)
# Removed import of c_speed and grid_search_rvs from model_builder,
# since they're not actually used here.

from src.model_builder import build_output_subfolder_name  # If you have such a function, or remove if not used
from src.ratio_fit_model import (
    setup_parameters as ratio_setup,
    residuals
)
from src.plot_results import report_fit_results


def auto_locate_standard_json(data_dir, profile_type, use_weighted, fit_baseline):
    w_str = "weighted" if use_weighted else "unweighted"
    b_str = "baseline" if fit_baseline else "nobaseline"
    subfolder = f"standard_{profile_type}_{w_str}_{b_str}_fit_results"
    candidate_json = os.path.join(data_dir, subfolder, "standard_fit_params.json")
    if os.path.exists(candidate_json):
        return candidate_json
    return None


def load_standard_fit_json(json_path):
    if not os.path.isfile(json_path):
        print(f"Warning: no JSON found at: {json_path}")
        return {}
    with open(json_path, 'r') as f:
        data = json.load(f)
    out = {}
    for k, subd in data.items():
        val = subd.get("value", 0.0)
        vmin = subd.get("min", None)
        vmax = subd.get("max", None)
        out[k] = (val, vmin, vmax)
    print(f"Loaded {len(out)} parameters from standard-fit JSON: {json_path}")
    return out


def derive_ratio_vsys_linfit(standard_params):
    import numpy as np
    rv_dict = {}
    for name, (val, _, _) in standard_params.items():
        if name.startswith("rv1_epoch"):
            ep_str = name.replace("rv1_epoch", "")
            try:
                ep = int(ep_str)
                rv_dict.setdefault(ep, [None, None])[0] = val
            except:
                pass
        elif name.startswith("rv2_epoch"):
            ep_str = name.replace("rv2_epoch", "")
            try:
                ep = int(ep_str)
                rv_dict.setdefault(ep, [None, None])[1] = val
            except:
                pass

    rv1_list = []
    rv2_list = []
    ep_list = []
    for ep, (r1, r2) in sorted(rv_dict.items()):
        if (r1 is not None) and (r2 is not None):
            rv1_list.append(r1)
            rv2_list.append(r2)
            ep_list.append(ep)

    if len(rv1_list) < 2:
        ratio_est = 1.0
        v_sys_est = 0.0
        rv2_init_dict = {ep: (r2 if r2 else 0.0) for ep, (r1, r2) in rv_dict.items()}
        return ratio_est, v_sys_est, rv2_init_dict

    rv1_arr = np.array(rv1_list)
    rv2_arr = np.array(rv2_list)

    # Fit: rv1 = intercept + slope*rv2
    slope, intercept = np.polyfit(rv2_arr, rv1_arr, 1)
    b = slope
    a = intercept

    # ratio = -b, v_sys = -a / (ratio+1)
    ratio_est = -b
    if abs(ratio_est + 1) < 1e-8:
        ratio_est = 1.0
    v_sys_est = -a / (ratio_est + 1)

    if ratio_est < 0.01:
        ratio_est = 0.01
    elif ratio_est > 20.0:
        ratio_est = 20.0

    rv2_init_dict = {}
    for ep, (r1, r2) in rv_dict.items():
        if r2 is None:
            r2 = 0.0
        rv2_init_dict[ep] = r2

    return ratio_est, v_sys_est, rv2_init_dict


class BasinHoppingCallback:
    def __init__(self, niter):
        self.niter = niter
        self.current = 0
        self.pbar = tqdm(total=niter, desc="Basin hopping")

    def __call__(self, x, f, accept):
        self.current += 1
        self.pbar.update(1)
        return False

    def close(self):
        self.pbar.close()


def main():
    parser = argparse.ArgumentParser(
        description="Ratio+v_sys SB2 fit with auto-locating standard-fit JSON."
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="output_ratio_fit")
    parser.add_argument("--profile_type", type=str, default='sym')
    parser.add_argument("--lines", type=str, default=None)
    parser.add_argument("--unweighted", action='store_true')
    parser.add_argument("--fit_baseline", action='store_true')
    parser.add_argument("--init_params_json", type=str, default=None,
                        help="Path to standard-fit JSON (if not, auto-locate).")

    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.output_dir
    profile_type = args.profile_type.strip().lower()
    use_weighted = (not args.unweighted)
    fit_baseline = bool(args.fit_baseline)

    os.makedirs(out_dir, exist_ok=True)

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

    print(f"\nData dir: {data_dir}")
    print(f"Output:   {out_dir}")
    print(f"Lines:    {list(spectral_lines.keys())}")
    print(f"Profile: '{profile_type}', Weighted={use_weighted}, Baseline={fit_baseline}")

    # Auto-locate JSON if not passed
    if not args.init_params_json:
        auto_found = auto_locate_standard_json(
            data_dir=data_dir,
            profile_type=profile_type,
            use_weighted=use_weighted,
            fit_baseline=fit_baseline
        )
        if auto_found:
            print(f"\nAuto-located standard-fit JSON at: {auto_found}")
            args.init_params_json = auto_found
        else:
            print("\nNo standard-fit JSON found => default init guesses.")
    else:
        print(f"\nWill attempt to initialize from: {args.init_params_json}")

    epoch_files = find_observation_files(data_dir)
    if not epoch_files:
        raise ValueError(f"No files found in {data_dir}.")

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

    # Build arrays
    print(f"\nFound {len(epoch_files)} observation files. Building arrays now...")
    for (ep, filepath) in tqdm(epoch_files, desc="Loading data"):
        df = load_data_for_epoch(filepath)
        if df.empty:
            continue

        for ln_name, ln_info in spectral_lines.items():
            line_id = f"line_{int(ln_info['rest_wave']*10)}"
            restw = ln_info['rest_wave']
            halfw = ln_info['window']/2

            # Big search window
            search_extra = 25.0
            smin = restw - search_extra
            smax = restw + search_extra
            msearch = (df['wavelength'] >= smin) & (df['wavelength'] <= smax)
            wv_search = df['wavelength'][msearch].values
            fl_search = df['flux'][msearch].values
            if len(wv_search) < 5:
                continue

            found_center = find_line_center_smoothed(wv_search, fl_search, sigma=1.0, absorption=True)
            if found_center is None:
                found_center = restw

            lw_min = found_center - halfw
            lw_max = found_center + halfw
            mm_fit = (df['wavelength'] >= lw_min) & (df['wavelength'] <= lw_max)
            wv_fit = df['wavelength'][mm_fit].values
            fl_fit = df['flux'][mm_fit].values
            if len(wv_fit) == 0:
                continue

            nrs = find_noise_regions(df, lw_min, lw_max, noise_window=5, min_separation=5, max_offset=30)
            if nrs:
                sig_list = []
                for (nm, nM, _) in nrs:
                    mm_noise = (df['wavelength'] >= nm) & (df['wavelength'] <= nM)
                    fl_n = df['flux'][mm_noise].values
                    if len(fl_n):
                        sig_list.append(fl_n.std())
                if sig_list:
                    epoch_sigma = np.mean(sig_list)
                else:
                    epoch_sigma = max(0.02, fl_fit.std())
            else:
                epoch_sigma = max(0.02, fl_fit.std())

            fit_wv.setdefault(line_id, []).append(wv_fit)
            fit_fl.setdefault(line_id, []).append(fl_fit)
            fit_un.setdefault(line_id, []).append(np.full_like(fl_fit, epoch_sigma))
            fit_ep.setdefault(line_id, []).append(np.full_like(fl_fit, ep, dtype=int))
            fit_noise.setdefault(line_id, {})
            fit_noise[line_id][ep] = nrs

            # Plot arrays
            edges_min = [lw_min] + [nm for nm, _, _ in nrs]
            edges_max = [lw_max] + [nM for _, nM, _ in nrs]
            pm = min(edges_min)
            pM = max(edges_max)
            if pM <= pm:
                pm = lw_min
                pM = lw_max
            mplot = (df['wavelength'] >= pm) & (df['wavelength'] <= pM)
            wv_plot = df['wavelength'][mplot].values
            fl_plot = df['flux'][mplot].values
            if len(wv_plot) == 0:
                continue

            plot_wv.setdefault(line_id, []).append(wv_plot)
            plot_fl.setdefault(line_id, []).append(fl_plot)
            plot_un.setdefault(line_id, []).append(np.full_like(fl_plot, epoch_sigma))
            plot_ep.setdefault(line_id, []).append(np.full_like(fl_plot, ep, dtype=int))
            plot_noise.setdefault(line_id, {})
            plot_noise[line_id][ep] = nrs

    # Flatten
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
        lid = f"line_{int(ln_info['rest_wave']*10)}"
        central_map[lid] = ln_info['rest_wave']

    loaded_params = {}
    if args.init_params_json and os.path.exists(args.init_params_json):
        loaded_params = load_standard_fit_json(args.init_params_json)
    else:
        print("No standard-fit JSON path or file not found => default guesses.")

    # Derive ratio, v_sys from standard-fit if possible
    ratio_est = 1.0
    v_sys_est = 0.0
    rv2_init_dict = {}
    if loaded_params:
        ratio_est, v_sys_est, rv2_init_dict = derive_ratio_vsys_linfit(loaded_params)

    # Setup ratio-based parameters
    params = ratio_setup(
        central_wavelengths=central_map,
        all_epochs=all_ep,
        profile_type=profile_type,
        fit_baseline=fit_baseline,
        initial_rvs=rv2_init_dict,
        initial_state=loaded_params,
        ratio_estimate=ratio_est,
        vsys_estimate=v_sys_est
    )

    # Overwrite line-shape param guesses if present
    for p_name in params.keys():
        if p_name in loaded_params:
            val, vmin, vmax = loaded_params[p_name]
            params[p_name].set(value=val)
            # if vmin is not None: params[p_name].min = vmin
            # if vmax is not None: params[p_name].max = vmax

    # Minimizer
    minimizer = Minimizer(
        residuals,
        params,
        fcn_args=(fit_wv, fit_fl, fit_ep, fit_un, central_map),
        fcn_kws={'profile_type': profile_type, 'weighted': use_weighted, 'fit_baseline': fit_baseline}
    )

    print("\nPerforming basin hopping optimization...")
    callback = BasinHoppingCallback(niter=10)
    try:
        result_bh = minimizer.minimize(
            method='basinhopping',
            niter=10,
            T=5.0,
            stepsize=0.3,
            callback=callback,
            minimizer_kwargs={'method': 'L-BFGS-B'}
        )
    finally:
        callback.close()

    print("\nRefining with least squares...")
    result_final = minimizer.minimize(
        method='leastsq',
        params=result_bh.params,
        max_nfev=20000
    )

    print("\n===== Final Fit Report =====\n")
    report_fit(result_final)

    windows_map = {}
    for ln_name, ln_info in spectral_lines.items():
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
        output_directory=out_dir,
        profile_type=profile_type,
        plot_wavelengths_line=plot_wv,
        plot_fluxes_line=plot_fl,
        plot_uncertainties_line=plot_un,
        plot_epochs_line=plot_ep,
        plot_noise_dict=plot_noise
    )

    print(f"\nAll done. Results saved in {out_dir}\n")
    if 'ratio' in result_final.params:
        print(f"Final ratio = {result_final.params['ratio'].value:.3f} ± {result_final.params['ratio'].stderr or 0.0:.3f}")
    if 'v_sys' in result_final.params:
        print(f"Final v_sys = {result_final.params['v_sys'].value:.2f} ± {result_final.params['v_sys'].stderr or 0.0:.2f} km/s")


if __name__ == "__main__":
    main()