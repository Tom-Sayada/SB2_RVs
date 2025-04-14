#!/usr/bin/env python3
# run_ratio_constrained_fit.py

"""
Implementation of ratio + v_sys approach:
  (rv1 + v_sys) = - ratio * (rv2 + v_sys)

Auto-locates standard-fit JSON if not provided, uses linear regression of (rv1, rv2)
to guess ratio & v_sys, and sets rv2_epoch for each epoch in the ratio-based fit.

Now also supports both Voigt and Gaussian profiles.

Additionally, if any key parameter's .stderr is NaN or zero after the LM fit
(e.g. ratio, v_sys, or rv2_epochN), an MCMC step is performed to derive a more
reliable uncertainty estimate. If the user also passes --mcmc, the MCMC step
is forced regardless.
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
from src.ratio_fit_model import (
    setup_parameters as ratio_setup,
    residuals
)
from src.plot_results import report_fit_results


###############################################################################
# (1) Helper to compute global chi^2 from final params
###############################################################################
def compute_global_chi2(params, minimizer,
                        fit_wv, fit_fl, fit_ep, fit_un,
                        central_wavelengths,
                        profile_type, line_profile,
                        weighted=True,
                        fit_baseline=False):
    """
    Compute total sum of squared residuals across all lines/epochs
    by calling the same residual function used in Minimizer.
    """
    res = minimizer.userfcn(
        params,
        fit_wv, fit_fl, fit_ep, fit_un, central_wavelengths,
        profile_type=profile_type,
        line_profile=line_profile,
        weighted=weighted,
        fit_baseline=fit_baseline
    )
    chi2 = np.sum(res**2)
    return chi2


###############################################################################
# (2) Main script function
###############################################################################
def main():
    parser = argparse.ArgumentParser(
        description="Ratio+v_sys SB2 fit with auto-locating standard-fit JSON + optional MCMC."
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="output_ratio_fit")
    parser.add_argument("--profile_type", type=str, default='sym')
    parser.add_argument("--line_profile", type=str, default='voigt',
                        choices=['voigt', 'gaussian'],
                        help="Profile type for stellar components")
    parser.add_argument("--lines", type=str, default=None)
    parser.add_argument("--unweighted", action='store_true')
    parser.add_argument("--fit_baseline", action='store_true')
    parser.add_argument("--init_params_json", type=str, default=None,
                        help="Path to standard-fit JSON (if not, auto-locate).")
    parser.add_argument("--mcmc", action='store_true',
                        help="Perform MCMC step after least-squares to compute ±1σ model band.")
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.output_dir
    profile_type = args.profile_type.strip().lower()
    line_profile = args.line_profile.strip().lower()
    use_weighted = (not args.unweighted)
    fit_baseline = bool(args.fit_baseline)
    user_requested_mcmc = args.mcmc

    os.makedirs(out_dir, exist_ok=True)

    # Example lines
    all_lines_info = {
        'He4471': {'rest_wave': 4471.5, 'window': 20.0},
        'He4026': {'rest_wave': 4026.0, 'window': 20.0},
        'He4388': {'rest_wave': 4388.0, 'window': 20.0},
        'H4340':  {'rest_wave': 4340.472, 'window': 20.0}
    }

    # If user specified lines
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
    print(f"Profile type: '{profile_type}', Line profile: {line_profile}")
    print(f"Weighted={use_weighted}, Baseline={fit_baseline}")
    if user_requested_mcmc:
        print("Will perform MCMC after final minimization => ±1σ flux envelopes.\n")

    # Possibly auto-locate JSON from standard fit
    def auto_locate_standard_json(data_dir, profile_type, line_profile, use_weighted, fit_baseline):
        w_str = "weighted" if use_weighted else "unweighted"
        b_str = "baseline" if fit_baseline else "nobaseline"
        subfolder = f"standard_{profile_type}_{line_profile}_{w_str}_{b_str}_fit_results"
        candidate_json = os.path.join(data_dir, subfolder, "standard_fit_params.json")
        if os.path.exists(candidate_json):
            return candidate_json
        return None

    if not args.init_params_json:
        auto_found = auto_locate_standard_json(
            data_dir=data_dir,
            profile_type=profile_type,
            line_profile=line_profile,
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

    # Identify observation files
    epoch_files = find_observation_files(data_dir)
    if not epoch_files:
        raise ValueError(f"No valid observation files found in {data_dir}.")

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

    # Dictionary to store MJD information for each epoch
    mjd_dict = {}
    from tqdm import tqdm
    # Build the arrays
    print(f"\nFound {len(epoch_files)} observation files. Building arrays now...")
    for (ep, filepath) in tqdm(epoch_files, desc="Loading data"):
        df = load_data_for_epoch(filepath)
        if df.empty:
            continue

        # Store MJD if available
        if hasattr(df, 'attrs') and 'MJD' in df.attrs:
            mjd_dict[ep] = df.attrs['MJD']
            print(f"Epoch {ep}: MJD = {df.attrs['MJD']}")
        else:
            mjd_dict[ep] = np.nan

        for ln_name, ln_info in spectral_lines.items():
            line_id = f"line_{int(ln_info['rest_wave'] * 10)}"
            restw = ln_info['rest_wave']
            halfw = ln_info['window'] / 2.0

            # Large search
            # Find approximate line center with a WIDER initial search (especially for high-velocity systems)
            search_extra = 25.0  # Increase from 25.0 to catch more shifted lines
            search_min = restw - search_extra
            search_max = restw + search_extra
            mask_search = (df['wavelength'] >= search_min) & (df['wavelength'] <= search_max)
            wv_search = df['wavelength'][mask_search].values
            fl_search = df['flux'][mask_search].values
            if len(wv_search) < 5:
                continue

            # Find the actual line center
            found_center = find_line_center_smoothed(wv_search, fl_search, sigma=1.0, absorption=True)
            if found_center is None:
                found_center = restw
            else:
                # Print diagnostic to see how much the center shifted
                print(
                    f"Line {line_id}: Shift from rest {restw} to found {found_center:.2f} = {found_center - restw:.2f}Å")

            # Then open the window around the FOUND center
            lw_min = found_center - halfw
            lw_max = found_center + halfw
            m_fit = (df['wavelength'] >= lw_min) & (df['wavelength'] <= lw_max)
            wv_fit = df['wavelength'][m_fit].values
            fl_fit = df['flux'][m_fit].values
            if len(wv_fit) == 0:
                continue

            nrs = find_noise_regions(df, lw_min, lw_max, noise_window=5, min_noise_separation=5)
            if nrs:
                sig_list = []
                for (nm, nM, _) in nrs:
                    mmn = (df['wavelength'] >= nm) & (df['wavelength'] <= nM)
                    fl_n = df['flux'][mmn].values
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
            fit_noise.setdefault(line_id, {}).setdefault(ep, [])
            fit_noise[line_id][ep] = nrs

            # For bigger plot
            edges_min = [lw_min]
            edges_max = [lw_max]
            for (nm, nM, _) in nrs:
                edges_min.append(nm)
                edges_max.append(nM)
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
            plot_noise.setdefault(line_id, {}).setdefault(ep, [])
            plot_noise[line_id][ep] = nrs

    # Flatten arrays
    all_ep = set()
    for line_id in fit_wv:
        if len(fit_wv[line_id]) > 0:
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

    for line_id in plot_wv:
        if len(plot_wv[line_id]) > 0:
            plot_wv[line_id] = np.concatenate(plot_wv[line_id])
            plot_fl[line_id] = np.concatenate(plot_fl[line_id])
            plot_un[line_id] = np.concatenate(plot_un[line_id])
            plot_ep[line_id] = np.concatenate(plot_ep[line_id])
        else:
            plot_wv[line_id] = np.array([])
            plot_fl[line_id] = np.array([])
            plot_un[line_id] = np.array([])
            plot_ep[line_id] = np.array([])

    all_ep = np.unique(list(all_ep))
    if len(all_ep) == 0:
        print("No data => exit.")
        return

    # Build central wavelength map
    central_map = {}
    for ln_name, ln_info in spectral_lines.items():
        lid = f"line_{int(ln_info['rest_wave'] * 10)}"
        central_map[lid] = ln_info['rest_wave']

    # Load standard-fit JSON if provided
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

    loaded_params = {}
    if args.init_params_json and os.path.exists(args.init_params_json):
        loaded_params = load_standard_fit_json(args.init_params_json)
    else:
        print("No standard-fit JSON path or file found => using default guesses.")

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
            # fallback
            ratio_est = 1.0
            v_sys_est = 0.0
            rv2_init_dict = {ep: (r2 if r2 else 0.0) for ep, (r1, r2) in rv_dict.items()}
            return ratio_est, v_sys_est, rv2_init_dict

        rv1_arr = np.array(rv1_list)
        rv2_arr = np.array(rv2_list)

        slope, intercept = np.polyfit(rv2_arr, rv1_arr, 1)
        b = slope
        a = intercept

        ratio_est = -b
        if abs(ratio_est + 1) < 1e-8:
            ratio_est = 1.0
        v_sys_est = -a / (ratio_est + 1)

        # Bound ratio
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

    ratio_est = 1.0
    v_sys_est = 0.0
    rv2_init_dict = {}
    if loaded_params:
        ratio_est, v_sys_est, rv2_init_dict = derive_ratio_vsys_linfit(loaded_params)

    # Create ratio-based param set
    params = ratio_setup(
        central_wavelengths=central_map,
        all_epochs=all_ep,
        profile_type=profile_type,
        fit_baseline=fit_baseline,
        line_profile=line_profile,
        initial_rvs=rv2_init_dict
    )

    # Overwrite line-shape param guesses
    for p_name in params.keys():
        if p_name in loaded_params:
            val, vmin, vmax = loaded_params[p_name]
            params[p_name].set(value=val)

    # Initialize ratio, v_sys
    if 'ratio' in params:
        params['ratio'].set(value=ratio_est)
    if 'v_sys' in params:
        params['v_sys'].set(value=v_sys_est)

    # Build Minimizer
    minimizer = Minimizer(
        residuals,
        params,
        fcn_args=(fit_wv, fit_fl, fit_ep, fit_un, central_map),
        fcn_kws={
            'profile_type': profile_type,
            'line_profile': line_profile,
            'weighted': use_weighted,
            'fit_baseline': fit_baseline
        }
    )

    # Basin-hopping with multiple seeds
    from tqdm import tqdm

    class BasinHoppingCallback:
        """Simple progress bar for basin-hopping iterations."""

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

    seeds_to_try = [101, 202, 303]
    best_seed = None
    best_result_bh = None
    best_chi2 = None

    for s in seeds_to_try:
        print(f"\n--- Basin hopping with seed={s}, niter=50 ---")
        callback = BasinHoppingCallback(niter=50)
        try:
            temp_result_bh = minimizer.minimize(
                method='basinhopping',
                niter=50,
                T=5.0,
                stepsize=0.3,
                callback=callback,
                minimizer_kwargs={'method': 'L-BFGS-B'},
                seed=s
            )
        finally:
            callback.close()

        # Refine with Levenberg-Marquardt
        temp_result_ls = minimizer.minimize(
            method='leastsq',
            params=temp_result_bh.params,
            max_nfev=20000
        )

        chi2_val = compute_global_chi2(temp_result_ls.params,
                                       minimizer,
                                       fit_wv, fit_fl, fit_ep, fit_un,
                                       central_map,
                                       profile_type, line_profile,
                                       weighted=use_weighted,
                                       fit_baseline=fit_baseline)
        print(f"  => final chi^2 with seed={s}: {chi2_val:.2f}")

        if (best_chi2 is None) or (chi2_val < best_chi2):
            best_chi2 = chi2_val
            best_result_bh = temp_result_ls
            best_seed = s

    print(f"\nBest seed: {best_seed}, best chi^2= {best_chi2:.2f}")
    result_final = best_result_bh

    print("\n===== Final Fit Report =====\n")
    report_fit(result_final)

    # Check uncertainties
    def any_parameter_uncertainty_zero_or_nan(result, param_names=None):
        import numpy as np
        if param_names is None:
            param_names = list(result.params.keys())
        for pname in param_names:
            if pname not in result.params:
                continue
            p = result.params[pname]
            if p.stderr is None or p.stderr == 0.0 or np.isnan(p.stderr):
                return True
        return False

    def update_uncertainties_from_mcmc(result, mcmc_chain, param_names=None):
        import numpy as np
        if param_names is None:
            param_names = mcmc_chain.columns
        for pname in param_names:
            if (pname in result.params) and (pname in mcmc_chain.columns):
                chain_vals = mcmc_chain[pname].values
                std_val = np.std(chain_vals)
                old_stderr = result.params[pname].stderr
                result.params[pname].stderr = std_val
                print(f"[MCMC] Overriding param '{pname}' stderr={old_stderr} => {std_val:.4f}")

    # Collect parameters of interest
    param_list_to_check = []
    if 'ratio' in result_final.params:
        param_list_to_check.append('ratio')
    if 'v_sys' in result_final.params:
        param_list_to_check.append('v_sys')
    for pname in result_final.params.keys():
        if pname.startswith("rv2_epoch"):
            param_list_to_check.append(pname)

    suspicious_uncert = any_parameter_uncertainty_zero_or_nan(result_final, param_list_to_check)
    do_post_mcmc = user_requested_mcmc or suspicious_uncert
    if suspicious_uncert:
        print("Some parameter .stderr is NaN or zero => Running MCMC to refine parameter errors.\n")

    mcmc_chain = None
    if do_post_mcmc:
        print("Performing MCMC sampling (emcee) ...\n")
        best_params = result_final.params.copy()
        result_mcmc = minimizer.minimize(
            method='emcee',
            params=best_params,
            steps=2000,
            nwalkers=150,
            burn=300,
            thin=20,
            is_weighted=use_weighted,
            seed=123
        )
        mcmc_chain = result_mcmc.flatchain
        print(f"MCMC chain shape: {mcmc_chain.shape}")

        update_uncertainties_from_mcmc(result_final, mcmc_chain, param_list_to_check)

    # Final summary + plots
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
        mcmc_chain=mcmc_chain,
        mjd_dict=mjd_dict
    )

    print(f"\nAll done. Results saved in {out_dir}\n")

    if 'ratio' in result_final.params:
        rv_ratio = result_final.params['ratio']
        print(f"Final ratio = {rv_ratio.value:.3f} ± {rv_ratio.stderr or 0.0:.3f}")
    if 'v_sys' in result_final.params:
        rv_vsys = result_final.params['v_sys']
        print(f"Final v_sys = {rv_vsys.value:.2f} ± {rv_vsys.stderr or 0.0:.2f} km/s")


if __name__ == "__main__":
    main()
