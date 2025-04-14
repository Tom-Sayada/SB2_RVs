#!/usr/bin/env python3
# run_standard_fit.py

import os
import argparse
import json
import numpy as np
from tqdm import tqdm
from lmfit import Minimizer, report_fit

# Local modules
from src.utils import (
    find_observation_files,
    load_data_for_epoch,
    find_noise_regions,
    find_line_center_smoothed
)
from src.model_builder import (
    setup_parameters,  # standard approach
    residuals  # standard approach
)
from src.plot_results import report_fit_results


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


def compute_global_chi2(params, minimizer,
                        fit_wv, fit_fl, fit_ep, fit_un,
                        central_map,
                        profile_type, line_profile,
                        weighted=True):
    """
    Compute total chi^2 = sum of (residuals^2) for all lines/epochs,
    given the current params. Re-uses the 'residuals' function from model_builder.
    """
    # We call the same residual function that Minimizer uses
    res = minimizer.userfcn(
        params,  # current parameters
        fit_wv, fit_fl, fit_ep, fit_un, central_map,
        profile_type=profile_type,
        line_profile=line_profile,
        weighted=weighted
    )
    chi2 = np.sum(res**2)
    return chi2


def save_bestfit_params_to_json(params, filepath):
    """
    Save the best-fit parameters to a JSON file.
    Format: {
      "param_name": {"value": float, "min": float, "max": float},
      ...
    }
    """
    data = {}
    for key, par in params.items():
        data[key] = {
            "value": par.value,
            "min": par.min,
            "max": par.max
        }
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved best-fit parameters to '{filepath}'")


def any_rv_stderr_zero(result):
    """Check if any RV param has .stderr == 0 or None."""
    for pname, par in result.params.items():
        if pname.startswith("rv1_epoch") or pname.startswith("rv2_epoch"):
            if par.stderr is not None and par.stderr == 0.0:
                return True
            if par.stderr is None:
                return True
    return False


def update_zero_rv_stderr_from_mcmc(result, mcmc_chain):
    """Update zero .stderr values with MCMC standard deviations."""
    for pname, par in result.params.items():
        if pname.startswith("rv1_epoch") or pname.startswith("rv2_epoch"):
            if (par.stderr is None) or (par.stderr == 0.0):
                if pname in mcmc_chain.columns:
                    param_samples = mcmc_chain[pname]
                    std_val = param_samples.std()
                    par.stderr = std_val
                    print(f"[MCMC override] Param '{pname}' had 0 stderr => using MCMC std={std_val:.4f}")
                else:
                    print(f"[Warning] MCMC chain has no column for '{pname}', cannot override.")


def main():
    parser = argparse.ArgumentParser(
        description="SB2 multi-line fit with dynamic line-windowing (standard)."
    )
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="output_fit_results")

    parser.add_argument("--profile_type", type=str, default='sym')
    parser.add_argument("--line_profile", type=str, default='voigt',
                        choices=['voigt', 'gaussian'],
                        help="Profile type for stellar components")

    parser.add_argument("--lines", type=str, default=None,
                        help="Comma-separated e.g. 'He4471,He4026,H4340'")
    parser.add_argument("--unweighted", action='store_true')
    parser.add_argument("--fit_baseline", action='store_true')
    parser.add_argument("--save_params_json", type=str, default="standard_fit_params.json",
                        help="Where to save the final best-fit parameters (JSON)")
    parser.add_argument("--mcmc", action='store_true',
                        help="If set, always do a post-fit MCMC. Otherwise do MCMC only if .stderr=0 for any RV param.")
    args = parser.parse_args()

    data_dir = args.data_dir
    out_dir = args.output_dir
    profile_type = args.profile_type.strip().lower()
    line_profile = args.line_profile.strip().lower()
    use_weighted = (not args.unweighted)
    fit_baseline = bool(args.fit_baseline)
    save_path = args.save_params_json
    user_requested_mcmc = args.mcmc

    os.makedirs(out_dir, exist_ok=True)

    # Example lines
    all_lines_info = {
        'He4471': {'rest_wave': 4471.5, 'window': 11.0},
        'He4026': {'rest_wave': 4026.0, 'window': 25.0},
        'He4388': {'rest_wave': 4388.0, 'window': 13.0},
        'H4340':  {'rest_wave': 4340.472, 'window': 30.0},
        'H4101':  {'rest_wave': 4101,     'window': 22.0},
        'He4144': {'rest_wave': 4144, 'window': 10.0},
        'He4120': {'rest_wave': 4120, 'window': 13.0}
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

    print(f"\nData from: {data_dir}")
    print(f"Output:   {out_dir}")
    print(f"Fitting lines: {list(spectral_lines.keys())}")
    print(f"Profile type: {profile_type}, Line profile: {line_profile}")
    print(f"Weighted={use_weighted}, Baseline={fit_baseline}")
    print(f"MCMC forcibly requested? {user_requested_mcmc}")

    # Find observation files
    epoch_files = find_observation_files(data_dir)
    if not epoch_files:
        raise ValueError(f"No observation files found in {data_dir}.")

    from collections import OrderedDict
    # Data structures for fitting
    fit_wv = OrderedDict()
    fit_fl = OrderedDict()
    fit_un = OrderedDict()
    fit_ep = OrderedDict()
    fit_noise = {}

    # Data structures for plotting
    plot_wv = OrderedDict()
    plot_fl = OrderedDict()
    plot_un = OrderedDict()
    plot_ep = OrderedDict()
    plot_noise = {}

    # Dictionary to store MJD information for each epoch
    mjd_dict = {}

    # Initialize
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

    print(f"Found {len(epoch_files)} files. Building arrays now...")
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

        # For each line
        for ln_name, ln_info in spectral_lines.items():
            line_id = f"line_{int(ln_info['rest_wave'] * 10)}"
            restw = ln_info['rest_wave']
            halfw = ln_info['window'] / 2.0

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

                # --- NEW PLOTTING CODE ---
                import matplotlib.pyplot as plt
                from scipy.ndimage import gaussian_filter1d

                # Smooth the flux using the same sigma as the finder
                smoothed_flux = gaussian_filter1d(fl_search, sigma=1.0)

                plt.figure(figsize=(8, 4))
                plt.plot(wv_search, fl_search, 'b-', label='Original Flux')
                plt.plot(wv_search, smoothed_flux, 'r-', label='Smoothed Flux')
                plt.axvline(found_center, color='g', linestyle='--', label=f'Found Center: {found_center:.2f} Å')
                plt.xlabel("Wavelength (Å)")
                plt.ylabel("Flux")
                plt.title(f"Line {line_id}: Found Center")
                plt.legend()
                plt.tight_layout()
                #plt.show()
                # ---------------------------

            # Then open the window around the FOUND center
            lw_min = found_center - halfw
            lw_max = found_center + halfw
            mask_fit = (df['wavelength'] >= lw_min) & (df['wavelength'] <= lw_max)
            wv_fit = df['wavelength'][mask_fit].values
            fl_fit = df['flux'][mask_fit].values
            if len(wv_fit) == 0:
                continue

            # Estimate noise
            nrs = find_noise_regions(
                df, lw_min, lw_max,
                noise_window=5,
                min_noise_separation=5,
                max_noise_offset=80
            )
            if len(nrs) > 0:
                sig_list = []
                for (nm, nM, dr) in nrs:
                    mm = (df['wavelength'] >= nm) & (df['wavelength'] <= nM)
                    fl_n = df['flux'][mm].values
                    if len(fl_n) > 0:
                        sig_list.append(fl_n.std())
                if sig_list:
                    epoch_sigma = np.mean(sig_list)
                else:
                    epoch_sigma = max(0.02, fl_fit.std())
            else:
                epoch_sigma = max(0.02, fl_fit.std())

            # Store final arrays
            fit_wv[line_id].append(wv_fit)
            fit_fl[line_id].append(fl_fit)
            fit_un[line_id].append(np.full_like(fl_fit, epoch_sigma))
            fit_ep[line_id].append(np.full_like(fl_fit, ep, dtype=int))
            fit_noise[line_id].setdefault(ep, [])
            fit_noise[line_id][ep] = nrs

            # For plotting
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

            mask_plot = (df['wavelength'] >= pm) & (df['wavelength'] <= pM)
            wv_plot = df['wavelength'][mask_plot].values
            fl_plot = df['flux'][mask_plot].values
            if len(wv_plot) == 0:
                continue

            plot_wv[line_id].append(wv_plot)
            plot_fl[line_id].append(fl_plot)
            plot_un[line_id].append(np.full_like(fl_plot, epoch_sigma))
            plot_ep[line_id].append(np.full_like(fl_plot, ep, dtype=int))
            plot_noise[line_id].setdefault(ep, [])
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

    # Map line ID -> rest wavelength
    central_map = {}
    for ln_name, ln_info in spectral_lines.items():
        lid = f"line_{int(ln_info['rest_wave'] * 10)}"
        central_map[lid] = ln_info['rest_wave']

    # Setup standard approach parameters
    params = setup_parameters(
        central_wavelengths=central_map,
        all_epochs=all_ep,
        profile_type=profile_type,
        fit_baseline=fit_baseline,
        line_profile=line_profile
    )

    # Minimizer
    minimizer = Minimizer(
        residuals,
        params,
        fcn_args=(fit_wv, fit_fl, fit_ep, fit_un, central_map),
        fcn_kws={
            'profile_type': profile_type,
            'line_profile': line_profile,
            'weighted': use_weighted
        }
    )

    # ======================================
    #  Basin hopping with multiple seeds
    # ======================================
    seeds_to_try = [101]  # customize as you want
    best_seed = None
    best_result_bh = None
    best_chi2 = None

    for s in seeds_to_try:
        print(f"\n--- Basin hopping with seed={s}, niter=10 ---")
        callback = BasinHoppingCallback(niter=10)
        try:
            temp_result_bh = minimizer.minimize(
                method='basinhopping',
                niter=10,
                T=5.0,
                stepsize=0.3,
                callback=callback,
                minimizer_kwargs={'method': 'L-BFGS-B'},
                seed=s
            )
        finally:
            callback.close()

        # Refine with leastsq from that BH solution
        temp_result_ls = minimizer.minimize(
            method='leastsq',
            params=temp_result_bh.params,
            max_nfev=20000
        )

        # Evaluate total chi^2
        chi2_val = compute_global_chi2(temp_result_ls.params, minimizer,
                                       fit_wv, fit_fl, fit_ep, fit_un,
                                       central_map,
                                       profile_type, line_profile,
                                       weighted=use_weighted)
        print(f"  => final chi^2 with seed={s}: {chi2_val:.2f}")

        if (best_chi2 is None) or (chi2_val < best_chi2):
            best_chi2 = chi2_val
            best_result_bh = temp_result_ls
            best_seed = s

    print(f"\nBest seed: {best_seed}, best chi^2= {best_chi2:.2f}")
    result_final = best_result_bh

    print("\n===== Fit Report =====\n")
    report_fit(result_final)

    # Check if we have suspicious zero uncertainties for RVs
    suspicious_rv_uncerts = any_rv_stderr_zero(result_final)

    do_post_mcmc = user_requested_mcmc or suspicious_rv_uncerts
    if suspicious_rv_uncerts:
        print("Warning: zero .stderr found for some RV param => post-fit MCMC to get better uncertainties.\n")

    mcmc_chain = None
    if do_post_mcmc:
        print("Performing MCMC sampling (emcee) post-fit...\n")
        # We'll do MCMC starting from the final best-fit
        best_params = result_final.params.copy()
        result_mcmc = minimizer.minimize(
            method='emcee',
            params=best_params,
            steps=2000,  # total MCMC steps
            nwalkers=200,  # number of MCMC walkers
            burn=500,  # discard first 300 steps
            thin=5,  # keep only every 15th step
            is_weighted=use_weighted,
            seed=123
        )
        mcmc_chain = result_mcmc.flatchain
        print(f"MCMC chain shape: {mcmc_chain.shape}")

        # Now override zero-stderr RV params with MCMC std
        update_zero_rv_stderr_from_mcmc(result_final, mcmc_chain)

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
        mcmc_chain=mcmc_chain,
        mjd_dict=mjd_dict
    )

    print(f"\nAll done. Results in {out_dir}\n")

    # Save best-fit parameters to JSON
    save_bestfit_params_to_json(result_final.params, os.path.join(out_dir, save_path))


if __name__ == "__main__":
    main()
