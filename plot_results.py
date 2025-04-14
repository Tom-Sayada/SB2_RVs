#!/usr/bin/env python3
# plot_results.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.optimize import curve_fit

from src.model_builder import compute_full_model, get_star_components
from src.model_builder import _compute_rv1_rv2  # keep the RV function
from src.line_profiles import (
    voigt_profile, skewed_voigt_profile,
    gaussian_profile, skewed_gaussian_profile
)


def compute_single_line_epoch(params, wv_arr, line_id, ep_int,
                              central_map,
                              profile_type='sym',
                              line_profile='voigt'):
    """
    Partial re-implementation for a single line+epoch model flux,
    used to build MCMC-based envelopes or quick checks.
    """
    c_speed = 299792.458

    # We do NOT need min_width if line_profiles already do the clamp.

    rv1, rv2 = _compute_rv1_rv2(params, ep_int)

    # retrieve line param values...
    # baseline, a1, s1, etc.
    bkey = f'baseline_{line_id}'
    base_val = params[bkey].value if bkey in params else 1.0
    a1 = params[f'a1_{line_id}'].value
    s1 = params[f'sigma1_{line_id}'].value
    a2 = params[f'a2_{line_id}'].value
    s2 = params[f'sigma2_{line_id}'].value

    # if voigt:
    if line_profile == 'voigt':
        g1 = params[f'gamma1_{line_id}'].value
        g2 = params[f'gamma2_{line_id}'].value
    else:
        g1 = g2 = None

    skew1p = params.get(f'skew1_{line_id}', None)
    skew1 = skew1p.value if skew1p else 0.0
    skew2p = params.get(f'skew2_{line_id}', None)
    skew2 = skew2p.value if skew2p else 0.0

    center1 = central_map[line_id] * (1 + rv1 / c_speed)
    center2 = central_map[line_id] * (1 + rv2 / c_speed)

    # Build star1, star2 line profiles:
    if line_profile == 'gaussian':
        if abs(skew1) > 1e-8:
            prof1 = skewed_gaussian_profile(wv_arr, -abs(a1), center1, s1, skew1)
        else:
            prof1 = gaussian_profile(wv_arr, -abs(a1), center1, s1)

        if abs(skew2) > 1e-8:
            prof2 = skewed_gaussian_profile(wv_arr, -abs(a2), center2, s2, skew2)
        else:
            prof2 = gaussian_profile(wv_arr, -abs(a2), center2, s2)
    else:
        if abs(skew1) > 1e-8:
            prof1 = skewed_voigt_profile(wv_arr, -abs(a1), center1, s1, g1, skew1)
        else:
            prof1 = voigt_profile(wv_arr, -abs(a1), center1, s1, g1)

        if abs(skew2) > 1e-8:
            prof2 = skewed_voigt_profile(wv_arr, -abs(a2), center2, s2, g2, skew2)
        else:
            prof2 = voigt_profile(wv_arr, -abs(a2), center2, s2, g2)

    return base_val + prof1 + prof2


def compute_mcmc_envelopes(
        result,
        mcmc_chain,
        wv_dict,
        ep_dict,
        central_map,
        profile_type='sym',
        line_profile='voigt',
        n_draws=300
):
    """
    Example function that:
      - Randomly samples from mcmc_chain
      - For each line+epoch, computes the model flux for each sample
      - Returns median ±1σ flux arrays for plotting.
    """
    from lmfit import Parameters
    import random

    # pick n_draws random rows from chain
    if n_draws > len(mcmc_chain):
        n_draws = len(mcmc_chain)
    samples = mcmc_chain.sample(n_draws, replace=False, random_state=123)

    # store => bands[line_id][epoch] => { 'wv', 'flux_median', 'flux_std' }
    bands = {}
    for lid in wv_dict:
        if len(wv_dict[lid]) == 0:
            continue
        bands[lid] = {}

    base_params = result.params.copy()

    all_ep = set()
    for lid in ep_dict:
        all_ep.update(ep_dict[lid].tolist())
    all_ep = np.unique(list(all_ep))

    for ep in all_ep:
        ep_int = int(ep)
        for lid in wv_dict:
            wv_arr = wv_dict[lid][ep_dict[lid] == ep]
            if len(wv_arr) == 0:
                continue

            flux_matrix = []
            for idx, row in samples.iterrows():
                # copy base_params => override .value
                trial_params = base_params.copy()
                for pname in row.index:
                    if pname in trial_params:
                        trial_params[pname].value = row[pname]

                # compute single line+epoch flux
                flux_sub = compute_single_line_epoch(
                    trial_params, wv_arr, lid, ep_int,
                    central_map, profile_type, line_profile
                )
                flux_matrix.append(flux_sub)
            flux_matrix = np.array(flux_matrix)  # shape => (n_draws, n_pts)

            flux_median = np.median(flux_matrix, axis=0)
            flux_std = np.std(flux_matrix, axis=0)

            if ep not in bands[lid]:
                bands[lid][ep] = {}
            bands[lid][ep]['wv'] = wv_arr
            bands[lid][ep]['flux_median'] = flux_median
            bands[lid][ep]['flux_std'] = flux_std

    return bands


def apply_star_label_swap_if_needed(result, central_wavelengths):
    """
    Evaluate final velocities => see if star2 amplitude < star1 amplitude => swap.
    This modifies result.params in-place.
    """
    has_ratio = ('ratio' in result.params) and ('v_sys' in result.params)
    ep_list = set()
    for pn in result.params.keys():
        if pn.startswith("rv2_epoch"):
            ep_list.add(int(pn.replace("rv2_epoch", "")))
        if pn.startswith("rv1_epoch"):
            ep_list.add(int(pn.replace("rv1_epoch", "")))
    ep_nums = sorted(ep_list)
    if not ep_nums:
        return

    ratio_val = result.params['ratio'].value if has_ratio else 1.0
    vsys_val = result.params['v_sys'].value if has_ratio else 0.0

    rv1_all = []
    rv2_all = []
    for e in ep_nums:
        rv1p = f"rv1_epoch{e}"
        rv2p = f"rv2_epoch{e}"

        if has_ratio:
            rv2_v = result.params[rv2p].value if rv2p in result.params else 0.0
            rv1_v = - ratio_val * (rv2_v + vsys_val) - vsys_val
        else:
            rv1_v = result.params[rv1p].value if rv1p in result.params else 0.0
            rv2_v = result.params[rv2p].value if rv2p in result.params else 0.0

        rv1_all.append(rv1_v)
        rv2_all.append(rv2_v)

    rv1_all = np.array(rv1_all)
    rv2_all = np.array(rv2_all)

    amp1 = rv1_all.max() - rv1_all.min()
    amp2 = rv2_all.max() - rv2_all.min()

    if amp2 < amp1:
        print(f"[apply_star_label_swap_if_needed] Swapping star1<->star2 by amplitude logic.\n"
              f"   amp1={amp1:.2f}, amp2={amp2:.2f}")

        # swap line-specific
        for lid in central_wavelengths:
            for prefix in ['a', 'sigma', 'gamma', 'skew']:
                p1 = f"{prefix}1_{lid}"
                p2 = f"{prefix}2_{lid}"
                if p1 in result.params and p2 in result.params:
                    val1 = result.params[p1].value
                    val2 = result.params[p2].value
                    result.params[p1].set(value=val2)
                    result.params[p2].set(value=val1)
                    err1 = result.params[p1].stderr
                    err2 = result.params[p2].stderr
                    result.params[p1].stderr = err2
                    result.params[p2].stderr = err1

        # swap rv1_epochN & rv2_epochN
        for e in ep_nums:
            rv1p = f"rv1_epoch{e}"
            rv2p = f"rv2_epoch{e}"
            if rv1p in result.params and rv2p in result.params:
                val1 = result.params[rv1p].value
                val2 = result.params[rv2p].value
                result.params[rv1p].set(value=val2)
                result.params[rv2p].set(value=val1)
                err1 = result.params[rv1p].stderr
                err2 = result.params[rv2p].stderr
                result.params[rv1p].stderr = err2
                result.params[rv2p].stderr = err1


def build_per_epoch_rvs(result):
    """
    Build a DataFrame of final RVs for each epoch, incl. errors.
    If ratio approach => we compute rv1 from ratio => we do partial derivative ignoring cov.
    """
    all_keys = list(result.params.keys())
    ep_list = set()
    for pk in all_keys:
        if pk.startswith("rv1_epoch"):
            ep_list.add(int(pk.replace("rv1_epoch", "")))
        elif pk.startswith("rv2_epoch"):
            ep_list.add(int(pk.replace("rv2_epoch", "")))

    ep_nums = sorted(ep_list)
    has_ratio = ('ratio' in result.params) and ('v_sys' in result.params)

    rows = []
    if has_ratio:
        ratio_val = result.params['ratio'].value
        ratio_err = result.params['ratio'].stderr or 0.0
        vsys_val = result.params['v_sys'].value
        vsys_err = result.params['v_sys'].stderr or 0.0
    else:
        ratio_val = 1.0
        ratio_err = 0.0
        vsys_val = 0.0
        vsys_err = 0.0

    for ep in ep_nums:
        rv1p = f"rv1_epoch{ep}"
        rv2p = f"rv2_epoch{ep}"
        if has_ratio and (rv2p in result.params):
            rv2_val = result.params[rv2p].value
            rv2_err = result.params[rv2p].stderr or 0.0

            rv1_val = - ratio_val * (rv2_val + vsys_val) - vsys_val
            # partial derivative ignoring correlation
            # d(rv1)/d(ratio) = -(rv2 + vsys), d(rv1)/d(rv2)= -ratio, d(rv1)/d(vsys)= -(ratio+1)
            drv1_dratio = -(rv2_val + vsys_val)
            drv1_drv2 = -ratio_val
            drv1_dvsys = -(ratio_val + 1)

            var_ratio = ratio_err ** 2
            var_rv2 = rv2_err ** 2
            var_vsys = vsys_err ** 2

            rv1_err_sq = (drv1_dratio ** 2) * var_ratio + (drv1_drv2 ** 2) * var_rv2 + (drv1_dvsys ** 2) * var_vsys
            rv1_err = np.sqrt(rv1_err_sq)

        else:
            rv1_val = result.params[rv1p].value if rv1p in result.params else 0.0
            rv1_err = (result.params[rv1p].stderr if (rv1p in result.params and
                                                      result.params[rv1p].stderr is not None) else 0.0)
            rv2_val = result.params[rv2p].value if rv2p in result.params else 0.0
            rv2_err = (result.params[rv2p].stderr if (rv2p in result.params and
                                                      result.params[rv2p].stderr is not None) else 0.0)

        rows.append({
            "Epoch": ep,
            "RV1": rv1_val,
            "RV1_err": rv1_err,
            "RV2": rv2_val,
            "RV2_err": rv2_err
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df


def plot_rv2_vs_rv1(df_rvs, outdir):
    """Plot RV2 vs RV1 with linear fit"""
    if df_rvs.empty:
        return

    rv1 = df_rvs["RV1"].values
    rv2 = df_rvs["RV2"].values
    e1 = df_rvs["RV1_err"].values
    e2 = df_rvs["RV2_err"].values
    e1[e1 <= 0] = 1e-5
    e2[e2 <= 0] = 1e-5

    def lin_func(x, a, b):
        return a * x + b

    try:
        popt, pcov = curve_fit(lin_func, rv1, rv2, sigma=e2, absolute_sigma=True)
        slope, intercept = popt
        slope_err, intercept_err = np.sqrt(np.diag(pcov))
    except Exception as e:
        print(f"Warning: Could not fit RV2 vs RV1: {e}")
        slope, intercept, slope_err, intercept_err = (np.nan,) * 4

    rv2_fit = lin_func(rv1, slope, intercept) if not np.isnan(slope) else np.zeros_like(rv1)
    res = rv2 - rv2_fit

    fig, (axm, axr) = plt.subplots(2, 1, figsize=(7, 10), sharex=True)
    axm.errorbar(rv1, rv2, xerr=e1, yerr=e2, fmt='o', color='blue', alpha=0.7, label='Calc RVs')
    if not np.isnan(slope):
        xseq = np.linspace(rv1.min(), rv1.max(), 100)
        yseq = lin_func(xseq, slope, intercept)
        lbl = f"slope={slope:.2f}±{slope_err:.2f}, intercept={intercept:.2f}±{intercept_err:.2f}"
        axm.plot(xseq, yseq, 'r--', label=lbl)

    axm.set_ylabel("RV2 (km/s)")
    axm.set_title("RV2 vs. RV1")
    axm.legend()
    axm.grid(alpha=0.3)

    axr.errorbar(rv1, res, xerr=e1, yerr=e2,
                 fmt='o', color='blue', alpha=0.7)
    axr.axhline(0, color='r', ls='--', alpha=0.5)
    axr.set_xlabel("RV1 (km/s)")
    axr.set_ylabel("Residual (km/s)")
    axr.grid(alpha=0.3)

    plt.tight_layout()
    outpng = os.path.join(outdir, "rv2_vs_rv1.png")
    plt.savefig(outpng, dpi=300)
    plt.close()


def plot_rv_vs_time(df_rvs, outdir):
    """
    New function to plot RV vs Time (MJD)
    Creates a figure with RV1 and RV2 vs MJD
    """
    if df_rvs.empty or "MJD" not in df_rvs.columns:
        return

    # Get data
    mjd = df_rvs["MJD"].values
    epoch = df_rvs["Epoch"].values
    rv1 = df_rvs["RV1"].values
    rv2 = df_rvs["RV2"].values
    e1 = df_rvs["RV1_err"].values
    e2 = df_rvs["RV2_err"].values

    # Filter out rows with invalid MJD values
    valid_mask = ~np.isnan(mjd)
    if not np.any(valid_mask):
        print("No valid MJD values for RV vs time plot")
        return

    mjd = mjd[valid_mask]
    epoch = epoch[valid_mask]
    rv1 = rv1[valid_mask]
    rv2 = rv2[valid_mask]
    e1 = e1[valid_mask]
    e2 = e2[valid_mask]

    # Set any zero errors to a small value to avoid divide by zero
    e1[e1 <= 0] = 1e-5
    e2[e2 <= 0] = 1e-5

    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Plot RV1 vs MJD
    ax1.errorbar(mjd, rv1, yerr=e1, fmt='o', color='blue', capsize=3,
                 markersize=6, label='Star 1')
    ax1.set_ylabel('RV1 (km/s)')
    ax1.set_title('Radial Velocity vs. Time')
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Plot RV2 vs MJD
    ax2.errorbar(mjd, rv2, yerr=e2, fmt='o', color='green', capsize=3,
                 markersize=6, label='Star 2')
    ax2.set_xlabel('MJD')
    ax2.set_ylabel('RV2 (km/s)')
    ax2.grid(alpha=0.3)
    ax2.legend()

    # Add epoch numbers as annotations
    for i, (m, r1, r2, ep) in enumerate(zip(mjd, rv1, rv2, epoch)):
        ax1.annotate(f"{int(ep)}", (m, r1), xytext=(5, 5),
                     textcoords='offset points', fontsize=8)
        ax2.annotate(f"{int(ep)}", (m, r2), xytext=(5, 5),
                     textcoords='offset points', fontsize=8)

    # Adjust layout
    plt.tight_layout()

    # Save
    outpng = os.path.join(outdir, "rv_vs_time.png")
    plt.savefig(outpng, dpi=300)
    plt.close()

    # Create a combined plot with both stars on same axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot both RVs on the same axis
    ax.errorbar(mjd, rv1, yerr=e1, fmt='o', color='blue', capsize=3,
                markersize=6, label='Star 1')
    ax.errorbar(mjd, rv2, yerr=e2, fmt='o', color='green', capsize=3,
                markersize=6, label='Star 2')

    # Add annotations for epochs
    for i, (m, r1, r2, ep) in enumerate(zip(mjd, rv1, rv2, epoch)):
        ax.annotate(f"{int(ep)}", (m, r1), xytext=(5, 5),
                    textcoords='offset points', fontsize=8, color='blue')
        ax.annotate(f"{int(ep)}", (m, r2), xytext=(5, 5),
                    textcoords='offset points', fontsize=8, color='green')

    ax.set_xlabel('MJD')
    ax.set_ylabel('Radial Velocity (km/s)')
    ax.set_title('Radial Velocities vs. Time')
    ax.grid(alpha=0.3)
    ax.legend()

    plt.tight_layout()
    outpng = os.path.join(outdir, "rv_vs_time_combined.png")
    plt.savefig(outpng, dpi=300)
    plt.close()


def plot_real_calc_comparison(df_calc, df_real, outdir):
    """
    If star_rvs_per_epoch.csv => compare 'RV1_calc' vs 'RV1_real', etc.
    Also plot comparison vs MJD if available
    """
    merged = pd.merge(df_calc, df_real, on="Epoch", suffixes=('', '_real'))
    if merged.empty:
        return

    ep = merged["Epoch"].values
    rv1c = merged["RV1"].values
    rv1r = merged["RV1_real"].values
    e1 = merged.get("RV1_err", pd.Series([0] * len(merged))).values
    rv2c = merged["RV2"].values
    rv2r = merged["RV2_real"].values
    e2 = merged.get("RV2_err", pd.Series([0] * len(merged))).values

    d1 = rv1c - rv1r
    d2 = rv2c - rv2r

    # First plot: RV vs Epoch
    fig = plt.figure(figsize=(10, 10))
    gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 3, 1])
    ax1 = fig.add_subplot(gs[0])
    ax1r = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    ax2r = fig.add_subplot(gs[3])

    # RV1
    ax1.errorbar(ep, rv1c, yerr=e1, fmt='ro', ms=4, capsize=3, label='Calc RV1')
    ax1.plot(ep, rv1r, 'bo', ms=4, label='Real RV1')
    for i, epp in enumerate(ep):
        ax1.plot([epp, epp], [rv1r[i], rv1c[i]], 'k--', alpha=0.5)
    ax1.set_ylabel("RV1 (km/s)")
    ax1.set_title("Real vs Calc: RV1")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax1r.errorbar(ep, d1, yerr=e1, fmt='ko', ms=4, capsize=3)
    ax1r.axhline(0, color='r', ls='--', alpha=0.5)
    ax1r.set_ylabel("Diff (km/s)")
    ax1r.grid(alpha=0.3)

    # RV2
    ax2.errorbar(ep, rv2c, yerr=e2, fmt='ro', ms=4, capsize=3, label='Calc RV2')
    ax2.plot(ep, rv2r, 'bo', ms=4, label='Real RV2')
    for i, epp in enumerate(ep):
        ax2.plot([epp, epp], [rv2r[i], rv2c[i]], 'k--', alpha=0.5)
    ax2.set_ylabel("RV2 (km/s)")
    ax2.set_title("Real vs Calc: RV2")
    ax2.legend()
    ax2.grid(alpha=0.3)

    ax2r.errorbar(ep, d2, yerr=e2, fmt='ko', ms=4, capsize=3)
    ax2r.axhline(0, color='r', ls='--', alpha=0.5)
    ax2r.set_xlabel("Epoch")
    ax2r.set_ylabel("Diff (km/s)")
    ax2r.grid(alpha=0.3)

    plt.tight_layout()
    outpng = os.path.join(outdir, "real_calc_comparison.png")
    plt.savefig(outpng, dpi=300)
    plt.close()

    # Second plot: RV vs MJD if MJD is available
    if "MJD" in merged.columns and not merged["MJD"].isna().all():
        mjd = merged["MJD"].values

        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(4, 1, height_ratios=[3, 1, 3, 1])
        ax1 = fig.add_subplot(gs[0])
        ax1r = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2])
        ax2r = fig.add_subplot(gs[3])

        # RV1
        ax1.errorbar(mjd, rv1c, yerr=e1, fmt='ro', ms=4, capsize=3, label='Calc RV1')
        ax1.plot(mjd, rv1r, 'bo', ms=4, label='Real RV1')
        for i, m in enumerate(mjd):
            ax1.plot([m, m], [rv1r[i], rv1c[i]], 'k--', alpha=0.5)
            ax1.annotate(f"{int(ep[i])}", (m, rv1c[i]), xytext=(3, 3),
                         textcoords='offset points', fontsize=8, color='red')
            ax1.annotate(f"{int(ep[i])}", (m, rv1r[i]), xytext=(3, 3),
                         textcoords='offset points', fontsize=8, color='blue')
        ax1.set_ylabel("RV1 (km/s)")
        ax1.set_title("Real vs Calc: RV1 vs Time")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax1r.errorbar(mjd, d1, yerr=e1, fmt='ko', ms=4, capsize=3)
        ax1r.axhline(0, color='r', ls='--', alpha=0.5)
        ax1r.set_ylabel("Diff (km/s)")
        ax1r.grid(alpha=0.3)

        # RV2
        ax2.errorbar(mjd, rv2c, yerr=e2, fmt='ro', ms=4, capsize=3, label='Calc RV2')
        ax2.plot(mjd, rv2r, 'bo', ms=4, label='Real RV2')
        for i, m in enumerate(mjd):
            ax2.plot([m, m], [rv2r[i], rv2c[i]], 'k--', alpha=0.5)
            ax2.annotate(f"{int(ep[i])}", (m, rv2c[i]), xytext=(3, 3),
                         textcoords='offset points', fontsize=8, color='red')
            ax2.annotate(f"{int(ep[i])}", (m, rv2r[i]), xytext=(3, 3),
                         textcoords='offset points', fontsize=8, color='blue')
        ax2.set_ylabel("RV2 (km/s)")
        ax2.set_title("Real vs Calc: RV2 vs Time")
        ax2.legend()
        ax2.grid(alpha=0.3)

        ax2r.errorbar(mjd, d2, yerr=e2, fmt='ko', ms=4, capsize=3)
        ax2r.axhline(0, color='r', ls='--', alpha=0.5)
        ax2r.set_xlabel("MJD")
        ax2r.set_ylabel("Diff (km/s)")
        ax2r.grid(alpha=0.3)

        plt.tight_layout()
        outpng = os.path.join(outdir, "real_calc_comparison_vs_mjd.png")
        plt.savefig(outpng, dpi=300)
        plt.close()


def plot_epoch_lines(
        result,
        wv_line_dict,
        fl_line_dict,
        un_line_dict,
        ep_line_dict,
        central_wavelengths,
        noise_regions,
        windows,
        df_results,
        df_chi2,
        output_directory,
        profile_type='sym',
        line_profile='voigt',
        mcmc_bands=None
):
    """
    Plots each line+epoch with:
      - Data + Combined Model + Star1,Star2
      - Residual panel
      - If mcmc_bands => ±1σ fill
    """
    df_global = df_chi2[df_chi2["Line_ID"] == "GLOBAL"]
    if len(df_global) == 1:
        global_chi = df_global["Chi_Square_reduced"].values[0]
        global_p = df_global["p_value"].values[0]
    else:
        global_chi = np.nan
        global_p = np.nan

    ratio_val = None
    ratio_err = None
    if 'ratio' in result.params:
        ratio_val = result.params['ratio'].value
        ratio_err = result.params['ratio'].stderr or np.nan

    line_map = {}
    for _, row in df_chi2.iterrows():
        if row["Line_ID"] == "GLOBAL":
            continue
        line_map[row["Line_ID"]] = (row["Chi_Square_reduced"], row["p_value"])

    model_flux = compute_full_model(
        result.params,
        wv_line_dict,
        ep_line_dict,
        central_wavelengths,
        profile_type=profile_type,
        line_profile=line_profile
    )

    all_ep = []
    for lid in wv_line_dict:
        if len(ep_line_dict[lid]) > 0:
            all_ep.extend(ep_line_dict[lid])
    all_ep = np.unique(all_ep)

    # build ep->(RV1±err, RV2±err)
    ep2rv = {}
    for _, row in df_results.iterrows():
        e = row["Epoch"]
        ep2rv[e] = (row["RV1"], row["RV1_err"], row["RV2"], row["RV2_err"])

    outplots = os.path.join(output_directory, "epoch_plots")
    os.makedirs(outplots, exist_ok=True)

    for ep in all_ep:
        lines_for_ep = []
        for lid in wv_line_dict:
            if np.any(ep_line_dict[lid] == ep):
                lines_for_ep.append(lid)
        if not lines_for_ep:
            continue

        rv1v, rv1e, rv2v, rv2e = ep2rv.get(ep, (np.nan,) * 4)
        suptt = (f"Epoch {ep} | RV1={rv1v:.2f}±{rv1e:.2f}, RV2={rv2v:.2f}±{rv2e:.2f}\n"
                 f"Global χ²ᵣ= {global_chi:.3f}, p= {global_p:.3g}")
        if ratio_val is not None:
            suptt += f"\nratio= {ratio_val:.2f}±{ratio_err:.2f}"

        fig, axes = plt.subplots(len(lines_for_ep), 2, figsize=(14, 4 * len(lines_for_ep)), sharex=False)
        if len(lines_for_ep) == 1:
            axes = [axes]

        fig.suptitle(suptt, fontsize=13)

        for i, lid in enumerate(lines_for_ep):
            axm, axr = axes[i]
            mk = (ep_line_dict[lid] == ep)
            wv_d = wv_line_dict[lid][mk]
            fl_d = fl_line_dict[lid][mk]
            un_d = un_line_dict[lid][mk]
            md_d = model_flux[lid][mk]

            resid = (fl_d - md_d) / un_d

            line_chi, line_p = line_map.get(lid, (np.nan, np.nan))
            restwv = central_wavelengths[lid]
            line_title = f"Line {lid} (~{restwv:.1f}) => χ²ᵣ= {line_chi:.3f}, p= {line_p:.3g}"
            axm.set_title(line_title, fontsize=10)

            s1f, s2f = get_star_components(
                result.params, lid, wv_d, restwv,
                rv1v, rv2v,
                profile_type=profile_type,
                line_profile=line_profile
            )

            halfw = windows.get(lid, 20.0) / 2
            cwv = central_wavelengths[lid]
            c_speed = 299792.458
            center1 = cwv * (1 + rv1v / c_speed)
            center2 = cwv * (1 + rv2v / c_speed)
            center_approx = 0.5 * (center1 + center2)
            lw_min = center_approx - halfw
            lw_max = center_approx + halfw

            ep_noise = noise_regions.get(lid, {}).get(ep, [])
            edges_min = [lw_min]
            edges_max = [lw_max]
            for (nm, nM, drc) in ep_noise:
                edges_min.append(nm)
                edges_max.append(nM)
            pm = min(edges_min)
            pM = max(edges_max)
            if pM <= pm:
                pm, pM = lw_min, lw_max

            axm.set_xlim(pm, pM)
            axm.errorbar(wv_d, fl_d, yerr=un_d,
                         fmt='o', color='blue', markersize=4, alpha=0.7,
                         ecolor='blue', elinewidth=1, capsize=2,
                         label='Data')
            axm.plot(wv_d, md_d, 'r-', lw=0.5, label='Combined Model')
            axm.plot(wv_d, s1f, '--', color='magenta', label='Star1')
            axm.plot(wv_d, s2f, '--', color='green', label='Star2')

            # MCMC envelopes => if available
            if (mcmc_bands is not None) and (lid in mcmc_bands) and (ep in mcmc_bands[lid]):
                band_info = mcmc_bands[lid][ep]
                wvb = band_info["wv"]
                fm = band_info["flux_median"]
                fs = band_info["flux_std"]
                axm.fill_between(wvb, fm - fs, fm + fs, color='r', alpha=0.2,
                                 label='±1σ MCMC' if i == 0 else None)

            axm.axvspan(lw_min, lw_max, color='gray', alpha=0.1, label='Line window')
            for idx_n, (nm, nM, drc) in enumerate(ep_noise):
                lb = 'Noise region' if idx_n == 0 else None
                axm.axvspan(nm, nM, color='yellow', alpha=0.2, label=lb)

            axm.grid(alpha=0.3)
            if i == 0:
                axm.legend(loc='best', fontsize=9)

            axr.set_xlim(pm, pM)
            mask_line = (wv_d >= lw_min) & (wv_d <= lw_max)
            wv_line = wv_d[mask_line]
            resid_line = resid[mask_line]
            axr.errorbar(wv_line, resid_line, yerr=np.ones_like(resid_line),
                         fmt='o', color='blue', markersize=4, alpha=0.7,
                         ecolor='blue', elinewidth=1, capsize=2)
            axr.axhline(0, color='r', ls='--', alpha=0.5)

            axr.axvspan(lw_min, lw_max, color='gray', alpha=0.1)
            for idx_n, (nm, nM, drc) in enumerate(ep_noise):
                axr.axvspan(nm, nM, color='yellow', alpha=0.2)

            axr.set_xlabel("Wavelength (Å)")
            axr.set_ylabel("Resid (σ)")
            axr.grid(alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        outpng = os.path.join(outplots, f"epoch_{ep}_fit.png")
        plt.savefig(outpng, dpi=300)
        plt.close()


# This is the main function that other modules import directly - it needs to be
# placed last in the file so all the functions it calls are defined before it
def report_fit_results(
        result,
        # narrower arrays => for chi^2
        wavelengths_line,
        fluxes_line,
        uncertainties_line,
        epochs_line,
        central_wavelengths,
        noise_regions,
        windows,
        output_directory,
        profile_type='sym',
        line_profile='voigt',  # can be 'voigt' or 'gaussian'
        # bigger arrays => for final plotting
        plot_wavelengths_line=None,
        plot_fluxes_line=None,
        plot_uncertainties_line=None,
        plot_epochs_line=None,
        plot_noise_dict=None,
        mcmc_chain=None,
        mjd_dict=None  # New parameter for MJD mapping
):
    """
    Main entry point for writing final results and generating plots:
      1) Compute a final model => chi-square stats => df_chi2
      2) Possibly do MCMC => flux envelopes
      3) Possibly do star1<->star2 swap
      4) Build df_results => epoch RVs
      5) Add ratio, ratio_err, weight columns to df_results => store in final Excel
      6) Insert final weighted ratio into df_chi2 (GLOBAL row)
      7) Write Excel
      8) Make epoch plots, rv2 vs rv1, real_calc if star_rvs_per_epoch.csv
      9) NEW: RV vs time plot if mjd_dict is provided
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    # 1) Evaluate final model on the 'fit arrays' => chi^2
    mod_fit = compute_full_model(
        result.params,
        wavelengths_line,
        epochs_line,
        central_wavelengths,
        profile_type=profile_type,
        line_profile=line_profile
    )

    var_names = list(result.var_names)
    n_params = len(var_names)
    lines_stat = []
    total_chi = 0.0
    total_pts = 0

    # Check if this is a ratio-constrained fit
    ratio_from_fit = 'ratio' in result.params and 'v_sys' in result.params
    if ratio_from_fit:
        ratio_val = result.params['ratio'].value
        ratio_err = result.params['ratio'].stderr if result.params['ratio'].stderr is not None else np.nan
    else:
        ratio_val = np.nan
        ratio_err = np.nan

    for lid in central_wavelengths:
        wv = wavelengths_line[lid]
        if len(wv) == 0:
            continue
        obs = fluxes_line[lid]
        mdl = mod_fit[lid]
        unc = uncertainties_line[lid]
        r = (obs - mdl) / unc
        chi_line = np.sum(r ** 2)
        n_line = len(obs)
        dof_line = n_line - n_params

        total_chi += chi_line
        total_pts += n_line

        if dof_line > 0:
            red_chi = chi_line / dof_line
            pval = 1 - chi2.cdf(chi_line, dof_line)
        else:
            red_chi = np.nan
            pval = np.nan

        lines_stat.append({
            'Line_ID': lid,
            'Chi_Square': chi_line,
            'N_points': n_line,
            'Chi_Square_reduced': red_chi,
            'p_value': pval
        })

    dof_global = total_pts - n_params
    if dof_global > 0:
        global_chi_r = total_chi / dof_global
        global_p = 1 - chi2.cdf(total_chi, dof_global)
    else:
        global_chi_r = np.nan
        global_p = np.nan

    # Add GLOBAL row with current ratio values
    lines_stat.append({
        'Line_ID': 'GLOBAL',
        'Chi_Square': total_chi,
        'N_points': total_pts,
        'Chi_Square_reduced': global_chi_r,
        'p_value': global_p,
        'ratio': ratio_val,
        'ratio_err': ratio_err
    })

    df_chi2 = pd.DataFrame(lines_stat)

    # 2) Possibly do MCMC => flux envelopes
    mcmc_bands = None
    if (mcmc_chain is not None) and (len(mcmc_chain) > 10):
        print("Computing MCMC-based ±1σ model envelopes...")
        mcmc_bands = compute_mcmc_envelopes(
            result=result,
            mcmc_chain=mcmc_chain,
            wv_dict=wavelengths_line,
            ep_dict=epochs_line,
            central_map=central_wavelengths,
            profile_type=profile_type,
            line_profile=line_profile,
            n_draws=1000  # adjust as needed
        )

    # 3) Possibly do star1<->star2 swap
    apply_star_label_swap_if_needed(result, central_wavelengths)

    # 4) Build per-epoch RV table => df_results
    df_results = build_per_epoch_rvs(result)

    # Add MJD column if mjd_dict was provided
    if mjd_dict is not None and df_results is not None and not df_results.empty:
        df_results["MJD"] = df_results["Epoch"].apply(lambda e: mjd_dict.get(e, np.nan))
        print(f"Added MJD column to results DataFrame.")

    # 5) Add ratio columns => ratio_i = - RV1_i / RV2_i, etc.
    # Only compute from RVs if this is not a ratio-constrained fit
    if not df_results.empty and not ratio_from_fit:
        # ratio
        df_results["ratio"] = - df_results["RV1"] / df_results["RV2"]

        # ratio_err via partial derivative
        def compute_ratio_err(row):
            rv1 = row["RV1"]
            rv2 = row["RV2"]
            e1 = row["RV1_err"]
            e2 = row["RV2_err"]
            if abs(rv2) < 1e-8:
                return np.nan
            term1 = (e1 ** 2) / (rv2 ** 2)
            term2 = (rv1 ** 2) * (e2 ** 2) / (rv2 ** 4)
            return np.sqrt(term1 + term2)

        df_results["ratio_err"] = df_results.apply(compute_ratio_err, axis=1)

        # weight = 1/ ratio_err^2
        df_results["weight"] = 1.0 / (df_results["ratio_err"] ** 2)
        # ratio * weight
        df_results["ratio_times_weight"] = df_results["ratio"] * df_results["weight"]

        # Weighted average ratio
        valid_mask = ~(df_results["ratio_err"].isna() | (df_results["ratio_err"] <= 0))
        q_wm = np.nan
        q_wm_err = np.nan
        if valid_mask.any():
            sum_w = df_results.loc[valid_mask, "weight"].sum()
            if sum_w > 0:
                sum_rw = df_results.loc[valid_mask, "ratio_times_weight"].sum()
                q_wm = sum_rw / sum_w
                q_wm_err = np.sqrt(1.0 / sum_w)
            # Optionally compute chi^2 => inflate if needed
            n_valid = valid_mask.sum()
            if n_valid > 1 and not np.isnan(q_wm):
                diffs = df_results.loc[valid_mask, "ratio"] - q_wm
                wghts = df_results.loc[valid_mask, "weight"]
                chi2_ratio = np.sum(wghts * (diffs ** 2))
                dof = n_valid - 1
                chi2_red = chi2_ratio / dof if dof > 0 else np.nan

            # Update global ratio only if not from fit
            mask_global = (df_chi2["Line_ID"] == "GLOBAL")
            if mask_global.any():
                df_chi2.loc[mask_global, "ratio"] = q_wm
                df_chi2.loc[mask_global, "ratio_err"] = q_wm_err

    # 7) Write Excel
    xlsx_path = os.path.join(output_directory, "fit_results.xlsx")
    with pd.ExcelWriter(xlsx_path) as writer:
        df_results.to_excel(writer, sheet_name="Per_Epoch_RVs", index=False)
        df_chi2.to_excel(writer, sheet_name="Chi_Square_Statistics", index=False)

    # 8) Plots => epoch lines, rv2 vs rv1, real_calc if star_rvs_per_epoch.csv, etc.
    pwv = plot_wavelengths_line if plot_wavelengths_line else wavelengths_line
    pfl = plot_fluxes_line if plot_fluxes_line else fluxes_line
    pun = plot_uncertainties_line if plot_uncertainties_line else uncertainties_line
    pep = plot_epochs_line if plot_epochs_line else epochs_line
    pnoise = plot_noise_dict if plot_noise_dict else noise_regions

    plot_epoch_lines(
        result,
        pwv,
        pfl,
        pun,
        pep,
        central_wavelengths,
        pnoise,
        windows,
        df_results,
        df_chi2,
        output_directory,
        profile_type=profile_type,
        line_profile=line_profile,
        mcmc_bands=mcmc_bands
    )

    plot_rv2_vs_rv1(df_results, output_directory)

    # If star_rvs_per_epoch.csv => compare real vs calc
    star_csv = os.path.join(os.path.dirname(output_directory), "star_rvs_per_epoch.csv")
    if os.path.exists(star_csv):
        real_df = pd.read_csv(star_csv)
        plot_real_calc_comparison(df_results, real_df, output_directory)

    # 9) NEW: Plot RV vs time if we have MJD data
    if mjd_dict is not None and df_results is not None and not df_results.empty and "MJD" in df_results.columns:
        # Check if we have any valid MJD values
        if not df_results["MJD"].isna().all():
            plot_rv_vs_time(df_results, output_directory)
        else:
            print("No valid MJD values found, skipping RV vs time plot.")

    return df_results, df_chi2