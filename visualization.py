import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from src.model_builder import compute_full_model
from src.utils import voigt_profile, skewed_voigt_profile

def linear_func(x, a, b):
    return a*x + b

def plot_epoch_lines(
    result,
    wavelengths_line,
    fluxes_line,
    uncertainties_line,
    epochs_line,
    central_wavelengths,
    noise_regions,
    windows,
    wavelengths_plot,
    fluxes_plot,
    epochs_plot,
    output_folder,
    line_stats=None,
    per_epoch_df=None
):
    """
    Create per-epoch line plots showing:
      - data
      - total model
      - star1, star2 components
      - optional baseline
      - residuals
      - line-by-line chisq info in the plot
      - global chisq info
      - (RV1, RV2) for that epoch
    """
    # Convert line_stats from DF to dictionary for easy lookup
    line_chi_map = {}
    global_chi2r = np.nan
    global_pval = np.nan

    if line_stats is not None:
        # line_stats is a DF with columns: 'Line','Chi_square_reduced','p_value', etc.
        # We map line_id -> (chi2r, pval)
        for _, row in line_stats.iterrows():
            if row['Line']=='GLOBAL':
                global_chi2r = row['Chi_square_reduced']
                global_pval  = row['p_value']
            else:
                lid = row['Line']
                chi2r = row['Chi_square_reduced']
                pval = row['p_value']
                line_chi_map[lid] = (chi2r, pval)

    # Sort epochs
    all_epochs = np.unique(np.concatenate(list(epochs_line.values())))

    # For each epoch, we create a figure of N_lines x 2 subplots
    from src.model_builder import compute_full_model
    c_speed = 299792.458

    # We'll also build a dict from per_epoch_df:
    #   epoch -> (rv1, erv1, rv2, erv2)
    epoch_rv_map = {}
    if per_epoch_df is not None:
        for _, row in per_epoch_df.iterrows():
            e = int(row['Epoch'])
            epoch_rv_map[e] = (row['RV1'], row['RV1_uncertainty'], row['RV2'], row['RV2_uncertainty'])

    for ep in all_epochs:
        ep_int = int(ep)
        # gather star RV for this epoch if available
        rv1_val, rv1_err, rv2_val, rv2_err = (np.nan, np.nan, np.nan, np.nan)
        if ep_int in epoch_rv_map:
            rv1_val, rv1_err, rv2_val, rv2_err = epoch_rv_map[ep_int]

        n_lines = len(central_wavelengths)
        fig, axes = plt.subplots(n_lines, 2, figsize=(12, 4*n_lines), sharex=False)
        if n_lines==1:
            axes = [axes]  # ensure it's a list

        # We can place a big title at the top with global info
        fig.suptitle(
            f"Epoch {ep_int}: "
            f"RV1={rv1_val:.2f}±{rv1_err:.2f}, "
            f"RV2={rv2_val:.2f}±{rv2_err:.2f}\n"
            f"GLOBAL chi2r={global_chi2r:.3f}, pval={global_pval:.3g}",
            fontsize=14
        )

        for i, (line_id, cwav) in enumerate(central_wavelengths.items()):
            # This line's stats
            line_chi2r, line_pval = line_chi_map.get(line_id, (np.nan, np.nan))

            ax_main, ax_res = axes[i]
            idx = (epochs_line[line_id]==ep_int)
            if not np.any(idx):
                # no data for this epoch, skip
                ax_main.text(0.5, 0.5, "No Data", transform=ax_main.transAxes, ha='center', va='center')
                ax_res.text(0.5, 0.5, "No Data", transform=ax_res.transAxes, ha='center', va='center')
                continue

            wv  = wavelengths_line[line_id][idx]
            fl  = fluxes_line[line_id][idx]
            err = uncertainties_line[line_id][idx]

            # Build the model
            # We'll compute star1 and star2 separately to plot them individually:
            baseline_key = f'baseline_{line_id}'
            if baseline_key in result.params:
                baseline_val = result.params[baseline_key].value
            else:
                baseline_val = 1.0

            a1 = result.params[f'a1_{line_id}'].value
            s1 = result.params[f'sigma1_{line_id}'].value
            g1 = result.params[f'gamma1_{line_id}'].value
            a2 = result.params[f'a2_{line_id}'].value
            s2 = result.params[f'sigma2_{line_id}'].value
            g2 = result.params[f'gamma2_{line_id}'].value

            skew1_val = 0.0
            skew2_val = 0.0
            if f'skew1_{line_id}' in result.params:
                skew1_val = result.params[f'skew1_{line_id}'].value
            if f'skew2_{line_id}' in result.params:
                skew2_val = result.params[f'skew2_{line_id}'].value

            if 'ratio' in result.params:
                # ratio approach => rv2 param only
                rv2 = result.params[f'rv2_epoch{ep_int}'].value
                rv1 = - result.params['ratio'].value * rv2
            else:
                rv1 = result.params[f'rv1_epoch{ep_int}'].value
                rv2 = result.params[f'rv2_epoch{ep_int}'].value

            center1 = cwav*(1 + rv1/c_speed)
            center2 = cwav*(1 + rv2/c_speed)

            if abs(skew1_val)>1e-8:
                star1 = skewed_voigt_profile(wv, -abs(a1), center1, s1, g1, skew1_val)
            else:
                star1 = voigt_profile(wv, -abs(a1), center1, s1, g1)

            if abs(skew2_val)>1e-8:
                star2 = skewed_voigt_profile(wv, -abs(a2), center2, s2, g2, skew2_val)
            else:
                star2 = voigt_profile(wv, -abs(a2), center2, s2, g2)

            tot_model = baseline_val + star1 + star2

            # Plot main
            ax_main.errorbar(wv, fl, yerr=err, fmt='o', color='blue', ms=3, alpha=0.7, label='Data')
            ax_main.plot(wv, tot_model, 'r-', lw=2, label='Total Model')
            ax_main.plot(wv, baseline_val+star1, '--', color='magenta', label='Star1')
            ax_main.plot(wv, baseline_val+star2, '--', color='green', label='Star2')

            # highlight noise region in yellow, line window in grey
            half_w = windows[line_id]/2
            lleft = cwav - half_w
            lright= cwav + half_w
            ax_main.axvspan(lleft, lright, color='gray', alpha=0.1, label='Line Window')
            nr_list = noise_regions[line_id].get(ep_int, [])
            for (nm, nM, direc) in nr_list:
                ax_main.axvspan(nm, nM, color='yellow', alpha=0.2)

            # Title with line stats
            ax_main.set_title(
                f"{line_id} (chi2r={line_chi2r:.3f}, pval={line_pval:.3g})",
                fontsize=11
            )
            ax_main.set_ylabel("Flux")
            ax_main.legend(loc='best', fontsize='small')
            ax_main.grid(alpha=0.3)

            # Residuals
            resid = (fl - tot_model)/err
            ax_res.axhline(0, color='k', ls='--', alpha=0.6)
            ax_res.errorbar(wv, resid, yerr=np.ones_like(resid), fmt='o', color='blue', ms=3, alpha=0.7)
            ax_res.set_ylabel("Resid (σ)")
            ax_res.set_xlabel("Wavelength (Å)")
            ax_res.grid(alpha=0.3)

            # Fit x-lims to data
            ax_main.set_xlim(wv.min(), wv.max())
            ax_res.set_xlim(wv.min(), wv.max())

        plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
        outpng = os.path.join(output_folder, f"epoch_{ep_int}_fit.png")
        plt.savefig(outpng, dpi=300)
        plt.close()

def plot_rv2_vs_rv1(calc_rv_results, output_folder, real_df=None):
    """
    Plot RV2 vs RV1 with slope fit.
    """
    rv1 = calc_rv_results['RV1'].values
    rv2 = calc_rv_results['RV2'].values
    rv1_err = calc_rv_results['RV1_uncertainty'].values
    rv2_err = calc_rv_results['RV2_uncertainty'].values

    rv1_err[rv1_err<=0] = 1e-5
    rv2_err[rv2_err<=0] = 1e-5

    try:
        popt, pcov = curve_fit(linear_func, rv1, rv2, sigma=rv2_err, absolute_sigma=True)
        slope, intercept = popt
        slope_err, intercept_err = np.sqrt(np.diag(pcov))
    except RuntimeError:
        slope, intercept = np.nan, np.nan
        slope_err, intercept_err = np.nan, np.nan

    rv2_fit = linear_func(rv1, slope, intercept)
    resid = rv2 - rv2_fit

    fig, (ax1, ax2) = plt.subplots(2,1, figsize=(8,10), sharex=True)
    ax1.errorbar(rv1, rv2, xerr=rv1_err, yerr=rv2_err, fmt='o', color='b', alpha=0.7)
    xline = np.linspace(rv1.min(), rv1.max(), 200)
    yline = linear_func(xline, slope, intercept)
    if not np.isnan(slope):
        ax1.plot(xline, yline, 'r--',
            label=f"slope={slope:.2f}±{slope_err:.2f}, intcpt={intercept:.2f}±{intercept_err:.2f}")
    ax1.grid(alpha=0.3)
    ax1.legend()
    ax1.set_ylabel("RV2 (km/s)")
    ax1.set_title("RV2 vs RV1")

    ax2.errorbar(rv1, resid, xerr=rv1_err, yerr=rv2_err, fmt='o', color='b', alpha=0.7)
    ax2.axhline(0, color='r', ls='--', alpha=0.6)
    ax2.set_xlabel("RV1 (km/s)")
    ax2.set_ylabel("Resid (km/s)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    outpng = os.path.join(output_folder, "rv2_vs_rv1.png")
    plt.savefig(outpng, dpi=300)
    plt.close()

def plot_real_calc_comparison(calc_rv_results, real_df, output_folder):
    """
    Compare final calculated RVs with real star RVs.
    """
    merged = pd.merge(calc_rv_results, real_df, on='Epoch', suffixes=('_calc','_real'))
    epochs = merged['Epoch'].values
    rv1_calc = merged['RV1_calc'].values
    rv1_real = merged['RV1_real'].values
    rv1_err = merged.get('RV1_uncertainty', pd.Series([0]*len(merged))).values
    rv2_calc = merged['RV2_calc'].values
    rv2_real = merged['RV2_real'].values
    rv2_err = merged.get('RV2_uncertainty', pd.Series([0]*len(merged))).values

    rv1_diff = rv1_calc - rv1_real
    rv2_diff = rv2_calc - rv2_real

    fig = plt.figure(figsize=(10,10))
    gs = fig.add_gridspec(4,1, height_ratios=[3,1,3,1])
    ax_rv1    = fig.add_subplot(gs[0])
    ax_rv1res = fig.add_subplot(gs[1])
    ax_rv2    = fig.add_subplot(gs[2])
    ax_rv2res = fig.add_subplot(gs[3])

    # RV1
    ax_rv1.errorbar(epochs, rv1_calc, yerr=rv1_err, fmt='ro', label='Calc RV1')
    ax_rv1.plot(epochs, rv1_real, 'bo', label='Real RV1')
    for i,ep in enumerate(epochs):
        ax_rv1.plot([ep, ep], [rv1_real[i], rv1_calc[i]], 'k--', alpha=0.4)
    ax_rv1.legend()
    ax_rv1.grid(alpha=0.3)
    ax_rv1.set_ylabel("RV1 (km/s)")
    ax_rv1.set_title("Real vs Calc: RV1")

    # residual
    ax_rv1res.axhline(0, color='r', ls='--', alpha=0.6)
    ax_rv1res.errorbar(epochs, rv1_diff, yerr=rv1_err, fmt='ko', alpha=0.8)
    ax_rv1res.grid(alpha=0.3)
    ax_rv1res.set_ylabel("Diff (km/s)")

    # RV2
    ax_rv2.errorbar(epochs, rv2_calc, yerr=rv2_err, fmt='ro', label='Calc RV2')
    ax_rv2.plot(epochs, rv2_real, 'bo', label='Real RV2')
    for i,ep in enumerate(epochs):
        ax_rv2.plot([ep, ep], [rv2_real[i], rv2_calc[i]], 'k--', alpha=0.4)
    ax_rv2.legend()
    ax_rv2.grid(alpha=0.3)
    ax_rv2.set_ylabel("RV2 (km/s)")
    ax_rv2.set_title("Real vs Calc: RV2")

    # residual
    ax_rv2res.axhline(0, color='r', ls='--', alpha=0.6)
    ax_rv2res.errorbar(epochs, rv2_diff, yerr=rv2_err, fmt='ko', alpha=0.8)
    ax_rv2res.grid(alpha=0.3)
    ax_rv2res.set_xlabel("Epoch")
    ax_rv2res.set_ylabel("Diff (km/s)")

    plt.tight_layout()
    outpng = os.path.join(output_folder, 'real_calc_comparison.png')
    plt.savefig(outpng, dpi=300)
    plt.close()
