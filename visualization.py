import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from src.model_builder import compute_full_model
from src.utils import voigt_profile

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
    output_folder
):
    """
    Creates per-epoch line plots (data + fitted model + residuals).

    NEW LOGIC to ensure we see both noise regions and line window:
    - We define a broad_xmin and broad_xmax that covers:
         min( leftmost noise region, line_window_left )
       to
         max( rightmost noise region, line_window_right ).
    - We gather data only within that broad range.
    - We highlight noise regions in yellow, line window in gray.
    - Outside the line window, the model is forced to baseline=1.
    """

    all_epochs = np.unique(np.concatenate(list(epochs_line.values())))
    params = result.params
    has_ratio = ('ratio' in params)

    for epoch in all_epochs:
        ep_int = int(epoch)
        if has_ratio:
            ratio_val = params['ratio'].value
            rv2_val = params[f'rv2_epoch{ep_int}'].value
            rv1_val = - ratio_val * rv2_val
        else:
            rv1_val = params[f'rv1_epoch{ep_int}'].value
            rv2_val = params[f'rv2_epoch{ep_int}'].value

        n_lines = len(central_wavelengths)
        fig, axes = plt.subplots(n_lines, 2, figsize=(12, 4*n_lines), sharex=False)
        if n_lines == 1:
            axes = [axes]

        for i, (line_id, cwav) in enumerate(central_wavelengths.items()):
            main_ax, res_ax = axes[i]

            (plot_w,
             plot_f,
             plot_e,
             model_flux,
             star1_flux,
             star2_flux,
             broad_xmin,
             broad_xmax,
             noise_list) = get_line_epoch_data(
                line_id, ep_int,
                wavelengths_line, fluxes_line, uncertainties_line, epochs_line,
                cwav, windows[line_id], noise_regions, params,
                rv1_val, rv2_val
            )

            # --- Main Plot ---
            main_ax.errorbar(
                plot_w, plot_f, yerr=plot_e, fmt='o',
                color='blue', markersize=3, alpha=0.7, label='Data'
            )

            # Highlight noise region(s) in yellow
            for nr_i, (nm, nM, direction) in enumerate(noise_list):
                if nr_i == 0:
                    main_ax.axvspan(nm, nM, color='yellow', alpha=0.2, label='Noise region')
                else:
                    main_ax.axvspan(nm, nM, color='yellow', alpha=0.2)

            # Highlight line window in gray
            halfw = windows[line_id] / 2
            line_left = cwav - halfw
            line_right= cwav + halfw
            main_ax.axvspan(
                line_left, line_right,
                color='gray', alpha=0.1, label='Line Window'
            )

            # Combined model, star1, star2
            main_ax.plot(plot_w, model_flux, 'r-', lw=2, label='Combined Model')
            main_ax.plot(plot_w, star1_flux, '--', color='purple', label='Star 1')
            main_ax.plot(plot_w, star2_flux, '--', color='green', label='Star 2')

            main_ax.set_ylabel("Normalized Flux")
            main_ax.set_title(
                f"{line_id} ({cwav:.1f} Å), Ep {ep_int}\n"
                f"RV1={rv1_val:.2f}, RV2={rv2_val:.2f}"
            )
            main_ax.grid(alpha=0.3)
            main_ax.legend(loc='best', fontsize='small')

            # Residuals
            residuals = (plot_f - model_flux) / plot_e
            res_ax.errorbar(
                plot_w, residuals, yerr=np.ones_like(residuals),
                fmt='o', color='blue', markersize=3, alpha=0.7
            )
            res_ax.axhline(0, color='r', ls='--', alpha=0.5)
            res_ax.set_xlabel("Wavelength (Å)")
            res_ax.set_ylabel("Residuals (σ)")
            res_ax.grid(alpha=0.3)

            # Force x-range to show the entire broad region
            main_ax.set_xlim(broad_xmin, broad_xmax)
            res_ax.set_xlim(broad_xmin, broad_xmax)

            # Y-limits for residuals
            rmax = np.max(np.abs(residuals))
            res_ax.set_ylim(-max(3.5, rmax + 1), max(3.5, rmax + 1))

        plt.tight_layout()
        outname = os.path.join(output_folder, f'epoch_{ep_int}_fit.png')
        plt.savefig(outname, dpi=300)
        plt.close()


def get_line_epoch_data(
    line_id,
    epoch_int,
    wavelengths_line,
    fluxes_line,
    uncertainties_line,
    epochs_line,
    cwav,
    line_window,
    noise_regions_dict,
    params,
    rv1_val,
    rv2_val
):
    """
    Decide a broad range to show on the x-axis:
      - line window: [cwav-line_window/2, cwav+line_window/2]
      - noise region(s): if exist, we combine them
    Then we build the data & model within that broad range.
    Outside the line window, model => baseline=1.

    Returns (plot_w, plot_f, plot_e, model_flux, star1_flux, star2_flux, broad_xmin, broad_xmax, noise_list).
    """

    idx = (epochs_line[line_id] == epoch_int)
    wv = wavelengths_line[line_id][idx]
    fv = fluxes_line[line_id][idx]
    ev = uncertainties_line[line_id][idx]

    # For convenience
    halfw = line_window / 2
    line_left = cwav - halfw
    line_right= cwav + halfw

    noise_list = noise_regions_dict[line_id].get(epoch_int, [])

    if len(noise_list) == 0:
        # No noise => broad range = line window
        broad_xmin = line_left
        broad_xmax = line_right
    else:
        # We do min( all noise mins, line_left ) -> max( all noise maxs, line_right )
        n_min = min(n[0] for n in noise_list)
        n_max = max(n[1] for n in noise_list)
        broad_xmin = min(line_left, n_min)
        broad_xmax = max(line_right, n_max)

    # Now subset data in [broad_xmin, broad_xmax]
    broad_mask = (wv >= broad_xmin) & (wv <= broad_xmax)
    plot_w = wv[broad_mask]
    plot_f = fv[broad_mask]
    plot_e = ev[broad_mask]

    # Compute model in the same broad range
    from src.model_builder import compute_full_model
    temp_model = compute_full_model(
        params,
        {line_id: plot_w},
        {line_id: np.full_like(plot_w, epoch_int)},
        {line_id: cwav}
    )
    model_flux = temp_model[line_id]

    # Star1, star2
    a1 = params[f'a1_{line_id}'].value
    s1 = params[f'sigma1_{line_id}'].value
    g1 = params[f'gamma1_{line_id}'].value
    a2 = params[f'a2_{line_id}'].value
    s2 = params[f'sigma2_{line_id}'].value
    g2 = params[f'gamma2_{line_id}'].value

    c_light = 299792.458
    sc1 = cwav*(1 + rv1_val/c_light)
    sc2 = cwav*(1 + rv2_val/c_light)

    star1_profile = voigt_profile(plot_w, -abs(a1), sc1, s1, g1)
    star2_profile = voigt_profile(plot_w, -abs(a2), sc2, s2, g2)
    star1_flux = 1.0 + star1_profile
    star2_flux = 1.0 + star2_profile

    # Force baseline=1 outside the line window
    mask_line = (plot_w >= line_left) & (plot_w <= line_right)
    model_flux[~mask_line] = 1.0
    star1_flux[~mask_line] = 1.0
    star2_flux[~mask_line] = 1.0

    return (
        plot_w,
        plot_f,
        plot_e,
        model_flux,
        star1_flux,
        star2_flux,
        broad_xmin,
        broad_xmax,
        noise_list
    )


def plot_rv2_vs_rv1(calc_rv_results, output_folder, real_df=None):
    """
    RV2 vs RV1 with residuals; no line connecting data points
    """
    rv1 = calc_rv_results['RV1'].values
    rv2 = calc_rv_results['RV2'].values
    rv1_err = calc_rv_results['RV1_uncertainty'].values
    rv2_err = calc_rv_results['RV2_uncertainty'].values

    rv1_err[rv1_err <= 0] = 1e-5
    rv2_err[rv2_err <= 0] = 1e-5

    try:
        popt, pcov = curve_fit(linear_func, rv1, rv2, sigma=rv2_err, absolute_sigma=True)
        slope, intercept = popt
        slope_err, intercept_err = np.sqrt(np.diag(pcov))
    except RuntimeError:
        slope, intercept = np.nan, np.nan
        slope_err, intercept_err = np.nan, np.nan

    rv2_fit = linear_func(rv1, slope, intercept)
    residuals = rv2 - rv2_fit

    fig, (ax_main, ax_res) = plt.subplots(2,1, figsize=(8,10), sharex=True)
    ax_main.errorbar(rv1, rv2, xerr=rv1_err, yerr=rv2_err, fmt='o', color='blue', alpha=0.7,
                     label='Calculated RVs')
    xfit = np.linspace(rv1.min(), rv1.max(), 100)
    yfit = linear_func(xfit, slope, intercept)
    if not np.isnan(slope):
        ax_main.plot(xfit, yfit, 'r--',
                     label=f"slope={slope:.3f}±{slope_err:.3f}, intercept={intercept:.3f}±{intercept_err:.3f}")
    ax_main.set_ylabel("RV2 (km/s)")
    ax_main.set_title("RV2 vs RV1")
    ax_main.legend()
    ax_main.grid(alpha=0.3)

    ax_res.errorbar(rv1, residuals, xerr=rv1_err, yerr=rv2_err, fmt='o', color='blue', alpha=0.7)
    ax_res.axhline(0, color='r', ls='--', alpha=0.5)
    ax_res.set_xlabel("RV1 (km/s)")
    ax_res.set_ylabel("Residual (km/s)")
    ax_res.grid(alpha=0.3)

    plt.tight_layout()
    outpng = os.path.join(output_folder, 'rv2_vs_rv1.png')
    plt.savefig(outpng, dpi=300)
    plt.close()


def plot_real_calc_comparison(calc_rv_results, real_df, output_folder):
    """
    Compare final calculated RVs with real star RVs. No lines connecting points; error bars in residual.
    """
    merged = pd.merge(calc_rv_results, real_df, on='Epoch', suffixes=('_calc','_real'))
    epochs = merged['Epoch'].values

    rv1_calc = merged['RV1_calc'].values
    rv1_real = merged['RV1_real'].values
    rv1_err  = merged.get('RV1_uncertainty', pd.Series([0]*len(merged))).values
    rv2_calc = merged['RV2_calc'].values
    rv2_real = merged['RV2_real'].values
    rv2_err  = merged.get('RV2_uncertainty', pd.Series([0]*len(merged))).values

    rv1_diff = rv1_calc - rv1_real
    rv2_diff = rv2_calc - rv2_real
    rv1_diff_err = rv1_err
    rv2_diff_err = rv2_err

    fig = plt.figure(figsize=(10,10))
    gs = fig.add_gridspec(4,1, height_ratios=[3,1,3,1])
    ax_rv1    = fig.add_subplot(gs[0])
    ax_rv1res = fig.add_subplot(gs[1])
    ax_rv2    = fig.add_subplot(gs[2])
    ax_rv2res = fig.add_subplot(gs[3])

    # RV1 vs epoch
    ax_rv1.errorbar(epochs, rv1_calc, yerr=rv1_err, fmt='ro', markersize=4, capsize=3, label='Calc RV1')
    ax_rv1.plot(epochs, rv1_real, 'bo', markersize=4, label='Real RV1')
    for i, ep in enumerate(epochs):
        ax_rv1.plot([ep, ep], [rv1_real[i], rv1_calc[i]], 'k--', alpha=0.5)
    ax_rv1.set_ylabel("RV1 (km/s)")
    ax_rv1.set_title("Real vs. Calc: RV1")
    ax_rv1.legend()
    ax_rv1.grid(alpha=0.3)

    # RV1 residual
    ax_rv1res.errorbar(epochs, rv1_diff, yerr=rv1_diff_err, fmt='ko', markersize=4, capsize=3)
    ax_rv1res.axhline(0, color='r', ls='--', alpha=0.5)
    ax_rv1res.set_ylabel("Diff (km/s)")
    ax_rv1res.grid(alpha=0.3)

    # RV2 vs epoch
    ax_rv2.errorbar(epochs, rv2_calc, yerr=rv2_err, fmt='ro', markersize=4, capsize=3, label='Calc RV2')
    ax_rv2.plot(epochs, rv2_real, 'bo', markersize=4, label='Real RV2')
    for i, ep in enumerate(epochs):
        ax_rv2.plot([ep, ep], [rv2_real[i], rv2_calc[i]], 'k--', alpha=0.5)
    ax_rv2.set_ylabel("RV2 (km/s)")
    ax_rv2.set_title("Real vs. Calc: RV2")
    ax_rv2.legend()
    ax_rv2.grid(alpha=0.3)

    # RV2 residual
    ax_rv2res.errorbar(epochs, rv2_diff, yerr=rv2_diff_err, fmt='ko', markersize=4, capsize=3)
    ax_rv2res.axhline(0, color='r', ls='--', alpha=0.5)
    ax_rv2res.set_xlabel("Epoch")
    ax_rv2res.set_ylabel("Diff (km/s)")
    ax_rv2res.grid(alpha=0.3)

    plt.tight_layout()
    outpng = os.path.join(output_folder, 'real_calc_comparison.png')
    plt.savefig(outpng, dpi=300)
    plt.close()
