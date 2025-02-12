#!/usr/bin/env python3
# plot_results.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2

from src.model_builder import compute_full_model, get_star_components
from src.standard_fit_model import analyze_line_quality  # if needed in residual checks


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
    # bigger arrays => for final plotting
    plot_wavelengths_line=None,
    plot_fluxes_line=None,
    plot_uncertainties_line=None,
    plot_epochs_line=None,
    plot_noise_dict=None
):
    """
    Summarizes the fit, computes chi^2, saves results to Excel,
    and makes diagnostic plots.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    # 1) Evaluate final model on the 'fit arrays'
    mod_fit = compute_full_model(
        result.params,
        wavelengths_line,
        epochs_line,
        central_wavelengths,
        profile_type=profile_type
    )

    var_names = list(result.var_names)
    n_params = len(var_names)
    lines_stat = []
    total_chi = 0.0
    total_pts = 0

    for lid in central_wavelengths:
        wv = wavelengths_line[lid]
        if len(wv) == 0:
            continue
        obs = fluxes_line[lid]
        mdl = mod_fit[lid]
        unc = uncertainties_line[lid]
        r = (obs - mdl) / unc

        chi_line = np.sum(r**2)
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

    ratio_val = np.nan
    ratio_err = np.nan
    if 'ratio' in result.params:
        ratio_val = result.params['ratio'].value
        if result.params['ratio'].stderr is not None:
            ratio_err = result.params['ratio'].stderr

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

    # 2) build per-epoch rvs
    df_results = build_per_epoch_rvs(result, ratio_val, ratio_err)

    # 3) Write excel
    xlsx_path = os.path.join(output_directory, "fit_results.xlsx")
    with pd.ExcelWriter(xlsx_path) as writer:
        df_results.to_excel(writer, sheet_name="Per_Epoch_RVs", index=False)
        df_chi2.to_excel(writer, sheet_name="Chi_Square_Statistics", index=False)

    # 4) If we have 'plot arrays', use them; else fallback
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
        profile_type
    )

    # 5) rv2 vs rv1
    plot_rv2_vs_rv1(df_results, output_directory)

    # 6) real vs calc if star_rvs_per_epoch.csv found
    star_csv = os.path.join(os.path.dirname(output_directory), "star_rvs_per_epoch.csv")
    if os.path.exists(star_csv):
        real_df = pd.read_csv(star_csv)
        plot_real_calc_comparison(df_results, real_df, output_directory)

    return df_results, df_chi2


def build_per_epoch_rvs(result, ratio_val, ratio_err):
    """
    Builds a DataFrame of (Epoch, RV1, RV1_err, RV2, RV2_err) from the final fit.
    Then does a post-fit amplitude approach to ensure star1 is the lower-amplitude star.
    """
    import numpy as np
    import pandas as pd

    all_keys = list(result.params.keys())
    ep_list = set()

    # find which epochs exist
    for pk in all_keys:
        if pk.startswith("rv1_epoch"):
            ep_str = pk.replace("rv1_epoch", "")
            try:
                ep = int(ep_str)
                ep_list.add(ep)
            except:
                pass
        elif pk.startswith("rv2_epoch"):
            ep_str = pk.replace("rv2_epoch", "")
            try:
                ep = int(ep_str)
                ep_list.add(ep)
            except:
                pass

    ep_nums = sorted(list(ep_list))
    has_ratio = ('ratio' in result.params)

    rows = []
    for ep in ep_nums:
        p_rv1 = result.params.get(f"rv1_epoch{ep}", None)
        p_rv2 = result.params.get(f"rv2_epoch{ep}", None)

        if has_ratio and (p_rv2 is not None):
            # ratio approach might define rv1 from rv2, but let's just store them directly
            rv2_val = p_rv2.value
            rv2_err = p_rv2.stderr if (p_rv2.stderr is not None) else 0.0
            # if we followed ratio logic, we could do:
            # rv1_val = - ratio_val*(rv2_val + v_sys) - v_sys
            # but let's just store param's version
            if p_rv1 is not None:
                rv1_val = p_rv1.value
                rv1_err = p_rv1.stderr if (p_rv1.stderr is not None) else 0.0
            else:
                rv1_val = 0.0
                rv1_err = 0.0
        else:
            # standard approach
            if p_rv1 is not None:
                rv1_val = p_rv1.value
                rv1_err = p_rv1.stderr if (p_rv1.stderr is not None) else 0.0
            else:
                rv1_val = 0.0
                rv1_err = 0.0
            if p_rv2 is not None:
                rv2_val = p_rv2.value
                rv2_err = p_rv2.stderr if (p_rv2.stderr is not None) else 0.0
            else:
                rv2_val = 0.0
                rv2_err = 0.0

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

    # --- Post-Fit Amplitude Approach ---
    # 1) measure each star's amplitude around its own mean
    rv1_mean = df["RV1"].mean()
    rv2_mean = df["RV2"].mean()

    rv1_amp = np.sqrt(np.mean((df["RV1"] - rv1_mean)**2))
    rv2_amp = np.sqrt(np.mean((df["RV2"] - rv2_mean)**2))

    # 2) If RV1 has LARGER amplitude than RV2 => swap columns
    # Because star1 should be the more massive star => smaller amplitude
    if rv1_amp > rv2_amp:
        tmp_rv1 = df["RV1"].copy()
        tmp_rv1e = df["RV1_err"].copy()

        df["RV1"] = df["RV2"]
        df["RV1_err"] = df["RV2_err"]
        df["RV2"] = tmp_rv1
        df["RV2_err"] = tmp_rv1e

    return df


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
    out_dir,
    profile_type='sym'
):
    """
    Plots each epoch's line fit: data vs combined model, star1 & star2, plus residuals
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from src.model_builder import compute_full_model, get_star_components

    df_global = df_chi2[df_chi2["Line_ID"] == "GLOBAL"]
    if len(df_global) == 1:
        global_chi = df_global["Chi_Square_reduced"].values[0]
        global_p = df_global["p_value"].values[0]
    else:
        global_chi = np.nan
        global_p = np.nan

    ratio_val = np.nan
    ratio_err = np.nan
    if 'ratio' in result.params:
        ratio_val = result.params['ratio'].value
        ratio_err = result.params['ratio'].stderr or np.nan

    line_map = {}
    for _, row in df_chi2.iterrows():
        if row["Line_ID"]=="GLOBAL":
            continue
        line_map[row["Line_ID"]] = (row["Chi_Square_reduced"], row["p_value"])

    model_flux = compute_full_model(
        result.params,
        wv_line_dict,
        ep_line_dict,
        central_wavelengths,
        profile_type=profile_type
    )

    all_ep = []
    for lid in wv_line_dict:
        if len(ep_line_dict[lid])>0:
            all_ep.extend(ep_line_dict[lid])
    all_ep = np.unique(all_ep)

    # ep->(RV1±err, RV2±err)
    ep2rv = {}
    for _, row in df_results.iterrows():
        e = row["Epoch"]
        ep2rv[e] = (row["RV1"], row["RV1_err"], row["RV2"], row["RV2_err"])

    outplots = os.path.join(out_dir,"epoch_plots")
    os.makedirs(outplots, exist_ok=True)

    for ep in all_ep:
        lines_for_ep = []
        for lid in wv_line_dict:
            if np.any(ep_line_dict[lid]==ep):
                lines_for_ep.append(lid)
        if not lines_for_ep:
            continue

        rv1v, rv1e, rv2v, rv2e = ep2rv.get(ep,(np.nan,)*4)
        suptt = (f"Epoch {ep} | RV1={rv1v:.2f}±{rv1e:.2f}, "
                 f"RV2={rv2v:.2f}±{rv2e:.2f}\n"
                 f"Global χ²ᵣ= {global_chi:.3f}, p= {global_p:.3g}")
        if not np.isnan(ratio_val):
            suptt += f"\nratio= {ratio_val:.2f}±{ratio_err:.2f}"

        fig, axes = plt.subplots(len(lines_for_ep),2, figsize=(14,4*len(lines_for_ep)), sharex=False)
        if len(lines_for_ep)==1:
            axes=[axes]

        fig.suptitle(suptt, fontsize=13)

        for i, lid in enumerate(lines_for_ep):
            axm, axr = axes[i]
            mk = (ep_line_dict[lid]==ep)
            wv_d = wv_line_dict[lid][mk]
            fl_d = fl_line_dict[lid][mk]
            un_d = un_line_dict[lid][mk]
            md_d = model_flux[lid][mk]

            resid = (fl_d - md_d)/un_d

            line_chi, line_p = line_map.get(lid,(np.nan, np.nan))
            restwv = central_wavelengths[lid]
            line_title = f"Line {lid} (~{restwv:.1f}) => χ²ᵣ= {line_chi:.3f}, p= {line_p:.3g}"
            axm.set_title(line_title, fontsize=10)

            # star1, star2
            s1f, s2f = get_star_components(result.params, lid, wv_d, restwv,
                                          rv1v, rv2v, profile_type)

            leftx = wv_d.min()
            rightx= wv_d.max()
            ep_noise = noise_regions.get(lid, {}).get(ep, [])

            axm.errorbar(wv_d, fl_d, yerr=un_d,
                         fmt='o', color='blue', markersize=4, alpha=0.7,
                         ecolor='blue', elinewidth=1, capsize=2,
                         label='Data')
            axm.plot(wv_d, md_d, 'r-', lw=2, label='Combined Model')
            axm.plot(wv_d, s1f, '--', color='magenta', label='Star1')
            axm.plot(wv_d, s2f, '--', color='green',  label='Star2')

            axm.axvspan(leftx, rightx, color='gray', alpha=0.1, label='Line window')
            for idx_n,(nm,nM,drc) in enumerate(ep_noise):
                lb= 'Noise region' if idx_n==0 else None
                axm.axvspan(nm,nM,color='yellow',alpha=0.2,label=lb)

            axm.grid(alpha=0.3)
            if i==0:
                axm.legend(loc='best', fontsize=9)

            axr.errorbar(wv_d, resid, yerr=np.ones_like(resid),
                         fmt='o', color='blue', markersize=4, alpha=0.7,
                         ecolor='blue', elinewidth=1, capsize=2)
            axr.axhline(0, color='r', ls='--', alpha=0.5)
            axr.axvspan(leftx, rightx, color='gray', alpha=0.1)
            for idx_n,(nm,nM,drc) in enumerate(ep_noise):
                axr.axvspan(nm,nM, color='yellow', alpha=0.2)

            axr.set_xlabel("Wavelength (Å)")
            axr.set_ylabel("Resid (σ)")
            axr.grid(alpha=0.3)

        fig.tight_layout(rect=[0,0,1,0.92])
        outpng= os.path.join(outplots, f"epoch_{ep}_fit.png")
        plt.savefig(outpng,dpi=300)
        plt.close()


def plot_rv2_vs_rv1(df_rvs, outdir):
    """
    Quick scatter of final RV2 vs RV1 for debugging
    """
    if df_rvs.empty:
        return
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    rv1 = df_rvs["RV1"].values
    rv2 = df_rvs["RV2"].values
    e1 = df_rvs["RV1_err"].values
    e2 = df_rvs["RV2_err"].values
    e1[e1<=0]=1e-5
    e2[e2<=0]=1e-5

    def lin_func(x,a,b):
        return a*x+b

    try:
        popt, pcov = curve_fit(lin_func, rv1, rv2, sigma=e2, absolute_sigma=True)
        slope, intercept = popt
        slope_err, intercept_err = np.sqrt(np.diag(pcov))
    except:
        slope, intercept, slope_err, intercept_err= (np.nan,)*4

    rv2_fit = lin_func(rv1, slope, intercept)
    res = rv2 - rv2_fit

    fig,(axm, axr) = plt.subplots(2,1, figsize=(7,10), sharex=True)
    axm.errorbar(rv1, rv2, xerr=e1, yerr=e2,
                 fmt='o', color='blue', alpha=0.7)
    if not np.isnan(slope):
        xseq= np.linspace(rv1.min(), rv1.max(),100)
        yseq= lin_func(xseq, slope, intercept)
        lbl= (f"slope={slope:.2f}±{slope_err:.2f}, "
              f"intercept={intercept:.2f}±{intercept_err:.2f}")
        axm.plot(xseq, yseq, 'r--', label=lbl)

    axm.set_ylabel("RV2 (km/s)")
    axm.set_title("RV2 vs. RV1")
    axm.legend()
    axm.grid(alpha=0.3)

    axr.errorbar(rv1, res, xerr=e1, yerr=e2,
                 fmt='o', color='blue', alpha=0.7)
    axr.axhline(0,color='r', ls='--', alpha=0.5)
    axr.set_xlabel("RV1 (km/s)")
    axr.set_ylabel("Residual (km/s)")
    axr.grid(alpha=0.3)

    plt.tight_layout()
    outpng = os.path.join(outdir, "rv2_vs_rv1.png")
    plt.savefig(outpng,dpi=300)
    plt.close()


def plot_real_calc_comparison(df_calc, df_real, outdir):
    """
    Compare final fitted RVs to 'real' or reference RVs if available
    """
    import numpy as np
    import matplotlib.pyplot as plt

    merged = pd.merge(df_calc, df_real, on="Epoch", suffixes=('_calc','_real'))
    if merged.empty:
        return

    ep = merged["Epoch"].values
    rv1c= merged["RV1_calc"].values
    rv1r= merged["RV1_real"].values
    e1 = merged.get("RV1_err", pd.Series([0]*len(merged))).values
    rv2c= merged["RV2_calc"].values
    rv2r= merged["RV2_real"].values
    e2 = merged.get("RV2_err", pd.Series([0]*len(merged))).values

    d1= rv1c - rv1r
    d2= rv2c - rv2r

    fig= plt.figure(figsize=(10,10))
    gs= fig.add_gridspec(4,1, height_ratios=[3,1,3,1])
    ax1= fig.add_subplot(gs[0])
    ax1r= fig.add_subplot(gs[1])
    ax2= fig.add_subplot(gs[2])
    ax2r= fig.add_subplot(gs[3])

    ax1.errorbar(ep, rv1c, yerr=e1, fmt='ro', ms=4, capsize=3, label='Calc RV1')
    ax1.plot(ep, rv1r, 'bo', ms=4, label='Real RV1')
    for i,epp in enumerate(ep):
        ax1.plot([epp,epp],[rv1r[i],rv1c[i]], 'k--',alpha=0.5)
    ax1.set_ylabel("RV1 (km/s)")
    ax1.set_title("Real vs Calc: RV1")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax1r.errorbar(ep, d1, yerr=e1, fmt='ko', ms=4, capsize=3)
    ax1r.axhline(0,color='r',ls='--',alpha=0.5)
    ax1r.set_ylabel("Diff (km/s)")
    ax1r.grid(alpha=0.3)

    ax2.errorbar(ep, rv2c, yerr=e2, fmt='ro', ms=4, capsize=3, label='Calc RV2')
    ax2.plot(ep, rv2r, 'bo', ms=4, label='Real RV2')
    for i,epp in enumerate(ep):
        ax2.plot([epp,epp],[rv2r[i], rv2c[i]], 'k--',alpha=0.5)
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
    outpng = os.path.join(outdir,"real_calc_comparison.png")
    plt.savefig(outpng,dpi=300)
    plt.close()