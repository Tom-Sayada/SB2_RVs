import os
import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.model_builder import compute_full_model
from src.visualization import (
    plot_epoch_lines,
    plot_rv2_vs_rv1,
    plot_real_calc_comparison
)

def report_fit_results(
    result,
    wavelengths_line, fluxes_line, uncertainties_line,
    epochs_line, central_wavelengths,
    noise_regions, windows,
    wavelengths_plot, fluxes_plot, epochs_plot,
    output_directory
):
    """
    Summarize the final fit and produce:
      - fit_results.xlsx (Per_Epoch_RVs + Chi_Square_Statistics)
      - Per-epoch line plots
      - RV2 vs RV1 plot
      - Optional real vs calc plot if star_rvs_per_epoch.csv is present
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory, exist_ok=True)

    # 1) Build the model flux on each line & epoch
    model_fluxes = compute_full_model(
        result.params,
        wavelengths_line,
        epochs_line,
        central_wavelengths
    )

    # 2) Compute line-by-line chi^2
    n_params = len(result.var_names)
    line_stats = []
    total_chi2 = 0.0
    total_points = 0

    for line_id in central_wavelengths:
        obs_flux = fluxes_line[line_id]
        mod_flux = model_fluxes[line_id]
        unc      = uncertainties_line[line_id]

        # Weighted or unweighted is already accounted for in the
        # residual construction. Typically "unc" might be the noise array.
        residuals_line = (obs_flux - mod_flux) / unc
        chi2_line = np.sum(residuals_line**2)
        n_points_line = len(obs_flux)
        dof_line = n_points_line - n_params

        line_dict = {
            'Line': line_id,
            'Chi_square': chi2_line,
            'N_points': n_points_line,
            'Chi_square_reduced': (chi2_line/dof_line) if dof_line>0 else np.nan,
            'p_value': (1 - chi2.cdf(chi2_line, dof_line)) if dof_line>0 else np.nan
        }
        line_stats.append(line_dict)

        total_chi2 += chi2_line
        total_points += n_points_line

    # 3) Summation => "GLOBAL" row
    dof_global = total_points - n_params
    if dof_global > 0:
        global_chi2_r = total_chi2 / dof_global
        global_p_value = 1 - chi2.cdf(total_chi2, dof_global)
    else:
        global_chi2_r = np.nan
        global_p_value = np.nan

    # 4) If ratio is in the params, store it
    if 'ratio' in result.params:
        ratio_val = result.params['ratio'].value
        ratio_err = result.params['ratio'].stderr or np.nan
    else:
        ratio_val = np.nan
        ratio_err = np.nan

    line_stats.append({
        'Line': 'GLOBAL',
        'Chi_square': total_chi2,
        'N_points': total_points,
        'Chi_square_reduced': global_chi2_r,
        'p_value': global_p_value,
        'ratio': ratio_val,
        'ratio_err': ratio_err
    })
    chi_square_df = pd.DataFrame(line_stats)

    # 5) Build per-epoch RVs with refined star assignment
    per_epoch_df = build_per_epoch_rvs(result)

    # 6) Write everything to fit_results.xlsx
    xlsx_path = os.path.join(output_directory, 'fit_results.xlsx')
    with pd.ExcelWriter(xlsx_path) as writer:
        per_epoch_df.to_excel(writer, sheet_name='Per_Epoch_RVs', index=False)
        chi_square_df.to_excel(writer, sheet_name='Chi_Square_Statistics', index=False)

    # 7) Do plotting
    plot_epoch_lines(
        result,
        wavelengths_line, fluxes_line, uncertainties_line,
        epochs_line, central_wavelengths,
        noise_regions, windows,
        wavelengths_plot, fluxes_plot, epochs_plot,
        output_directory
    )

    plot_rv2_vs_rv1(per_epoch_df, output_directory, real_df=None)

    # If star_rvs_per_epoch.csv is found, do real_calc_comparison
    real_csv = os.path.join(os.path.dirname(output_directory), 'star_rvs_per_epoch.csv')
    if os.path.exists(real_csv):
        real_df = pd.read_csv(real_csv)
        plot_real_calc_comparison(per_epoch_df, real_df, output_directory)

    return per_epoch_df, chi_square_df


def build_per_epoch_rvs(result):
    """
    Build a DataFrame with columns: [Epoch, RV1, RV1_uncertainty, RV2, RV2_uncertainty].
    If ratio param is present => we compute RV1 from ratio & RV2.
    Then we check each epoch to see if |RV2| > |RV1|. We count how many times
    star2 is bigger. If star2 is bigger in fewer than half the epochs, swap them.
    """
    all_params = list(result.params.keys())
    has_ratio = ('ratio' in all_params)

    # Gather epochs
    epoch_nums = set()
    for p in all_params:
        if p.startswith('rv1_epoch'):
            e = int(p.replace('rv1_epoch',''))
            epoch_nums.add(e)
        elif p.startswith('rv2_epoch'):
            e = int(p.replace('rv2_epoch',''))
            epoch_nums.add(e)

    ratio_val = 0.0
    ratio_err = 0.0
    if has_ratio:
        ratio_val = result.params['ratio'].value
        if result.params['ratio'].stderr:
            ratio_err = result.params['ratio'].stderr

    rows = []
    for e in sorted(epoch_nums):
        if has_ratio:
            rv2_val = result.params[f'rv2_epoch{e}'].value
            rv2_err = result.params[f'rv2_epoch{e}'].stderr or 0.0
            rv1_val = - ratio_val * rv2_val
            rv1_err = np.sqrt((rv2_val**2)*(ratio_err**2) + (ratio_val**2)*(rv2_err**2))
        else:
            rv1_val = result.params[f'rv1_epoch{e}'].value
            rv1_err = result.params[f'rv1_epoch{e}'].stderr or 0.0
            rv2_val = result.params[f'rv2_epoch{e}'].value
            rv2_err = result.params[f'rv2_epoch{e}'].stderr or 0.0

        rows.append({
            'Epoch': e,
            'RV1': rv1_val,
            'RV1_uncertainty': rv1_err,
            'RV2': rv2_val,
            'RV2_uncertainty': rv2_err
        })

    # Decide star assignment by majority
    # Count how many epochs have |RV2| > |RV1|
    count_star2_bigger = 0
    for r in rows:
        if abs(r['RV2']) > abs(r['RV1']):
            count_star2_bigger +=1

    # If star2 is not bigger in at least half the epochs, swap them
    n_epochs = len(rows)
    if count_star2_bigger < (n_epochs / 2):
        # swap star1 <-> star2 in all rows
        for r in rows:
            old_rv1, old_e1 = r['RV1'], r['RV1_uncertainty']
            r['RV1'] = r['RV2']
            r['RV1_uncertainty'] = r['RV2_uncertainty']
            r['RV2'] = old_rv1
            r['RV2_uncertainty'] = old_e1

    return pd.DataFrame(rows)
