"""
collate_analysis_results.py
Scans each simulation folder, reads fit_results.xlsx, extracts global stats, ratio, etc.
Builds one summary Excel with columns:
  [Main_Folder, Simulation_Number, Fit_Type, K1, K2, SNR, Q,
   Global_Chi2r, Global_PValue, Ratio, Ratio_Err,
   RV2_RV1_Slope, RV2_RV1_Slope_Err, etc.]
"""

import os
import re
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.optimize import curve_fit
from scipy.stats import chi2

def linear_func(x, a, b):
    return a*x + b

def parse_folder_name_for_params(folder_name):
    pattern = r"K_1_([0-9.]+)_K_2_([0-9.]+)_SNR_([0-9.]+)_Q_([0-9.]+)"
    m = re.search(pattern, folder_name)
    if m:
        K1  = float(m.group(1))
        K2  = float(m.group(2))
        SNR = float(m.group(3))
        Q   = float(m.group(4))
        return (K1, K2, SNR, Q)
    return (None, None, None, None)

def compute_slope(df_rvs):
    """
    Fit a line RV2 vs RV1 => slope, slope_err
    df_rvs columns: [Epoch, RV1, RV1_uncertainty, RV2, RV2_uncertainty]
    """
    rv1 = df_rvs['RV1'].values
    rv2 = df_rvs['RV2'].values
    rv1_err = df_rvs['RV1_uncertainty'].values
    rv2_err = df_rvs['RV2_uncertainty'].values

    rv1_err[rv1_err <= 0] = 1e-5
    rv2_err[rv2_err <= 0] = 1e-5

    try:
        popt, pcov = curve_fit(linear_func, rv1, rv2, sigma=rv2_err, absolute_sigma=True)
        slope, intercept = popt
        slope_err, intercept_err = np.sqrt(np.diag(pcov))
    except RuntimeError:
        slope, slope_err = np.nan, np.nan

    return slope, slope_err

def main():
    base_dir = Path(__file__).resolve().parent.parent
    data_dir = base_dir / "data"

    # Grab all main folders like K_1_XX_K_2_YY_SNR_ZZ_Q_WW
    main_folders = [
        d for d in data_dir.iterdir()
        if d.is_dir() and re.match(r"K_1_.*_K_2_.*_SNR_.*_Q_.*", d.name)
    ]

    all_rows = []

    for mf in main_folders:
        K1, K2, SNR, Q = parse_folder_name_for_params(mf.name)
        # find subfolders: simulation_1, simulation_2, ...
        sim_folders = [
            s for s in mf.iterdir()
            if s.is_dir() and s.name.startswith('simulation_')
        ]
        sim_folders.sort(key=lambda x: int(x.name.split('_')[-1]))

        for sf in sim_folders:
            sim_number_str = sf.name.split('_')[-1]
            try:
                sim_num = int(sim_number_str)
            except ValueError:
                sim_num = None

            # Look for fit_results.xlsx in standard_fit_results or ratio_constrained_results
            standard_fit = sf / "standard_fit_results" / "fit_results.xlsx"
            ratio_fit    = sf / "ratio_constrained_results" / "fit_results.xlsx"

            for fit_path, fit_type in [(standard_fit, "standard"), (ratio_fit, "ratio")]:
                if not fit_path.exists():
                    continue

                try:
                    df_chi = pd.read_excel(fit_path, sheet_name="Chi_Square_Statistics")
                    df_rvs = pd.read_excel(fit_path, sheet_name="Per_Epoch_RVs")
                except Exception as e:
                    print(f"Could not read {fit_path}: {e}")
                    continue

                # Grab the "GLOBAL" row
                df_global = df_chi[df_chi['Line'] == 'GLOBAL']
                if len(df_global) == 1:
                    row_g = df_global.iloc[0]
                    global_chi2_r = row_g['Chi_square_reduced']
                    global_p_val  = row_g['p_value']
                    ratio_val     = row_g.get('ratio', np.nan)
                    ratio_err     = row_g.get('ratio_err', np.nan)
                else:
                    global_chi2_r = np.nan
                    global_p_val  = np.nan
                    ratio_val     = np.nan
                    ratio_err     = np.nan

                # optional: slope from RV2 vs RV1
                slope, slope_err = compute_slope(df_rvs)

                # Append row
                all_rows.append({
                    'Main_Folder': mf.name,
                    'Simulation_Number': sim_num,
                    'Fit_Type': fit_type,
                    'K1': K1,
                    'K2': K2,
                    'SNR': SNR,
                    'Q': Q,
                    'Global_Chi2r': global_chi2_r,
                    'Global_PValue': global_p_val,
                    'Ratio': ratio_val,
                    'Ratio_Err': ratio_err,
                    'RV2_RV1_Slope': slope,
                    'RV2_RV1_Slope_Err': slope_err
                })

    df_all = pd.DataFrame(all_rows)
    # Reorder columns
    df_all = df_all[
      [
        'Main_Folder',
        'Simulation_Number',
        'Fit_Type',
        'K1','K2','SNR','Q',
        'Global_Chi2r','Global_PValue',
        'Ratio','Ratio_Err',
        'RV2_RV1_Slope','RV2_RV1_Slope_Err'
      ]
    ]

    out_xlsx = data_dir / "analysis_summary.xlsx"
    df_all.to_excel(out_xlsx, index=False)
    print(f"Final summary saved to {out_xlsx}")

if __name__ == "__main__":
    main()
