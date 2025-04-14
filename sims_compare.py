import os
import re
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#########################################
# PART A: Data Collection and Merging
#########################################

# Define base directories:
sb2_base = "/Users/tomsayada/spectral_analysis_project/data/for jaime/simulations_results"
main_base = "/Users/tomsayada/spectral_analysis_project/data/for jaime"
real_base = "/Users/tomsayada/spectral_analysis_project/data/K_1_100.0_K_2_200.0_SNR_50.0_Q_0.5_vsini_200.0"

def parse_epoch_str(epoch_str):
    """
    Extract a numeric epoch index from a string such as:
    'renamed_simulations/simulation_1/renamed_obs/sim_1_obs_0'
    This example extracts the number after 'obs_'.
    If no match is found, returns the original string.
    """
    match = re.search(r'obs_(\d+)$', epoch_str)
    if match:
        return int(match.group(1))
    else:
        return epoch_str

def collect_sb2_data(sb2_csv_path):
    """
    Reads the SB2 CSV file, which has columns: epoch, mean_rv, mean_rv_er, comp.
    It pivots the data so that each epoch becomes one row with separate columns for component 1 and 2.
    Also creates an 'epoch_index' column (as string) from the epoch text.
    """
    df = pd.read_csv(sb2_csv_path)
    df_pivot = df.pivot(index='epoch', columns='comp', values=['mean_rv', 'mean_rv_er'])
    # Flatten multi-level columns
    df_pivot.columns = [f"SB2_{col[0]}_{int(col[1])}" for col in df_pivot.columns]
    # Rename for clarity:
    rename_map = {
        'SB2_mean_rv_1': 'SB2_rv_1',
        'SB2_mean_rv_er_1': 'SB2_rv_1_er',
        'SB2_mean_rv_2': 'SB2_rv_2',
        'SB2_mean_rv_er_2': 'SB2_rv_2_er'
    }
    df_pivot.rename(columns=rename_map, inplace=True)
    df_pivot['epoch_str_sb2'] = df_pivot.index
    df_pivot['epoch_index'] = df_pivot['epoch_str_sb2'].apply(parse_epoch_str).astype(str)
    return df_pivot.reset_index(drop=True)

def collect_excel_data(xlsx_path, label):
    """
    Reads a fit_results.xlsx file from the given folder.
    Assumes a worksheet named 'Per_Epoch_RVs' with columns:
      Epoch, RV1, RV1_err, RV2, RV2_err.
    Renames the columns to include the provided label (e.g., 'voigt', 'gauss', 'grid')
    and creates an 'epoch_index' column.
    """
    df = pd.read_excel(xlsx_path, sheet_name='Per_Epoch_RVs')
    df[f'epoch_str_{label}'] = df['Epoch'].astype(str)
    df['epoch_index'] = df[f'epoch_str_{label}'].apply(parse_epoch_str).astype(str)
    df.rename(columns={
        'RV1': f'{label}_rv_1',
        'RV1_err': f'{label}_rv_1_err',
        'RV2': f'{label}_rv_2',
        'RV2_err': f'{label}_rv_2_err'
    }, inplace=True)
    cols = [f'epoch_str_{label}', 'epoch_index', f'{label}_rv_1', f'{label}_rv_1_err', f'{label}_rv_2', f'{label}_rv_2_err']
    return df[cols]

def collect_real_data(real_csv_path):
    """
    Reads the real RV CSV file, which is assumed to have columns:
      Epoch, RV1, RV2.
    It renames the RV columns to 'real_rv_1' and 'real_rv_2' and creates an 'epoch_index' column.
    """
    df = pd.read_csv(real_csv_path)
    df['epoch_str_real'] = df['Epoch'].astype(str)
    df['epoch_index'] = df['epoch_str_real'].apply(parse_epoch_str).astype(str)
    df.rename(columns={'RV1': 'real_rv_1', 'RV2': 'real_rv_2'}, inplace=True)
    cols = ['epoch_str_real', 'epoch_index', 'real_rv_1', 'real_rv_2']
    return df[cols]

num_simulations = 50
all_sim_results = []

for sim_num in range(1, num_simulations + 1):
    dfs = []  # To collect DataFrames from different methods
    # 1) SB2
    sb2_csv_path = os.path.join(sb2_base, f"simulation_{sim_num}", "SB2", "fit_values.csv")
    if os.path.isfile(sb2_csv_path):
        try:
            df_sb2 = collect_sb2_data(sb2_csv_path)
            dfs.append(df_sb2)
        except Exception as e:
            print(f"[ERROR] Reading SB2 for simulation {sim_num}: {e}")
    else:
        print(f"[WARNING] Missing SB2 file for simulation {sim_num}: {sb2_csv_path}")

    # 2) Voigt
    voigt_path = os.path.join(main_base, f"simulation_{sim_num}", "standard_sym_voigt_weighted_nobaseline_fit_results", "fit_results.xlsx")
    if os.path.isfile(voigt_path):
        try:
            df_voigt = collect_excel_data(voigt_path, label='voigt')
            dfs.append(df_voigt)
        except Exception as e:
            print(f"[ERROR] Reading Voigt for simulation {sim_num}: {e}")
    else:
        print(f"[WARNING] Missing Voigt Excel for simulation {sim_num}: {voigt_path}")

    # 3) Gaussian
    gauss_path = os.path.join(main_base, f"simulation_{sim_num}", "standard_sym_gaussian_weighted_nobaseline_auto_fit_results", "fit_results.xlsx")
    if os.path.isfile(gauss_path):
        try:
            df_gauss = collect_excel_data(gauss_path, label='gauss')
            dfs.append(df_gauss)
        except Exception as e:
            print(f"[ERROR] Reading Gaussian for simulation {sim_num}: {e}")
    else:
        print(f"[WARNING] Missing Gaussian Excel for simulation {sim_num}: {gauss_path}")

    # 4) Grid refined data
    grid_path = os.path.join(main_base, f"simulation_{sim_num}", "grid_refined_fit_results_weighted", "fit_results.xlsx")
    if os.path.isfile(grid_path):
        try:
            df_grid = collect_excel_data(grid_path, label='grid')
            dfs.append(df_grid)
        except Exception as e:
            print(f"[ERROR] Reading Grid data for simulation {sim_num}: {e}")
    else:
        print(f"[WARNING] Missing Grid Excel for simulation {sim_num}: {grid_path}")

    # 5) Real RV data
    real_csv_path = os.path.join(real_base, f"simulation_{sim_num}", "star_rvs_per_epoch.csv")
    if os.path.isfile(real_csv_path):
        try:
            df_real = collect_real_data(real_csv_path)
            dfs.append(df_real)
        except Exception as e:
            print(f"[ERROR] Reading Real RV data for simulation {sim_num}: {e}")
    else:
        print(f"[WARNING] Missing Real RV file for simulation {sim_num}: {real_csv_path}")

    # If at least one dataset exists, merge on 'epoch_index'
    if not dfs:
        print(f"[WARNING] No data found for simulation {sim_num}.")
        continue

    df_merge = dfs[0]
    for df_ in dfs[1:]:
        df_merge = pd.merge(df_merge, df_, on='epoch_index', how='outer')
    df_merge['simulation'] = sim_num
    all_sim_results.append(df_merge)

if not all_sim_results:
    print("No simulation data was collected. Please check file paths and folder structure.")
    exit()

final_df = pd.concat(all_sim_results, ignore_index=True)
final_df.to_excel("all_comparisons.xlsx", index=False)
print("Collected and saved merged data to 'all_comparisons.xlsx'.")

#########################################
# PART B: Data Reassignment and Plotting
#########################################

# Set the output folder for plots.
plots_folder = "/Users/tomsayada/spectral_analysis_project/src/Simulations_Comparison_Plots"
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)

# -----------------------------
# PART B1: Data Reassignment
# -----------------------------
df = pd.read_excel("all_comparisons.xlsx")

def swap_if_needed(row, col1, col2):
    if pd.notna(row[col1]) and pd.notna(row[col2]):
        if abs(row[col1]) > abs(row[col2]):
            row[col1], row[col2] = row[col2], row[col1]
    return row

# Methods to reassign; applies to SB2, voigt, gauss, grid, and real.
methods = ["SB2", "voigt", "gauss", "grid", "real"]

for method in methods:
    col1 = f"{method}_rv_1"
    col2 = f"{method}_rv_2"
    if col1 in df.columns and col2 in df.columns:
        df = df.apply(lambda row: swap_if_needed(row, col1, col2), axis=1)

for method in methods:
    col1 = f"{method}_rv_1"
    col2 = f"{method}_rv_2"
    if col1 in df.columns and col2 in df.columns:
        condition = df.apply(lambda row: pd.isna(row[col1]) or pd.isna(row[col2]) or (abs(row[col1]) <= abs(row[col2])), axis=1)
        if not condition.all():
            print(f"Warning: Condition not met for '{method}' in some rows.")
        else:
            print(f"All rows for '{method}' satisfy |{col1}| <= |{col2}|.")
new_excel_file = "all_comparisons_reassigned.xlsx"
df.to_excel(new_excel_file, index=False)
print(f"Saved reassigned data to '{new_excel_file}'.")

# -----------------------------
# PART B2: Generate Per-Simulation Plots
# -----------------------------
df_new = pd.read_excel(new_excel_file)
df_new['simulation'] = pd.to_numeric(df_new['simulation'], errors='coerce')
simulations = sorted(df_new['simulation'].dropna().unique())

def plot_identity_line(ax, x_min, x_max):
    x_line = np.linspace(x_min, x_max, 100)
    ax.plot(x_line, x_line, 'k--', label="Identity")

for sim in simulations:
    df_sim = df_new[df_new['simulation'] == sim]

    # SB2: RV₁ vs RV₂ with trend line.
    if 'SB2_rv_1' in df_sim.columns and 'SB2_rv_2' in df_sim.columns:
        df_plot = df_sim[['SB2_rv_1', 'SB2_rv_2']].dropna()
        if not df_plot.empty:
            plt.figure(figsize=(8,6))
            plt.scatter(df_plot['SB2_rv_1'], df_plot['SB2_rv_2'], alpha=0.7, label="Data")
            x_data = df_plot['SB2_rv_1'].values
            y_data = df_plot['SB2_rv_2'].values
            if len(x_data) >= 2:
                p, cov = np.polyfit(x_data, y_data, 1, cov=True)
                slope, intercept = p
                slope_err, intercept_err = np.sqrt(np.diag(cov))
                x_line = np.linspace(np.min(x_data), np.max(x_data), 100)
                plt.plot(x_line, slope*x_line+intercept, 'r--', label="Trend")
                plt.text(np.min(x_data), np.max(y_data),
                         f"y = {slope:.2f}x + {intercept:.2f}\n"
                         f"m = {slope:.2f} ± {slope_err:.2f}\n"
                         f"b = {intercept:.2f} ± {intercept_err:.2f}",
                         color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
            plt.xlabel("SB2 RV₁")
            plt.ylabel("SB2 RV₂")
            plt.title(f"Simulation {int(sim)}: SB2 RV₁ vs RV₂")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(plots_folder, f"sim_{int(sim)}_SB2_RV1_vs_RV2.png"))
            plt.close()

    # Voigt: RV₁ vs RV₂.
    if 'voigt_rv_1' in df_sim.columns and 'voigt_rv_2' in df_sim.columns:
        df_plot = df_sim[['voigt_rv_1', 'voigt_rv_2']].dropna()
        if not df_plot.empty:
            plt.figure(figsize=(8,6))
            plt.scatter(df_plot['voigt_rv_1'], df_plot['voigt_rv_2'], alpha=0.7, label="Data")
            x_data = df_plot['voigt_rv_1'].values
            y_data = df_plot['voigt_rv_2'].values
            if len(x_data) >= 2:
                p, cov = np.polyfit(x_data, y_data, 1, cov=True)
                slope, intercept = p
                slope_err, intercept_err = np.sqrt(np.diag(cov))
                x_line = np.linspace(np.min(x_data), np.max(x_data), 100)
                plt.plot(x_line, slope*x_line+intercept, 'r--', label="Trend")
                plt.text(np.min(x_data), np.max(y_data),
                         f"y = {slope:.2f}x + {intercept:.2f}\n"
                         f"m = {slope:.2f} ± {slope_err:.2f}\n"
                         f"b = {intercept:.2f} ± {intercept_err:.2f}",
                         color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
            plt.xlabel("Voigt RV₁")
            plt.ylabel("Voigt RV₂")
            plt.title(f"Simulation {int(sim)}: Voigt RV₁ vs RV₂")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(plots_folder, f"sim_{int(sim)}_Voigt_RV1_vs_RV2.png"))
            plt.close()

    # Gaussian: RV₁ vs RV₂.
    if 'gauss_rv_1' in df_sim.columns and 'gauss_rv_2' in df_sim.columns:
        df_plot = df_sim[['gauss_rv_1', 'gauss_rv_2']].dropna()
        if not df_plot.empty:
            plt.figure(figsize=(8,6))
            plt.scatter(df_plot['gauss_rv_1'], df_plot['gauss_rv_2'], alpha=0.7, label="Data")
            x_data = df_plot['gauss_rv_1'].values
            y_data = df_plot['gauss_rv_2'].values
            if len(x_data) >= 2:
                p, cov = np.polyfit(x_data, y_data, 1, cov=True)
                slope, intercept = p
                slope_err, intercept_err = np.sqrt(np.diag(cov))
                x_line = np.linspace(np.min(x_data), np.max(x_data), 100)
                plt.plot(x_line, slope*x_line+intercept, 'r--', label="Trend")
                plt.text(np.min(x_data), np.max(y_data),
                         f"y = {slope:.2f}x + {intercept:.2f}\n"
                         f"m = {slope:.2f} ± {slope_err:.2f}\n"
                         f"b = {intercept:.2f} ± {intercept_err:.2f}",
                         color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
            plt.xlabel("Gaussian RV₁")
            plt.ylabel("Gaussian RV₂")
            plt.title(f"Simulation {int(sim)}: Gaussian RV₁ vs RV₂")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(plots_folder, f"sim_{int(sim)}_Gaussian_RV1_vs_RV2.png"))
            plt.close()

    # Grid: RV₁ vs RV₂.
    if 'grid_rv_1' in df_sim.columns and 'grid_rv_2' in df_sim.columns:
        df_plot = df_sim[['grid_rv_1', 'grid_rv_2']].dropna()
        if not df_plot.empty:
            plt.figure(figsize=(8,6))
            plt.scatter(df_plot['grid_rv_1'], df_plot['grid_rv_2'], alpha=0.7, label="Data")
            x_data = df_plot['grid_rv_1'].values
            y_data = df_plot['grid_rv_2'].values
            if len(x_data) >= 2:
                p, cov = np.polyfit(x_data, y_data, 1, cov=True)
                slope, intercept = p
                slope_err, intercept_err = np.sqrt(np.diag(cov))
                x_line = np.linspace(np.min(x_data), np.max(x_data), 100)
                plt.plot(x_line, slope*x_line+intercept, 'r--', label="Trend")
                plt.text(np.min(x_data), np.max(y_data),
                         f"y = {slope:.2f}x + {intercept:.2f}\n"
                         f"m = {slope:.2f} ± {slope_err:.2f}\n"
                         f"b = {intercept:.2f} ± {intercept_err:.2f}",
                         color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
            plt.xlabel("Grid RV₁")
            plt.ylabel("Grid RV₂")
            plt.title(f"Simulation {int(sim)}: Grid RV₁ vs RV₂")
            plt.grid(True)
            plt.legend()
            plt.savefig(os.path.join(plots_folder, f"sim_{int(sim)}_Grid_RV1_vs_RV2.png"))
            plt.close()

    # Real vs Measured RV₁.
    if 'real_rv_1' in df_sim.columns:
        plt.figure(figsize=(8,6))
        meas_rv1 = []
        labels = []
        for method, col in zip(["SB2", "Voigt", "Gaussian", "Grid"],
                               ["SB2_rv_1", "voigt_rv_1", "gauss_rv_1", "grid_rv_1"]):
            if col in df_sim.columns:
                meas_rv1.append(col)
                labels.append(method)
        if meas_rv1:
            all_vals = df_sim[['real_rv_1'] + meas_rv1].values.flatten()
            all_vals = all_vals[~np.isnan(all_vals)]
            if all_vals.size:
                x_min, x_max = np.min(all_vals), np.max(all_vals)
                ax = plt.gca()
                plot_identity_line(ax, x_min, x_max)
            for method, col in zip(labels, meas_rv1):
                temp = df_sim[['real_rv_1', col]].dropna()
                if not temp.empty:
                    plt.scatter(temp['real_rv_1'], temp[col], alpha=0.7, label=method)
            plt.xlabel("Real RV₁")
            plt.ylabel("Measured RV₁")
            plt.title(f"Simulation {int(sim)}: Real RV₁ vs Measured RV₁")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_folder, f"sim_{int(sim)}_RV1_Real_vs_Measured.png"))
            plt.close()

    # Real vs Measured RV₂.
    if 'real_rv_2' in df_sim.columns:
        plt.figure(figsize=(8,6))
        meas_rv2 = []
        labels = []
        for method, col in zip(["SB2", "Voigt", "Gaussian", "Grid"],
                               ["SB2_rv_2", "voigt_rv_2", "gauss_rv_2", "grid_rv_2"]):
            if col in df_sim.columns:
                meas_rv2.append(col)
                labels.append(method)
        if meas_rv2:
            all_vals = df_sim[['real_rv_2'] + meas_rv2].values.flatten()
            all_vals = all_vals[~np.isnan(all_vals)]
            if all_vals.size:
                x_min, x_max = np.min(all_vals), np.max(all_vals)
                ax = plt.gca()
                plot_identity_line(ax, x_min, x_max)
            for method, col in zip(labels, meas_rv2):
                temp = df_sim[['real_rv_2', col]].dropna()
                if not temp.empty:
                    plt.scatter(temp['real_rv_2'], temp[col], alpha=0.7, label=method)
            plt.xlabel("Real RV₂")
            plt.ylabel("Measured RV₂")
            plt.title(f"Simulation {int(sim)}: Real RV₂ vs Measured RV₂")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_folder, f"sim_{int(sim)}_RV2_Real_vs_Measured.png"))
            plt.close()

    # Additional pairwise comparisons for RV₁.
    if 'SB2_rv_1' in df_sim.columns and 'voigt_rv_1' in df_sim.columns:
        plt.figure(figsize=(8,6))
        df_plot = df_sim[['SB2_rv_1', 'voigt_rv_1']].dropna()
        if not df_plot.empty:
            plt.scatter(df_plot['SB2_rv_1'], df_plot['voigt_rv_1'], alpha=0.7)
            x_data = df_plot['SB2_rv_1'].values
            y_data = df_plot['voigt_rv_1'].values
            if len(x_data) >= 2:
                p, cov = np.polyfit(x_data, y_data, 1, cov=True)
                slope, intercept = p
                slope_err, intercept_err = np.sqrt(np.diag(cov))
                x_line = np.linspace(np.min(x_data), np.max(x_data), 100)
                plt.plot(x_line, slope*x_line+intercept, 'r--', label="Trend line")
                plt.text(np.min(x_data), np.max(y_data),
                         f"y = {slope:.2f}x + {intercept:.2f}\n"
                         f"m = {slope:.2f} ± {slope_err:.2f}\n"
                         f"b = {intercept:.2f} ± {intercept_err:.2f}",
                         color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
            plt.xlabel("SB2 RV₁")
            plt.ylabel("Voigt RV₁")
            plt.title(f"Simulation {int(sim)}: SB2 RV₁ vs Voigt RV₁")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_folder, f"sim_{int(sim)}_SB2_vs_Voigt_RV1.png"))
            plt.close()

    if 'SB2_rv_1' in df_sim.columns and 'gauss_rv_1' in df_sim.columns:
        plt.figure(figsize=(8,6))
        df_plot = df_sim[['SB2_rv_1', 'gauss_rv_1']].dropna()
        if not df_plot.empty:
            plt.scatter(df_plot['SB2_rv_1'], df_plot['gauss_rv_1'], alpha=0.7)
            x_data = df_plot['SB2_rv_1'].values
            y_data = df_plot['gauss_rv_1'].values
            if len(x_data) >= 2:
                p, cov = np.polyfit(x_data, y_data, 1, cov=True)
                slope, intercept = p
                slope_err, intercept_err = np.sqrt(np.diag(cov))
                x_line = np.linspace(np.min(x_data), np.max(x_data), 100)
                plt.plot(x_line, slope*x_line+intercept, 'r--', label="Trend line")
                plt.text(np.min(x_data), np.max(y_data),
                         f"y = {slope:.2f}x + {intercept:.2f}\n"
                         f"m = {slope:.2f} ± {slope_err:.2f}\n"
                         f"b = {intercept:.2f} ± {intercept_err:.2f}",
                         color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
            plt.xlabel("SB2 RV₁")
            plt.ylabel("Gaussian RV₁")
            plt.title(f"Simulation {int(sim)}: SB2 RV₁ vs Gaussian RV₁")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_folder, f"sim_{int(sim)}_SB2_vs_Gaussian_RV1.png"))
            plt.close()

    if 'SB2_rv_1' in df_sim.columns and 'grid_rv_1' in df_sim.columns:
        plt.figure(figsize=(8,6))
        df_plot = df_sim[['SB2_rv_1', 'grid_rv_1']].dropna()
        if not df_plot.empty:
            plt.scatter(df_plot['SB2_rv_1'], df_plot['grid_rv_1'], alpha=0.7)
            x_data = df_plot['SB2_rv_1'].values
            y_data = df_plot['grid_rv_1'].values
            if len(x_data) >= 2:
                p, cov = np.polyfit(x_data, y_data, 1, cov=True)
                slope, intercept = p
                slope_err, intercept_err = np.sqrt(np.diag(cov))
                x_line = np.linspace(np.min(x_data), np.max(x_data), 100)
                plt.plot(x_line, slope*x_line+intercept, 'r--', label="Trend line")
                plt.text(np.min(x_data), np.max(y_data),
                         f"y = {slope:.2f}x + {intercept:.2f}\n"
                         f"m = {slope:.2f} ± {slope_err:.2f}\n"
                         f"b = {intercept:.2f} ± {intercept_err:.2f}",
                         color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
            plt.xlabel("SB2 RV₁")
            plt.ylabel("Grid RV₁")
            plt.title(f"Simulation {int(sim)}: SB2 RV₁ vs Grid RV₁")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_folder, f"sim_{int(sim)}_SB2_vs_Grid_RV1.png"))
            plt.close()

print("Per-simulation comparison diagrams generated and saved in:")
print(plots_folder)

# -----------------------------
# PART B3: Overall (Global) Checks with Uniform Bin Widths
# -----------------------------
# For RV₁ residuals:
methods_for_hist = [('SB2', 'SB2_rv_1'), ('Voigt', 'voigt_rv_1'), ('Gaussian', 'gauss_rv_1'), ('Grid', 'grid_rv_1')]
all_rv1_residuals = []
for label, col in methods_for_hist:
    if col in df_new.columns and 'real_rv_1' in df_new.columns:
        temp = df_new[['real_rv_1', col]].dropna()
        if not temp.empty:
            residual = temp[col] - temp['real_rv_1']
            all_rv1_residuals.append(residual.values)
all_rv1_residuals = np.concatenate(all_rv1_residuals) if len(all_rv1_residuals) > 0 else np.array([])
if all_rv1_residuals.size > 0:
    min_val = np.min(all_rv1_residuals)
    max_val = np.max(all_rv1_residuals)
    bins_rv1 = np.linspace(min_val, max_val, 31)  # 30 bins
else:
    bins_rv1 = 30

plt.figure(figsize=(8,6))
for label, col in methods_for_hist:
    if col in df_new.columns and 'real_rv_1' in df_new.columns:
        temp = df_new[['real_rv_1', col]].dropna()
        if not temp.empty:
            residual = temp[col] - temp['real_rv_1']
            plt.hist(residual, bins=bins_rv1, alpha=0.5, label=f"{label} Residuals")
plt.xlabel("Residual (Measured RV₁ - Real RV₁)")
plt.ylabel("Frequency")
plt.title("Overall Histogram of RV₁ Residuals (All Simulations)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_folder, "Overall_RV1_Residuals_Histogram.png"))
plt.close()

# For RV₂ residuals:
methods_for_hist_rv2 = [('SB2', 'SB2_rv_2'), ('Voigt', 'voigt_rv_2'), ('Gaussian', 'gauss_rv_2'), ('Grid', 'grid_rv_2')]
all_rv2_residuals = []
for label, col in methods_for_hist_rv2:
    if col in df_new.columns and 'real_rv_2' in df_new.columns:
        temp = df_new[['real_rv_2', col]].dropna()
        if not temp.empty:
            residual = temp[col] - temp['real_rv_2']
            all_rv2_residuals.append(residual.values)
all_rv2_residuals = np.concatenate(all_rv2_residuals) if len(all_rv2_residuals) > 0 else np.array([])
if all_rv2_residuals.size > 0:
    min_val = np.min(all_rv2_residuals)
    max_val = np.max(all_rv2_residuals)
    bins_rv2 = np.linspace(min_val, max_val, 31)  # 30 bins
else:
    bins_rv2 = 30

plt.figure(figsize=(8,6))
for label, col in methods_for_hist_rv2:
    if col in df_new.columns and 'real_rv_2' in df_new.columns:
        temp = df_new[['real_rv_2', col]].dropna()
        if not temp.empty:
            residual = temp[col] - temp['real_rv_2']
            plt.hist(residual, bins=bins_rv2, alpha=0.5, label=f"{label} Residuals")
plt.xlabel("Residual (Measured RV₂ - Real RV₂)")
plt.ylabel("Frequency")
plt.title("Overall Histogram of RV₂ Residuals (All Simulations)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_folder, "Overall_RV2_Residuals_Histogram.png"))
plt.close()

print("Overall (global) comparison diagrams generated and saved in:")
print(plots_folder)

# -----------------------------
# PART B4: Create Combined Figures of Real vs Measured Plots (Split into Two)
# -----------------------------
def create_combined_figure(file_list, title, out_filename):
    n_plots = len(file_list)
    ncols = 5
    nrows = math.ceil(n_plots / ncols)
    plt.figure(figsize=(ncols * 3, nrows * 3))
    for i, file in enumerate(file_list):
        img = plt.imread(file)
        ax = plt.subplot(nrows, ncols, i + 1)
        ax.imshow(img)
        sim_label = os.path.basename(file).split("_")[1]
        ax.set_title(f"Sim {sim_label}")
        ax.axis("off")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(plots_folder, out_filename))
    plt.close()

rv1_files = []
for sim in range(1, 51):
    file_path = os.path.join(plots_folder, f"sim_{sim}_RV1_Real_vs_Measured.png")
    if os.path.exists(file_path):
        rv1_files.append(file_path)
if rv1_files:
    split_index = math.ceil(len(rv1_files) / 2)
    rv1_files_part1 = rv1_files[:split_index]
    rv1_files_part2 = rv1_files[split_index:]
    create_combined_figure(rv1_files_part1,
                           "Combined RV₁: Real vs Measured (Part 1)",
                           "Combined_RV1_Real_vs_Measured_Part1.png")
    create_combined_figure(rv1_files_part2,
                           "Combined RV₁: Real vs Measured (Part 2)",
                           "Combined_RV1_Real_vs_Measured_Part2.png")

rv2_files = []
for sim in range(1, 51):
    file_path = os.path.join(plots_folder, f"sim_{sim}_RV2_Real_vs_Measured.png")
    if os.path.exists(file_path):
        rv2_files.append(file_path)
if rv2_files:
    split_index = math.ceil(len(rv2_files) / 2)
    rv2_files_part1 = rv2_files[:split_index]
    rv2_files_part2 = rv2_files[split_index:]
    create_combined_figure(rv2_files_part1,
                           "Combined RV₂: Real vs Measured (Part 1)",
                           "Combined_RV2_Real_vs_Measured_Part1.png")
    create_combined_figure(rv2_files_part2,
                           "Combined RV₂: Real vs Measured (Part 2)",
                           "Combined_RV2_Real_vs_Measured_Part2.png")

print("Combined figures (split) created and saved in:")
print(plots_folder)

# -----------------------------
# PART B5: Single Combined Scatter Plots Using the DataFrame (All Simulations) for Real vs Measured
# -----------------------------
df_all = pd.read_excel(new_excel_file)

# Combined scatter plot for RV₁.
methods_for_rv1 = [("SB2", "SB2_rv_1"), ("Voigt", "voigt_rv_1"), ("Gaussian", "gauss_rv_1"), ("Grid", "grid_rv_1")]
df_rv1 = df_all.dropna(subset=["real_rv_1"])
plt.figure(figsize=(8,6))
cols_rv1 = ["real_rv_1"] + [col for label, col in methods_for_rv1]
all_vals = df_rv1[cols_rv1].values.flatten()
all_vals = all_vals[~np.isnan(all_vals)]
x_min, x_max = np.min(all_vals), np.max(all_vals)
plt.plot(np.linspace(x_min, x_max, 100), np.linspace(x_min, x_max, 100), 'k--', label="Identity")
for label, col in methods_for_rv1:
    temp = df_rv1.dropna(subset=[col])
    plt.scatter(temp["real_rv_1"], temp[col], alpha=0.3, label=label)
plt.xlabel("Real RV₁")
plt.ylabel("Measured RV₁")
plt.title("All Simulations: Real RV₁ vs Measured RV₁")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_folder, "AllSim_RV1_Real_vs_Measured.png"))
plt.close()

# Combined scatter plot for RV₂.
methods_for_rv2 = [("SB2", "SB2_rv_2"), ("Voigt", "voigt_rv_2"), ("Gaussian", "gauss_rv_2"), ("Grid", "grid_rv_2")]
df_rv2 = df_all.dropna(subset=["real_rv_2"])
plt.figure(figsize=(8,6))
cols_rv2 = ["real_rv_2"] + [col for label, col in methods_for_rv2]
all_vals = df_rv2[cols_rv2].values.flatten()
all_vals = all_vals[~np.isnan(all_vals)]
x_min, x_max = np.min(all_vals), np.max(all_vals)
plt.plot(np.linspace(x_min, x_max, 100), np.linspace(x_min, x_max, 100), 'k--', label="Identity")
for label, col in methods_for_rv2:
    temp = df_rv2.dropna(subset=[col])
    plt.scatter(temp["real_rv_2"], temp[col], alpha=0.3, label=label)
plt.xlabel("Real RV₂")
plt.ylabel("Measured RV₂")
plt.title("All Simulations: Real RV₂ vs Measured RV₂")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(plots_folder, "AllSim_RV2_Real_vs_Measured.png"))
plt.close()

print("Single combined scatter plots (all simulations) for Real vs Measured have been created and saved in:")
print(plots_folder)

# -----------------------------
# PART B6: Global Plot of RV₁ vs RV₂ for Each Method (All Simulations)
# -----------------------------
# Here we plot, for each method (SB2, voigt, gauss, grid), a global scatter plot of method RV₁ vs method RV₂
# with a linear regression fit and equation annotation.
methods_for_global = ["SB2", "voigt", "gauss", "grid"]
for method in methods_for_global:
    rv1_col = f"{method}_rv_1"
    rv2_col = f"{method}_rv_2"
    if rv1_col in df_all.columns and rv2_col in df_all.columns:
        df_method = df_all.dropna(subset=[rv1_col, rv2_col])
        if not df_method.empty:
            plt.figure(figsize=(8,6))
            plt.scatter(df_method[rv1_col], df_method[rv2_col], alpha=0.5, label=f"{method} data")
            x_data = df_method[rv1_col].values
            y_data = df_method[rv2_col].values
            if len(x_data) >= 2:
                p, cov = np.polyfit(x_data, y_data, 1, cov=True)
                slope, intercept = p
                slope_err, intercept_err = np.sqrt(np.diag(cov))
                x_line = np.linspace(np.min(x_data), np.max(x_data), 100)
                plt.plot(x_line, slope*x_line+intercept, 'r--', label="Trend line")
                x_text = np.min(x_data)
                y_text = np.max(y_data)
                eq_str = (f"y = {slope:.2f}x + {intercept:.2f}\n"
                          f"m = {slope:.2f} ± {slope_err:.2f}\n"
                          f"b = {intercept:.2f} ± {intercept_err:.2f}")
                plt.text(x_text, y_text, eq_str, color='red', fontsize=10,
                         bbox=dict(facecolor='white', alpha=0.5))
            plt.xlabel(f"{method} RV₁")
            plt.ylabel(f"{method} RV₂")
            plt.title(f"All Simulations: {method} RV₁ vs {method} RV₂")
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(plots_folder, f"AllSim_{method}_RV1_vs_RV2.png"))
            plt.close()

print("Global (all simulations) plots of RV₁ vs RV₂ for each method have been created and saved in:")
print(plots_folder)
