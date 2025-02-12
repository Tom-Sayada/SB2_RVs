#!/usr/bin/env python3
"""
run_stats_summary.py

Example post-processing script that:
  - Iterates over simulation_* subfolders in the parent data directory
  - Reads final results from standard & ratio fit => "fit_results.xlsx"
  - Compares recovered RV1, RV2 to known K1=50, K2=100 in the simulations
  - Gathers the average offsets for each simulation & approach
  - Produces histograms + scatter plots of the offsets
  - Saves a summary CSV + the plots

USAGE:
  python run_stats_summary.py \
    --data_dir /path/to/parent_folder  \
    --standard_sub "standard_sym_weighted_nobaseline_fit_results" \
    --ratio_sub "ratio_sym_weighted_nobaseline_fit_results" \
    --k1_true 50 --k2_true 100 \
    --outfile "summary_stats.csv"

Then, check the "plots/" subdir for generated histogram/scatter plots.
"""

import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def gather_results(sim_folder, standard_subdir, ratio_subdir,
                   true_k1=50.0, true_k2=100.0):
    """
    Attempt to read 'fit_results.xlsx' from both standard and ratio subfolders,
    compute average offsets for (RV1, RV2), plus store each epoch's RV offset for histogramming.
    Returns a dict:
      {
        'standard': {
          'mean_dk1': float, 'std_dk1': float, 'mean_dk2': float, 'std_dk2': float,
          'dk1_all': array of per-epoch offsets, 'dk2_all': array of per-epoch offsets,
          'chi2red': float (or np.nan if not found),
        },
        'ratio': {...same structure...}
      }
    """
    output = {
        'standard': {
            'mean_dk1': np.nan,
            'std_dk1':  np.nan,
            'mean_dk2': np.nan,
            'std_dk2':  np.nan,
            'dk1_all':  [],
            'dk2_all':  [],
            'chi2red':  np.nan
        },
        'ratio': {
            'mean_dk1': np.nan,
            'std_dk1':  np.nan,
            'mean_dk2': np.nan,
            'std_dk2':  np.nan,
            'dk1_all':  [],
            'dk2_all':  [],
            'chi2red':  np.nan
        }
    }

    # 1) standard
    st_xlsx = os.path.join(sim_folder, standard_subdir, "fit_results.xlsx")
    if os.path.isfile(st_xlsx):
        try:
            st_df_rv = pd.read_excel(st_xlsx, sheet_name="Per_Epoch_RVs")
            st_df_chi= pd.read_excel(st_xlsx, sheet_name="Chi_Square_Statistics")

            # collect all epoch-level offsets
            rv1_vals = st_df_rv["RV1"].values
            rv2_vals = st_df_rv["RV2"].values
            dk1 = rv1_vals - true_k1
            dk2 = rv2_vals - true_k2

            output['standard']['mean_dk1'] = np.mean(dk1)
            output['standard']['std_dk1']  = np.std(dk1)
            output['standard']['mean_dk2'] = np.mean(dk2)
            output['standard']['std_dk2']  = np.std(dk2)

            output['standard']['dk1_all']  = dk1
            output['standard']['dk2_all']  = dk2

            row_global = st_df_chi[ st_df_chi["Line_ID"]=="GLOBAL" ]
            if not row_global.empty:
                c2red = row_global["Chi_Square_reduced"].values[0]
                output['standard']['chi2red'] = c2red

        except Exception as e:
            print(f"[Warning] Could not parse standard results from {st_xlsx}: {e}")

    # 2) ratio
    rt_xlsx = os.path.join(sim_folder, ratio_subdir, "fit_results.xlsx")
    if os.path.isfile(rt_xlsx):
        try:
            rt_df_rv = pd.read_excel(rt_xlsx, sheet_name="Per_Epoch_RVs")
            rt_df_chi= pd.read_excel(rt_xlsx, sheet_name="Chi_Square_Statistics")

            rv1_vals = rt_df_rv["RV1"].values
            rv2_vals = rt_df_rv["RV2"].values
            dk1 = rv1_vals - true_k1
            dk2 = rv2_vals - true_k2

            output['ratio']['mean_dk1'] = np.mean(dk1)
            output['ratio']['std_dk1']  = np.std(dk1)
            output['ratio']['mean_dk2'] = np.mean(dk2)
            output['ratio']['std_dk2']  = np.std(dk2)

            output['ratio']['dk1_all']  = dk1
            output['ratio']['dk2_all']  = dk2

            row_global = rt_df_chi[ rt_df_chi["Line_ID"]=="GLOBAL" ]
            if not row_global.empty:
                c2red = row_global["Chi_Square_reduced"].values[0]
                output['ratio']['chi2red'] = c2red

        except Exception as e:
            print(f"[Warning] Could not parse ratio results from {rt_xlsx}: {e}")

    return output


def main():
    parser = argparse.ArgumentParser(
        description="Collect final K1,K2 offsets from simulation subfolders and produce histograms."
    )
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Parent directory containing simulation_* subfolders.")
    parser.add_argument("--standard_sub", type=str,
                        default="standard_sym_weighted_nobaseline_fit_results",
                        help="Name of standard-fit subfolder inside each sim folder.")
    parser.add_argument("--ratio_sub", type=str,
                        default="ratio_sym_weighted_nobaseline_fit_results",
                        help="Name of ratio-fit subfolder inside each sim folder.")
    parser.add_argument("--k1_true", type=float, default=50.0,
                        help="True K1 used in the simulations")
    parser.add_argument("--k2_true", type=float, default=100.0,
                        help="True K2 used in the simulations")
    parser.add_argument("--outfile", type=str, default="summary_stats.csv",
                        help="Where to save aggregated summary table.")
    args = parser.parse_args()

    data_dir = args.data_dir
    st_sub   = args.standard_sub
    rt_sub   = args.ratio_sub
    k1_true  = args.k1_true
    k2_true  = args.k2_true
    outcsv   = args.outfile

    # discover all simulation_* subfolders
    sim_folders = []
    for entry in os.listdir(data_dir):
        if entry.startswith("simulation_"):
            fullp = os.path.join(data_dir, entry)
            if os.path.isdir(fullp):
                sim_folders.append(entry)
    sim_folders.sort()
    if not sim_folders:
        print(f"No simulation_* folders found in {data_dir}")
        return

    results_rows = []
    # We'll accumulate arrays of all (dk1) from standard, ratio across all simulations
    all_std_dk1 = []
    all_std_dk2 = []
    all_rat_dk1 = []
    all_rat_dk2 = []

    # Also gather global Chi^2 reduced
    std_chi2_list = []
    rat_chi2_list = []

    for simf in sim_folders:
        sim_path = os.path.join(data_dir, simf)
        stats = gather_results(sim_path, st_sub, rt_sub, true_k1=k1_true, true_k2=k2_true)

        row = {
            "Simulation": simf,
            # standard
            "std_mean_dk1": stats["standard"]["mean_dk1"],
            "std_std_dk1":  stats["standard"]["std_dk1"],
            "std_mean_dk2": stats["standard"]["mean_dk2"],
            "std_std_dk2":  stats["standard"]["std_dk2"],
            "std_chi2red":  stats["standard"]["chi2red"],
            # ratio
            "rat_mean_dk1": stats["ratio"]["mean_dk1"],
            "rat_std_dk1":  stats["ratio"]["std_dk1"],
            "rat_mean_dk2": stats["ratio"]["mean_dk2"],
            "rat_std_dk2":  stats["ratio"]["std_dk2"],
            "rat_chi2red":  stats["ratio"]["chi2red"]
        }
        results_rows.append(row)

        # accumulate epoch-level offsets for histogram
        all_std_dk1.extend(stats["standard"]["dk1_all"])
        all_std_dk2.extend(stats["standard"]["dk2_all"])
        all_rat_dk1.extend(stats["ratio"]["dk1_all"])
        all_rat_dk2.extend(stats["ratio"]["dk2_all"])

        if not np.isnan(stats["standard"]["chi2red"]):
            std_chi2_list.append(stats["standard"]["chi2red"])
        if not np.isnan(stats["ratio"]["chi2red"]):
            rat_chi2_list.append(stats["ratio"]["chi2red"])

    df = pd.DataFrame(results_rows)
    if df.empty:
        print("No data => no summary produced.")
        return

    # Save the table
    summary_csv = os.path.join(data_dir, outcsv)
    df.to_csv(summary_csv, index=False)
    print(f"Saved summary CSV => {summary_csv}")

    # Print quick overview
    print("\n===== Summary of Simulation Offsets =====\n")
    print(df)

    # Now produce histograms
    plots_dir = os.path.join(data_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # Convert to numpy arrays
    all_std_dk1 = np.array(all_std_dk1)
    all_std_dk2 = np.array(all_std_dk2)
    all_rat_dk1 = np.array(all_rat_dk1)
    all_rat_dk2 = np.array(all_rat_dk2)
    std_chi2_arr= np.array(std_chi2_list)
    rat_chi2_arr= np.array(rat_chi2_list)

    # 1) Hist of standard vs ratio for DK1
    plt.figure(figsize=(8,6))
    plt.hist(all_std_dk1, bins=20, alpha=0.5, color='blue', label='Standard dK1')
    plt.hist(all_rat_dk1, bins=20, alpha=0.5, color='red',  label='Ratio dK1')
    plt.axvline(0, color='k', ls='--', alpha=0.7)
    plt.title("Distribution of (K1_fit - K1_true)")
    plt.xlabel("Offset in K1 (km/s)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(alpha=0.3)
    outpng = os.path.join(plots_dir, "hist_K1_offsets.png")
    plt.savefig(outpng, dpi=300)
    plt.close()
    print(f"Saved histogram: {outpng}")

    # 2) Hist of standard vs ratio for DK2
    plt.figure(figsize=(8,6))
    plt.hist(all_std_dk2, bins=20, alpha=0.5, color='blue', label='Standard dK2')
    plt.hist(all_rat_dk2, bins=20, alpha=0.5, color='red',  label='Ratio dK2')
    plt.axvline(0, color='k', ls='--', alpha=0.7)
    plt.title("Distribution of (K2_fit - K2_true)")
    plt.xlabel("Offset in K2 (km/s)")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(alpha=0.3)
    outpng = os.path.join(plots_dir, "hist_K2_offsets.png")
    plt.savefig(outpng, dpi=300)
    plt.close()
    print(f"Saved histogram: {outpng}")

    # 3) Scatter plot of standard vs ratio for each simulation's mean offset
    #    e.g., compare mean_dk1 for standard vs ratio
    #    or we can do a single 2D plot with x=std_mean_dk1, y=rat_mean_dk1
    #    to see if ratio method is systematically smaller or bigger
    plt.figure(figsize=(8,6))
    x_std = df["std_mean_dk1"].values
    y_rat = df["rat_mean_dk1"].values
    plt.scatter(x_std, y_rat, color='purple', alpha=0.7)
    plt.axhline(0, color='k', ls='--', alpha=0.5)
    plt.axvline(0, color='k', ls='--', alpha=0.5)
    plt.xlabel("Standard mean_dK1 (km/s)")
    plt.ylabel("Ratio mean_dK1 (km/s)")
    plt.title("Comparison: Standard vs. Ratio (mean K1 offsets, per simulation)")
    plt.grid(alpha=0.3)
    outpng = os.path.join(plots_dir, "scatter_meanDK1_std_vs_ratio.png")
    plt.savefig(outpng, dpi=300)
    plt.close()
    print(f"Saved scatter plot: {outpng}")

    # 4) If you want a quick histogram of reduced chi^2 for standard vs ratio:
    if len(std_chi2_arr)>0 or len(rat_chi2_arr)>0:
        plt.figure(figsize=(8,6))
        if len(std_chi2_arr)>0:
            plt.hist(std_chi2_arr, bins=10, alpha=0.5, label='Standard Chi^2_red')
        if len(rat_chi2_arr)>0:
            plt.hist(rat_chi2_arr, bins=10, alpha=0.5, label='Ratio Chi^2_red')
        plt.title("Distribution of Reduced Chi^2 (Global)")
        plt.xlabel("Reduced Chi^2")
        plt.ylabel("Count")
        plt.legend()
        plt.grid(alpha=0.3)
        outpng = os.path.join(plots_dir, "hist_chi2red.png")
        plt.savefig(outpng, dpi=300)
        plt.close()
        print(f"Saved histogram: {outpng}")

    print("\nAll plots saved in 'plots' subfolder. Done.\n")


if __name__ == "__main__":
    main()
