import os
import re
import pandas as pd


def parse_epoch_str(epoch_str):
    """
    Extract a numeric epoch index from a string like:
    'renamed_simulations/simulation_1/renamed_obs/sim_1_obs_0'
    or from an epoch string in the real RV files. Here we extract the number after 'obs_'
    (if available), otherwise returns the string unchanged.
    """
    match = re.search(r'obs_(\d+)$', epoch_str)
    if match:
        return int(match.group(1))
    else:
        return epoch_str


def collect_sb2_data(sb2_csv_path):
    """
    Read the SB2 CSV, which has columns: epoch, mean_rv, mean_rv_er, comp.
    Pivot the data so that each epoch becomes one row with separate columns
    for each component, and add an 'epoch_index' (as string) for merging.
    """
    df = pd.read_csv(sb2_csv_path)
    df_pivot = df.pivot(index='epoch', columns='comp', values=['mean_rv', 'mean_rv_er'])
    # Flatten the multi-index columns
    df_pivot.columns = [f"SB2_{col[0]}_{int(col[1])}" for col in df_pivot.columns]
    # Rename for clarity
    rename_map = {
        'SB2_mean_rv_1': 'SB2_rv_1',
        'SB2_mean_rv_er_1': 'SB2_rv_1_er',
        'SB2_mean_rv_2': 'SB2_rv_2',
        'SB2_mean_rv_er_2': 'SB2_rv_2_er'
    }
    df_pivot = df_pivot.rename(columns=rename_map)
    # Save the original epoch string (from SB2)
    df_pivot['epoch_str_sb2'] = df_pivot.index
    # Create a common 'epoch_index' for merging and cast to string
    df_pivot['epoch_index'] = df_pivot['epoch_str_sb2'].apply(parse_epoch_str).astype(str)
    return df_pivot.reset_index(drop=True)


def collect_excel_data(xlsx_path, label):
    """
    Reads an Excel file's 'Per_Epoch_RVs' sheet.
    Expected columns: Epoch, RV1, RV1_err, RV2, RV2_err.
    Renames the RV columns to include the provided label, and
    adds a common 'epoch_index' (as string) for merging.
    """
    df = pd.read_excel(xlsx_path, sheet_name='Per_Epoch_RVs')
    df[f'epoch_str_{label}'] = df['Epoch'].astype(str)
    df['epoch_index'] = df[f'epoch_str_{label}'].apply(parse_epoch_str).astype(str)
    df = df.rename(columns={
        'RV1': f'{label}_rv_1',
        'RV1_err': f'{label}_rv_1_err',
        'RV2': f'{label}_rv_2',
        'RV2_err': f'{label}_rv_2_err'
    })
    cols = [f'epoch_str_{label}', 'epoch_index', f'{label}_rv_1', f'{label}_rv_1_err', f'{label}_rv_2',
            f'{label}_rv_2_err']
    return df[cols]


def collect_real_data(real_csv_path):
    """
    Reads the real RV CSV file (star_rvs_per_epoch.csv).
    Expected columns: 'Epoch', 'RV1', 'RV2'.

    This function:
      1) Renames 'RV1' -> 'real_rv_1'
      2) Renames 'RV2' -> 'real_rv_2'
      3) Uses 'Epoch' to build 'epoch_str_real' and 'epoch_index' for merging
         (cast to string to match other data).
    """
    df = pd.read_csv(real_csv_path)

    # Use the correct column name from your CSV, e.g. 'Epoch' (capital E).
    # Create an epoch string and epoch_index (coerced to string).
    df['epoch_str_real'] = df['Epoch'].astype(str)
    df['epoch_index'] = df['epoch_str_real'].apply(parse_epoch_str).astype(str)

    # Rename columns so the final comparison file has 'real_rv_1' and 'real_rv_2'
    df = df.rename(columns={
        'RV1': 'real_rv_1',
        'RV2': 'real_rv_2'
    })

    # Select only the columns you need
    cols = ['epoch_str_real', 'epoch_index', 'real_rv_1', 'real_rv_2']
    return df[cols]


def main():
    # Define base directories for each type of data
    sb2_base = "/Users/tomsayada/spectral_analysis_project/data/for jaime/simulations_results"
    main_base = "/Users/tomsayada/spectral_analysis_project/data/for jaime"
    real_base = "/Users/tomsayada/spectral_analysis_project/data/K_1_100.0_K_2_200.0_SNR_50.0_Q_0.5_vsini_200.0"

    num_simulations = 50
    all_sim_results = []

    for sim_num in range(1, num_simulations + 1):
        dfs = []  # To collect DataFrames for SB2, Voigt, Gaussian, and real RVs

        # --- SB2 Data ---
        sb2_csv_path = os.path.join(sb2_base, f"simulation_{sim_num}", "SB2", "fit_values.csv")
        if os.path.isfile(sb2_csv_path):
            try:
                df_sb2 = collect_sb2_data(sb2_csv_path)
                dfs.append(df_sb2)
            except Exception as e:
                print(f"[ERROR] Reading SB2 for simulation {sim_num} at {sb2_csv_path}: {e}")
        else:
            print(f"[WARNING] Missing SB2 file for simulation {sim_num}: {sb2_csv_path}")

        # --- Voigt Data ---
        voigt_path = os.path.join(main_base, f"simulation_{sim_num}",
                                  "standard_sym_voigt_weighted_nobaseline_fit_results", "fit_results.xlsx")
        if os.path.isfile(voigt_path):
            try:
                df_voigt = collect_excel_data(voigt_path, label='voigt')
                dfs.append(df_voigt)
            except Exception as e:
                print(f"[ERROR] Reading Voigt for simulation {sim_num} at {voigt_path}: {e}")
        else:
            print(f"[WARNING] Missing Voigt Excel for simulation {sim_num}: {voigt_path}")

        # --- Gaussian Data ---
        gauss_path = os.path.join(main_base, f"simulation_{sim_num}",
                                  "standard_sym_gaussian_weighted_nobaseline_auto_fit_results", "fit_results.xlsx")
        if os.path.isfile(gauss_path):
            try:
                df_gauss = collect_excel_data(gauss_path, label='gauss')
                dfs.append(df_gauss)
            except Exception as e:
                print(f"[ERROR] Reading Gaussian for simulation {sim_num} at {gauss_path}: {e}")
        else:
            print(f"[WARNING] Missing Gaussian Excel for simulation {sim_num}: {gauss_path}")

        # --- Real RVs Data ---
        real_csv_path = os.path.join(real_base, f"simulation_{sim_num}", "star_rvs_per_epoch.csv")
        if os.path.isfile(real_csv_path):
            try:
                df_real = collect_real_data(real_csv_path)
                dfs.append(df_real)
            except Exception as e:
                print(f"[ERROR] Reading real RVs for simulation {sim_num} at {real_csv_path}: {e}")
        else:
            print(f"[WARNING] Missing real RV file for simulation {sim_num}: {real_csv_path}")

        # If no data was found for this simulation, skip it
        if not dfs:
            print(f"[WARNING] No data found for simulation {sim_num}.")
            continue

        # Merge available DataFrames using an outer join on 'epoch_index'
        df_merged = dfs[0]
        for df in dfs[1:]:
            df_merged = pd.merge(df_merged, df, on='epoch_index', how='outer')

        # Add the simulation number column
        df_merged['simulation'] = sim_num

        all_sim_results.append(df_merged)

    # Check if any data was collected overall
    if not all_sim_results:
        print("No data was collected. Check your file paths and structure.")
        return

    final_df = pd.concat(all_sim_results, ignore_index=True)
    # Optionally reorder columns placing simulation and epoch_index first
    col_order = ['simulation', 'epoch_index']
    final_cols = col_order + [col for col in final_df.columns if col not in col_order]
    final_df = final_df[final_cols]

    output_file = "all_comparisons.xlsx"
    final_df.to_excel(output_file, index=False)
    print(f"Successfully wrote combined data to {output_file}")


if __name__ == "__main__":
    main()
