import os
import re
import pandas as pd


def parse_top_folder_name(folder_name):
    """
    Given a folder name like 'K_1_50.0_K_2_100.0_SNR_50.0_Q_0.5',
    extract the numeric values for K1, K2, SNR, Q.
    Returns (K1, K2, SNR, Q) as floats if parsing works, otherwise None.
    """
    pattern = r'^K_1_(?P<K1>[\d\.]+)_K_2_(?P<K2>[\d\.]+)_SNR_(?P<SNR>[\d\.]+)_Q_(?P<Q>[\d\.]+)$'
    match = re.match(pattern, folder_name)
    if match:
        return (
            float(match.group('K1')),
            float(match.group('K2')),
            float(match.group('SNR')),
            float(match.group('Q'))
        )
    return None


def parse_simulation_folder_name(folder_name):
    """
    Given a folder name like 'simulation_1', extract the integer 1.
    Returns the simulation number as int if parsing works, otherwise None.
    """
    pattern = r'^simulation_(?P<num>\d+)$'
    match = re.match(pattern, folder_name)
    if match:
        return int(match.group('num'))
    return None


def parse_method_from_folder(folder_name):
    """
    Given a results folder like:
        'ratio_sym_voigt_weighted_nobaseline_fit_results'
        'standard_sym_voigt_weighted_nobaseline_fit_results'
        'standard_sym_gaussian_weighted_nobaseline_fit_results'
        etc.
    returns a shortened or parsed method name (or the entire string).
    """
    return folder_name.replace('_fit_results', '')


def main():
    # Change this to your main data folder
    data_root = "/Users/tomsayada/spectral_analysis_project/data"

    # Lists for collecting records that will become DataFrames
    all_per_epoch_records = []
    all_chi_square_records = []

    # Walk through the top-level directory (K_1_#_K_2_#_SNR_#_Q_#)
    for folder_name in os.listdir(data_root):
        folder_path = os.path.join(data_root, folder_name)

        if os.path.isdir(folder_path):
            # Attempt to parse the K1, K2, SNR, Q from the folder name
            parsed = parse_top_folder_name(folder_name)
            if not parsed:
                # If it doesn't match the pattern, skip
                continue

            K1_val, K2_val, SNR_val, Q_val = parsed

            # Now look for subfolders named simulation_<number>
            for sim_folder_name in os.listdir(folder_path):
                sim_folder_path = os.path.join(folder_path, sim_folder_name)
                if os.path.isdir(sim_folder_path):
                    sim_number = parse_simulation_folder_name(sim_folder_name)
                    if sim_number is None:
                        continue  # Not a valid simulation folder

                    # Look for method result folders
                    for method_folder_name in os.listdir(sim_folder_path):
                        method_folder_path = os.path.join(sim_folder_path, method_folder_name)
                        if os.path.isdir(method_folder_path) and method_folder_name.endswith("_fit_results"):
                            # Parse out the method name
                            method_name = parse_method_from_folder(method_folder_name)

                            # The fit_results.xlsx is supposed to be in this folder
                            fit_results_path = os.path.join(method_folder_path, "fit_results.xlsx")

                            # If the file exists, try reading it
                            if os.path.isfile(fit_results_path):
                                try:
                                    # First check what sheets are present
                                    xls_file = pd.ExcelFile(fit_results_path)
                                    sheet_names = xls_file.sheet_names

                                    # 1) Read Per_Epoch_RVs sheet if it exists
                                    if "Per_Epoch_RVs" in sheet_names:
                                        per_epoch_df = pd.read_excel(fit_results_path,
                                                                     sheet_name="Per_Epoch_RVs")
                                        # Extend each row with global info
                                        for _, row in per_epoch_df.iterrows():
                                            record = {
                                                "K1": K1_val,
                                                "K2": K2_val,
                                                "SNR": SNR_val,
                                                "Q": Q_val,
                                                "simulation_number": sim_number,
                                                "method": method_name,
                                                # Original columns
                                                "Epoch": row.get("Epoch"),
                                                "RV1": row.get("RV1"),
                                                "RV1_err": row.get("RV1_err"),
                                                "RV2": row.get("RV2"),
                                                "RV2_err": row.get("RV2_err"),
                                            }
                                            all_per_epoch_records.append(record)
                                    else:
                                        print(f"Warning: 'Per_Epoch_RVs' sheet not found "
                                              f"in {fit_results_path}. Skipping Per_Epoch_RVs data.")

                                    # 2) Read Chi_Square_statistics sheet if it exists
                                    if "Chi_Square_Statistics" in sheet_names:
                                        chi_square_df = pd.read_excel(fit_results_path,
                                                                      sheet_name="Chi_Square_Statistics")
                                        for _, row in chi_square_df.iterrows():
                                            record = {
                                                "K1": K1_val,
                                                "K2": K2_val,
                                                "SNR": SNR_val,
                                                "Q": Q_val,
                                                "simulation_number": sim_number,
                                                "method": method_name,
                                                # Original columns
                                                "Line_ID": row.get("Line_ID"),
                                                "Chi_Square": row.get("Chi_Square"),
                                                "N_points": row.get("N_points"),
                                                "Chi_Square_reduced": row.get("Chi_Square_reduced"),
                                                "p_value": row.get("p_value"),
                                                "ratio": row.get("ratio"),
                                                "ratio_err": row.get("ratio_err"),
                                            }
                                            all_chi_square_records.append(record)
                                    else:
                                        print(f"Warning: 'Chi_Square_Statistics' sheet not found "
                                              f"in {fit_results_path}. Skipping Chi_Square data.")

                                except Exception as e:
                                    # Catch any read/parsing errors and keep going
                                    print(f"Error reading {fit_results_path}: {e}")

    # Convert our records to DataFrames
    df_per_epoch = pd.DataFrame(all_per_epoch_records)
    df_chi_square = pd.DataFrame(all_chi_square_records)

    # Write a combined Excel file with two sheets
    output_path = os.path.join(data_root, "combined_fit_results.xlsx")
    with pd.ExcelWriter(output_path) as writer:
        df_per_epoch.to_excel(writer, sheet_name="All_Per_Epoch_RVs", index=False)
        df_chi_square.to_excel(writer, sheet_name="All_Chi_Square_stats", index=False)

    print(f"Combined results written to {output_path}")


if __name__ == "__main__":
    main()
