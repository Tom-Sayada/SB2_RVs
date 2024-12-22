import os
import re
import shutil
from pathlib import Path
import subprocess
import pandas as pd


def parse_and_write_rvs(sim_folder: Path):
    """
    Scan the simulation folder for all 'obs_XX_V1_YY_V2_ZZ.txt' files,
    extract epoch and RVs, and save them to star_rvs_per_epoch.csv.
    """
    # Regex to match filenames like: obs_0_V1_12.34_V2_-56.78.txt
    pattern = r'obs_(\d+)_V1_([+-]?\d+\.\d+)_V2_([+-]?\d+\.\d+)'

    data = []

    # Iterate over all files in this simulation folder
    for filename in os.listdir(sim_folder):
        match = re.search(pattern, filename)
        if match:
            epoch = int(match.group(1))
            rv1 = float(match.group(2))
            rv2 = float(match.group(3))
            data.append({'Epoch': epoch, 'RV1': rv1, 'RV2': rv2})

    if data:
        df = pd.DataFrame(data).sort_values(by='Epoch').reset_index(drop=True)
        csv_path = sim_folder / 'star_rvs_per_epoch.csv'
        df.to_csv(csv_path, index=False)
        print(f"  -> Saved star_rvs_per_epoch.csv in {csv_path}")
    else:
        print(f"  -> No matching observation files found in {sim_folder}")


def main():
    # Prompt user for input parameters
    K1 = float(input("Enter the value for K1: "))
    K2 = float(input("Enter the value for K2: "))
    snr = float(input("Enter the value for SNR: "))
    Q = float(input("Enter the value for Q: "))
    num_simulations = int(input("Enter the number of simulations to run: "))

    # Define base directory
    base_dir = Path("/Users/tomsayada/spectral_analysis_project/data")

    # Create main folder name
    main_folder_name = f"K_1_{K1}_K_2_{K2}_SNR_{snr}_Q_{Q}"
    main_folder_path = base_dir / main_folder_name

    # Check if folder exists, if so delete it to start fresh
    if main_folder_path.exists():
        shutil.rmtree(main_folder_path)

    # Create main folder
    main_folder_path.mkdir(parents=True)

    # Generate simulations
    for sim_idx in range(1, num_simulations + 1):
        sim_folder = main_folder_path / f"simulation_{sim_idx}"
        sim_folder.mkdir()

        print(f"\n=== Generating simulation {sim_idx} in {sim_folder} ===")
        # Generate 25 observation files using make_spectra_SB2.py
        command = [
            "python", "make_spectra_SB2.py",
            "--K1", str(K1),
            "--K2", str(K2),
            "--S2N", str(snr),
            "--Q", str(Q),
            "--output_dir", str(sim_folder)
        ]
        subprocess.run(command, check=True)

        # After generating the obs_*.txt files, parse them to create star_rvs_per_epoch.csv
        parse_and_write_rvs(sim_folder)

    print(f"\nAll simulations successfully created in {main_folder_path}")


if __name__ == "__main__":
    main()
