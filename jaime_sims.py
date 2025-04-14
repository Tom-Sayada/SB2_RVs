import os
import shutil
import zipfile


def main():
    # Define the root directory containing the simulation folders.
    root_dir = '/Users/tomsayada/spectral_analysis_project/data/K_1_100.0_K_2_200.0_SNR_50.0_Q_0.5_vsini_200.0 copy'

    # Define the name of the new folder to hold renamed files inside each simulation folder.
    new_folder_name = "renamed_obs"

    # List all simulation folders (assumed to start with "simulation_").
    simulation_folders = [
        f for f in os.listdir(root_dir)
        if f.startswith("simulation_") and os.path.isdir(os.path.join(root_dir, f))
    ]

    # To collect paths of all newly created folders for zipping later.
    new_folders_paths = []

    for sim_folder in simulation_folders:
        sim_path = os.path.join(root_dir, sim_folder)
        new_folder_path = os.path.join(sim_path, new_folder_name)
        os.makedirs(new_folder_path, exist_ok=True)
        new_folders_paths.append(new_folder_path)

        # Extract the simulation number from the folder name.
        # Assumes folder name is like "simulation_x".
        parts_sim = sim_folder.split('_')
        if len(parts_sim) < 2:
            print(f"Skipping folder {sim_folder} as it doesn't match expected naming convention.")
            continue
        sim_number = parts_sim[1]

        # Process each file in the simulation folder.
        for file in os.listdir(sim_path):
            file_path = os.path.join(sim_path, file)
            # Check if it's a file and its name starts with "obs_"
            if os.path.isfile(file_path) and file.startswith("obs_"):
                parts = file.split('_')
                if len(parts) >= 2:
                    # Extract the epoch number (the number right after "obs")
                    epoch_number = parts[1]
                    # Construct the new file name as "sim_x_obs_y"
                    new_file_name = f"sim_{sim_number}_obs_{epoch_number}"
                    dest_file_path = os.path.join(new_folder_path, new_file_name)
                    shutil.copyfile(file_path, dest_file_path)

    # Create a single zip file containing all the new folders.
    zip_filename = os.path.join(root_dir, "renamed_simulations.zip")
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for folder in new_folders_paths:
            for folderpath, subfolders, filenames in os.walk(folder):
                for filename in filenames:
                    file_full_path = os.path.join(folderpath, filename)
                    # Archive with a relative path to preserve structure inside the zip.
                    arcname = os.path.relpath(file_full_path, root_dir)
                    zipf.write(file_full_path, arcname)

    print(f"Zipped all renamed folders into: {zip_filename}")


if __name__ == '__main__':
    main()
