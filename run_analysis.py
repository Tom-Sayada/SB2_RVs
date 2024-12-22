import subprocess
import os
import shutil
import matplotlib.pyplot as plt

# If you want to pre-define a set of known lines, you can do so here.
lines_info = {
    'He4471': {'wavelength': 4471.5, 'window': 15.0},
    'He4026': {'wavelength': 4026.0, 'window': 15.0},
    'He4388': {'wavelength': 4388.0, 'window': 15.0},
}

def prompt_for_lines():
    if lines_info:
        print("Available lines:")
        for ln in lines_info.keys():
            print(f" - {ln}")
        print("Enter the lines you'd like to run the fit on, separated by commas.")
        while True:
            selection = input("Lines: ").strip()
            selected_lines = [line.strip() for line in selection.split(',') if line.strip()]
            if all(line in lines_info for line in selected_lines):
                return selected_lines
            else:
                print("One or more selected lines are not available. Please try again.")
    else:
        selection = input("Enter lines separated by commas: ").strip()
        return [line.strip() for line in selection.split(',') if line.strip()]

def run_fit_for_simulations(base_dir, simulations_folder, fit_type, profile_type, selected_lines):
    """
    Runs standard or ratio fits on all subfolders (e.g. simulation_1, simulation_2, ...).
    Updated to place results in separate folders if using skewed profiles:
     - standard_fit_results   or standard_skewed_fit_results
     - ratio_constrained_results or ratio_constrained_skewed_results
    """

    import os
    import subprocess
    import matplotlib.pyplot as plt

    print("Do you want to use weighted or unweighted residuals?")
    print("1) Weighted (default)")
    print("2) Unweighted")
    choice_unweighted = input("Enter 1 or 2: ").strip()
    use_unweighted = (choice_unweighted == '2')

    # Build lines argument
    lines_arg = ",".join(selected_lines)

    sim_dirs = [d for d in os.listdir(simulations_folder) if d.startswith('simulation_')]
    sim_dirs.sort(key=lambda x: int(x.split('_')[-1]))  # Sort by simulation number

    results_map = {}  # simulation_number -> (results_folder, fit_type, success)

    for sim_dir in sim_dirs:
        sim_path = os.path.join(simulations_folder, sim_dir)
        sim_num = int(sim_dir.split('_')[-1])

        # Decide output folder name based on fit_type + profile_type
        if fit_type == 'standard':
            if profile_type == 'skewed':
                output_dir = os.path.join(sim_path, 'standard_skewed_fit_results')
            else:
                output_dir = os.path.join(sim_path, 'standard_fit_results')
            cmd_py = "run_standard_fit.py"
        else:
            if profile_type == 'skewed':
                output_dir = os.path.join(sim_path, 'ratio_constrained_skewed_results')
            else:
                output_dir = os.path.join(sim_path, 'ratio_constrained_results')
            cmd_py = "run_ratio_constrained_fit.py"

        cmd = [
            "python",
            os.path.join(base_dir, "src", cmd_py),
            "--data_dir", sim_path,
            "--output_dir", output_dir,
            "--profile_type", profile_type,
            "--lines", lines_arg
        ]
        if use_unweighted:
            cmd.append("--unweighted")

        print(f"\nRunning {fit_type} fit for {sim_dir} with profile_type={profile_type} on lines: {lines_arg}")
        print("Command:", " ".join(cmd))

        os.makedirs(output_dir, exist_ok=True)
        try:
            subprocess.run(cmd, check=True)
            results_map[sim_num] = (output_dir, fit_type, True)
        except subprocess.CalledProcessError as e:
            print(f"Error running {fit_type} fit for {sim_dir}: {e}")
            results_map[sim_num] = (output_dir, fit_type, False)
            continue

        # Attempt to show result plots if they exist
        for plotname in ['rv2_vs_rv1.png', 'rv_vs_epoch.png']:
            p = os.path.join(output_dir, plotname)
            if os.path.exists(p):
                img = plt.imread(p)
                plt.figure()
                plt.imshow(img)
                plt.title(f"{sim_dir} - {plotname}")
                plt.axis('off')

        ep_plot = os.path.join(output_dir, 'epoch_0_fit.png')
        if os.path.exists(ep_plot):
            img = plt.imread(ep_plot)
            plt.figure()
            plt.imshow(img)
            plt.title(f"{sim_dir} - epoch_0_fit.png")
            plt.axis('off')

        plt.show()

    return results_map

def run_fit_for_real_data(base_dir, profile_type):
    import subprocess
    import matplotlib.pyplot as plt

    selected_lines = prompt_for_lines()
    lines_arg = ",".join(selected_lines)

    print("Do you want to use weighted or unweighted residuals?")
    print("1) Weighted (default)")
    print("2) Unweighted")
    choice_unweighted = input("Enter 1 or 2: ").strip()
    use_unweighted = (choice_unweighted == '2')

    while True:
        print("Choose a fit model:")
        print("1) Standard Fit")
        print("2) Ratio-Constrained Fit")
        print("3) Exit")
        choice = input("Enter 1, 2 or 3: ").strip()

        if choice == '1':
            # If user picks skewed => standard_skewed_fit_results, else standard_fit_results
            outname = 'standard_skewed_fit_results' if (profile_type=='skewed') else 'standard_fit_results'
            cmd = [
                "python",
                os.path.join(base_dir, "src", "run_standard_fit.py"),
                "--profile_type", profile_type,
                "--lines", lines_arg
            ]
            if use_unweighted:
                cmd.append("--unweighted")

            try:
                subprocess.run(cmd, check=True)
                output_dir = os.path.join(base_dir, 'output', outname)
                for plotname in ['rv2_vs_rv1.png','rv_vs_epoch.png']:
                    p = os.path.join(output_dir,plotname)
                    if os.path.exists(p):
                        img=plt.imread(p)
                        plt.figure()
                        plt.imshow(img)
                        plt.title(plotname)
                        plt.axis('off')
                ep_plot = os.path.join(output_dir,'epoch_0_fit.png')
                if os.path.exists(ep_plot):
                    img=plt.imread(ep_plot)
                    plt.figure()
                    plt.imshow(img)
                    plt.title('Epoch 0 fit')
                    plt.axis('off')
                plt.show()
            except subprocess.CalledProcessError as e:
                print(f"Error running standard fit: {e}")

        elif choice == '2':
            # If user picks skewed => ratio_constrained_skewed_results, else ratio_constrained_results
            outname = 'ratio_constrained_skewed_results' if (profile_type=='skewed') else 'ratio_constrained_results'
            cmd = [
                "python",
                os.path.join(base_dir,'src','run_ratio_constrained_fit.py'),
                "--profile_type", profile_type,
                "--lines", lines_arg
            ]
            if use_unweighted:
                cmd.append("--unweighted")

            try:
                subprocess.run(cmd, check=True)
                output_dir = os.path.join(base_dir,'output', outname)
                for plotname in ['rv2_vs_rv1.png','rv_vs_epoch.png']:
                    p=os.path.join(output_dir,plotname)
                    if os.path.exists(p):
                        img=plt.imread(p)
                        plt.figure()
                        plt.imshow(img)
                        plt.title(plotname)
                        plt.axis('off')
                ep_plot = os.path.join(output_dir,'epoch_0_fit.png')
                if os.path.exists(ep_plot):
                    img=plt.imread(ep_plot)
                    plt.figure()
                    plt.imshow(img)
                    plt.title('Epoch 0 fit')
                    plt.axis('off')
                plt.show()
            except subprocess.CalledProcessError as e:
                print(f"Error running ratio-constrained fit: {e}")

        elif choice=='3':
            break
        else:
            print("Invalid choice.")

def gather_and_save_results(simulations_folder, results_map):
    """
    existing logic that copies fit_results.xlsx
    from each simulation to a summary location
    """
    base_out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'output'))
    main_sim_name = os.path.basename(os.path.normpath(simulations_folder))
    main_sim_output_dir = os.path.join(base_out_dir, main_sim_name)
    os.makedirs(main_sim_output_dir, exist_ok=True)

    standard_map = {k:v for k,v in results_map.items() if v[1]=='standard' and v[2]}
    ratio_map = {k:v for k,v in results_map.items() if v[1]=='ratio' and v[2]}

    # Copy standard results
    if len(standard_map)>0:
        standard_dest = os.path.join(main_sim_output_dir, 'standard')
        os.makedirs(standard_dest, exist_ok=True)
        for sim_num,(res_dir, ftype, success) in standard_map.items():
            src = os.path.join(res_dir, 'fit_results.xlsx')
            if os.path.exists(src):
                dest_filename = f"standard_fit_results_{sim_num}.xlsx"
                dest = os.path.join(standard_dest, dest_filename)
                shutil.copyfile(src, dest)
            else:
                print(f"No fit_results.xlsx found for simulation {sim_num} in {res_dir}")

    # Copy ratio results
    if len(ratio_map)>0:
        ratio_dest = os.path.join(main_sim_output_dir, 'ratio')
        os.makedirs(ratio_dest, exist_ok=True)
        for sim_num,(res_dir, ftype, success) in ratio_map.items():
            src = os.path.join(res_dir, 'fit_results.xlsx')
            if os.path.exists(src):
                dest_filename = f"ratio_fit_results_{sim_num}.xlsx"
                dest = os.path.join(ratio_dest, dest_filename)
                shutil.copyfile(src, dest)
            else:
                print(f"No fit_results.xlsx found for simulation {sim_num} in {res_dir}")

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..'))

    print("Are you running on simulated or real data?")
    print("1) Simulated Data")
    print("2) Real Data")
    data_choice = input("Enter 1 or 2: ").strip()

    print("Do you want symmetric Voigts or skewed Voigts?")
    print("1) Symmetric")
    print("2) Skewed")
    profile_choice = input("Enter 1 or 2: ").strip()
    if profile_choice == '1':
        profile_type = 'sym'
    else:
        profile_type = 'skewed'

    if data_choice=='1':
        print("Please provide the full path to the folder containing the simulations.")
        simulations_folder = input("Enter the path: ").strip()

        if not os.path.isdir(simulations_folder):
            print("Provided path is not a directory. Exiting.")
            return

        while True:
            print("Choose a fit model to run on ALL simulations in this folder:")
            print("1) Standard Fit")
            print("2) Ratio-Constrained Fit")
            print("3) Exit")
            choice = input("Enter 1, 2 or 3: ").strip()
            if choice=='1':
                selected_lines = prompt_for_lines()
                results_map = run_fit_for_simulations(base_dir, simulations_folder, 'standard', profile_type, selected_lines)
                gather_and_save_results(simulations_folder, results_map)
            elif choice=='2':
                selected_lines = prompt_for_lines()
                results_map = run_fit_for_simulations(base_dir, simulations_folder, 'ratio', profile_type, selected_lines)
                gather_and_save_results(simulations_folder, results_map)
            elif choice=='3':
                break
            else:
                print("Invalid choice.")
    else:
        # Real data
        run_fit_for_real_data(base_dir, profile_type)

if __name__=="__main__":
    main()
