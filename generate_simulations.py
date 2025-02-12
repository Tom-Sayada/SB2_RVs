import os
import re
import shutil
from pathlib import Path
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.ndimage import convolve1d


def apply_rotational_broadening(wave_array, flux_array, v_rot, epsilon=0.0):
    """
    Apply rotational broadening to a spectrum in linear wavelength scale.
    """
    if v_rot == 0:
        return flux_array

    c = 299792.458  # speed of light in km/s

    # Convert wavelength to logarithmic scale
    log_wavelengths = np.log(wave_array)

    # Interpolate flux on a uniform logarithmic wavelength grid
    delta_log_lambda = np.mean(np.diff(log_wavelengths))
    log_wavelengths_uniform = np.arange(log_wavelengths.min(),
                                        log_wavelengths.max(),
                                        delta_log_lambda)
    flux_uniform = np.interp(log_wavelengths_uniform, log_wavelengths, flux_array)

    # Convert v_rot to the equivalent delta_log_lambda
    delta_log_lambda_vrot = abs(v_rot) / c

    # Calculate the number of points needed for the broadening kernel
    n_points = int(2 * delta_log_lambda_vrot / delta_log_lambda) + 1
    if n_points % 2 == 0:
        n_points += 1

    x = np.linspace(-abs(v_rot), abs(v_rot), n_points)
    kernel = np.zeros_like(x)

    # Calculate the rotational broadening kernel
    mask = np.abs(x) <= abs(v_rot)
    kernel[mask] = (2 * (1 - epsilon) * np.sqrt(v_rot ** 2 - x[mask] ** 2) +
                    epsilon * (v_rot ** 2 - x[mask] ** 2)) / v_rot ** 2

    # Normalize the kernel
    kernel /= np.sum(kernel)

    # Convolve the flux with the broadening kernel
    broadened_flux_uniform = convolve1d(flux_uniform, kernel, mode='reflect')

    # Interpolate back to the original wavelength grid
    broadened_flux = np.interp(log_wavelengths,
                               log_wavelengths_uniform,
                               broadened_flux_uniform)

    return broadened_flux


def show_template_comparison(vsini):
    """Show the secondary template with and without rotational broadening"""
    # Set up the wavelength grid
    lamB, lamR = 3900.0, 4600.0
    Resolution = 6200.0
    Sampling = 2.0
    lammid = (lamB + lamR) / 2.0
    DlamRes = lammid / Resolution
    Dlam = DlamRes / Sampling
    wavegrid = np.arange(lamB, lamR, Dlam)

    # Load the template
    MaskPath2 = '/Users/tomsayada/spectral_analysis_project/data/templates/BG22000g400v2.vis.rect.dat'
    MaskTemp2 = np.loadtxt(MaskPath2)
    Waves2 = MaskTemp2[:, 0] + np.random.normal(0, 1E-10, len(MaskTemp2))

    # Interpolate to our wavelength grid
    Mask2 = interp1d(Waves2, MaskTemp2[:, 1], bounds_error=False, fill_value=1.0, kind='cubic')(wavegrid)

    # Apply rotational broadening
    Mask2_broadened = apply_rotational_broadening(wavegrid, Mask2, vsini, epsilon=0.0)

    # Create the comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot original template
    ax1.plot(wavegrid, Mask2, 'b-', label='Original Template')
    ax1.set_title('Original Template')
    ax1.set_xlabel('Wavelength (Å)')
    ax1.set_ylabel('Normalized Flux')
    ax1.grid(True)

    # Plot broadened template
    ax2.plot(wavegrid, Mask2_broadened, 'r-', label=f'With v sin i = {vsini} km/s')
    ax2.set_title(f'Template with v sin i = {vsini} km/s')
    ax2.set_xlabel('Wavelength (Å)')
    ax2.set_ylabel('Normalized Flux')
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def parse_and_write_rvs(sim_folder: Path):
    """
    Scan the simulation folder for all 'obs_XX_V1_YY_V2_ZZ.txt' files,
    extract epoch and RVs, and save them to star_rvs_per_epoch.csv.
    """
    pattern = r'obs_(\d+)_V1_([+-]?\d+\.\d+)_V2_([+-]?\d+\.\d+)'
    data = []

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
    K1 = float(input("Enter the value for K1: "))
    K2 = float(input("Enter the value for K2: "))
    snr = float(input("Enter the value for SNR: "))
    Q = float(input("Enter the value for Q: "))
    vsini = float(input("Enter the desired v sin i for the secondary star (km/s): "))

    # Show template comparison and wait for user to close the plot
    print("\nShowing template comparison. Close the plot window to continue with simulation generation...")
    show_template_comparison(vsini)

    num_simulations = int(input("Enter the number of simulations to run: "))

    base_dir = Path("/Users/tomsayada/spectral_analysis_project/data")
    main_folder_name = f"K_1_{K1}_K_2_{K2}_SNR_{snr}_Q_{Q}_vsini_{vsini}"
    main_folder_path = base_dir / main_folder_name

    if main_folder_path.exists():
        shutil.rmtree(main_folder_path)

    main_folder_path.mkdir(parents=True)

    for sim_idx in range(1, num_simulations + 1):
        sim_folder = main_folder_path / f"simulation_{sim_idx}"
        sim_folder.mkdir()

        print(f"\n=== Generating simulation {sim_idx} in {sim_folder} ===")
        command = [
            "python", "make_spectra_SB2.py",
            "--K1", str(K1),
            "--K2", str(K2),
            "--S2N", str(snr),
            "--Q", str(Q),
            "--vsini", str(vsini),
            "--output_dir", str(sim_folder)
        ]
        subprocess.run(command, check=True)

        parse_and_write_rvs(sim_folder)

    print(f"\nAll simulations successfully created in {main_folder_path}")


if __name__ == "__main__":
    main()