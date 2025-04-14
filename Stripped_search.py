#!/usr/bin/env python3

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
from astropy.io import fits


###############################################################################
# 1) Reading logic (from your “Voigt” script) – handles FITS or ASCII
###############################################################################

def read_fits(infile):
    """
    Read a FITS file expecting columns WAVELENGTH, SCI_NORM in a BinTableHDU.
    Returns (wave, flux). If not found, returns (None, None).
    """
    print(f"Reading FITS file: {infile}")
    try:
        with fits.open(infile) as hdul:
            for i, hdu in enumerate(hdul):
                if isinstance(hdu, fits.BinTableHDU):
                    # Check if we have the columns
                    table_data = hdu.data
                    colnames = table_data.columns.names
                    if 'WAVELENGTH' in colnames and 'SCI_NORM' in colnames:
                        wave = table_data['WAVELENGTH']
                        flux = table_data['SCI_NORM']
                        return wave, flux
        print("No suitable WAVELENGTH/SCI_NORM found in the FITS file.")
    except Exception as e:
        print(f"Error reading FITS file: {e}")
    return None, None


def read_ascii(infile, col0=0, col1=1, comment='#', skip_lines=0):
    """
    Read an ASCII file with at least two columns: wave, flux.
    """
    print(f"Reading ASCII file: {infile}")
    try:
        spec = pd.read_csv(infile, header=None, delim_whitespace=True,
                           comment=comment, skiprows=skip_lines).values
        wave = spec[:, col0]
        flux = spec[:, col1]
        return wave, flux
    except Exception as e:
        print(f"Error reading ASCII file {infile}: {e}")
        return None, None


def read_file(infile, col0=0, col1=1, comment='#', skip_lines=0):
    """
    Decide whether to read FITS or ASCII based on extension.
    Returns (wave, flux) or (None, None) on failure.
    """
    ext = infile.lower().split('.')[-1]
    if ext in ['fits', 'fit']:
        return read_fits(infile)
    else:
        return read_ascii(infile, col0=col0, col1=col1, comment=comment, skip_lines=skip_lines)


###############################################################################
# 2) Checking line depths
###############################################################################

def get_min_flux_in_window(wave, flux, center, width=3.0):
    """
    Return the minimum flux in the +/- width range around 'center'.
    If the window is out of bounds (no points), returns None.
    """
    mask = (wave >= center - width) & (wave <= center + width)
    if not np.any(mask):
        return None
    return flux[mask].min()


###############################################################################
# 3) Main script: scanning /Users/tomsayada/spectral_analysis_project/data/BLOEM_ALL
###############################################################################

def main():
    # 3a) Directory that has subfolders "F_NNN" containing FITS or ASCII files
    BASE_DIR = "/Users/tomsayada/spectral_analysis_project/data/BLOEM_ALL"

    # We'll look for any subdirectory under BLOEM_ALL, then any file with extension.
    # If you only want .fits, change pattern to "*.fits"
    PATTERN = os.path.join(BASE_DIR, "*", "*")

    # Define line centers (in Angstroms)
    H_GAMMA_CENTER = 4340.0
    HE_LINES = [4026.0, 4143.0, 4387.0, 4471.0]

    # We'll store which folders pass the condition
    hgamma_shallower_objects = []

    # We'll optionally save plots. Set to False if you don't need them.
    SAVE_PLOTS = True
    PLOT_OUTDIR = os.path.join(BASE_DIR, "plots")
    os.makedirs(PLOT_OUTDIR, exist_ok=True)

    # 3b) Gather all files in all subfolders
    all_files = glob.glob(PATTERN)

    # 3c) For each file, attempt to read wave, flux using the read_file logic
    for filepath in all_files:
        if not os.path.isfile(filepath):
            continue

        # Example path: .../BLOEM_ALL/1_003/BLOeM_1-003_01_Combined.fits
        parent_folder = os.path.basename(os.path.dirname(filepath))  # e.g. "1_003" or "F_003"
        filename = os.path.basename(filepath)

        wave_in, flux_in = read_file(filepath)
        if wave_in is None or flux_in is None:
            # Reading failed or no columns found
            continue

        # Convert possible NaNs to a default flux
        flux_in = np.nan_to_num(flux_in, nan=1.0)

        # 3d) Restrict to 3900–4600 Å
        mask_3900_4600 = (wave_in >= 3900) & (wave_in <= 4600)
        wave_plot = wave_in[mask_3900_4600]
        flux_plot = flux_in[mask_3900_4600]
        if len(wave_plot) < 2:
            continue

        # 3e) Compute min flux near Hγ
        hgamma_min = get_min_flux_in_window(wave_plot, flux_plot, H_GAMMA_CENTER, width=3.0)
        if hgamma_min is None:
            # no data near 4340
            continue

        # Compute min flux for each He line
        he_mins = []
        for he_line_center in HE_LINES:
            he_min = get_min_flux_in_window(wave_plot, flux_plot, he_line_center, width=3.0)
            if he_min is not None:
                he_mins.append(he_min)

        if not he_mins:
            # None of the He lines had data
            continue

        # Check if Hgamma is "less deep" => means Hgamma has a higher min flux than at least one He line
        # i.e. hgamma_min > min(he_mins)
        if hgamma_min > min(he_mins):
            # It's shallower than at least one helium line
            hgamma_shallower_objects.append(parent_folder)

        # 3f) Make a plot (optional)
        if SAVE_PLOTS:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 4))
            plt.plot(wave_plot, flux_plot, color="black", lw=1)
            plt.title(f"{parent_folder}: {filename}")
            plt.xlabel("Wavelength (Å)")
            plt.ylabel("Flux")
            plt.xlim(3900, 4600)
            plt.ylim(bottom=0)

            # Mark Hgamma
            plt.axvline(H_GAMMA_CENTER, color="blue", linestyle="--", alpha=0.7)
            # Mark the He lines
            for he_line_center in HE_LINES:
                if 3900 <= he_line_center <= 4600:
                    plt.axvline(he_line_center, color="red", linestyle="--", alpha=0.5)

            out_plot_name = f"{parent_folder}_{filename}.png"
            out_plot_path = os.path.join(PLOT_OUTDIR, out_plot_name)
            plt.savefig(out_plot_path, dpi=150, bbox_inches="tight")
            plt.close()

    # 3g) Print the result
    # Convert to a set (in case multiple files from same folder matched)
    matched_folders = sorted(set(hgamma_shallower_objects))
    print("Folders where Hγ is shallower than at least one He I line:")
    for folder_name in matched_folders:
        print(f"  {folder_name}")

    print("\nDone! Check the 'plots' folder (if SAVE_PLOTS = True) for PNGs.")


if __name__ == "__main__":
    main()
