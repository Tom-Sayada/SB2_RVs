#!/usr/bin/env python3

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lmfit import Model
from tkinter import Tk, filedialog, simpledialog

import astropy.io.fits as fits
from scipy.special import wofz
from scipy.interpolate import interp1d

###############################################################################
# 1) Utility: Reading files (FITS or ASCII), adopting your approach
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
        # If you have a more flexible approach, use pandas
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
    Returns (wave, flux).
    """
    ext = infile.lower().split('.')[-1]
    if ext in ['fits','fit']:
        return read_fits(infile)
    else:
        return read_ascii(infile, col0=col0, col1=col1, comment=comment, skip_lines=skip_lines)

###############################################################################
# 2) Utility: Single-Voigt fit for line measurements
###############################################################################

def voigt_profile(x, amplitude, center, sigma, gamma):
    """
    amplitude < 0 => absorption line.
    """
    MIN_WIDTH = 1e-5
    sigma = max(sigma, MIN_WIDTH)
    gamma = max(gamma, MIN_WIDTH)
    z = ((x - center) + 1j*gamma) / (sigma*np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma*np.sqrt(2*np.pi))


def fit_single_voigt(wavelength, flux):
    """
    Fit a single Voigt absorption line to (wavelength, flux).
    Return an lmfit.ModelResult with .redchi, etc.
    """
    model = Model(voigt_profile, independent_vars=['x'])
    xmid = 0.5*(wavelength.min() + wavelength.max())

    # Some typical initial guesses
    pars = model.make_params(
        amplitude=-0.5,
        center=xmid,
        sigma=1.0,
        gamma=1.0
    )
    # Bound amplitude negative for absorption
    pars['amplitude'].min = -5.0
    pars['amplitude'].max = -0.01
    # center must remain in the data range
    pars['center'].min = wavelength.min()
    pars['center'].max = wavelength.max()
    # sigma,gamma => positive
    pars['sigma'].min = 0.01
    pars['gamma'].min = 0.01

    result = model.fit(flux, pars, x=wavelength)
    return result

###############################################################################
# 3) Possibly do a rough line-center detection by smoothing
###############################################################################
def find_line_center_smoothed(wv, fl, sigma=2.0, absorption=True):
    """
    A simple center finder: smooth flux, then pick min (for absorption).
    """
    import scipy.ndimage as ndimage
    if len(wv) < 5:
        return None
    sm_flux = ndimage.gaussian_filter1d(fl, sigma=sigma)
    if absorption:
        idx = np.argmin(sm_flux)
    else:
        idx = np.argmax(sm_flux)
    return wv[idx]


###############################################################################
# 4) Main script
###############################################################################

def main():
    # Step A: Ask user to select a folder with multiple observation files
    root = Tk()
    root.withdraw()
    print("Select folder containing your real data files (ASCII or FITS).")
    data_dir = filedialog.askdirectory(title="Select Data Folder")
    root.destroy()

    if not data_dir:
        print("No folder selected. Exiting.")
        sys.exit()

    # Step B: Let user pick line center
    root = Tk()
    root.withdraw()
    line_input = simpledialog.askstring("He I line center", "Enter e.g. 4471.5 for He I line:")
    root.destroy()
    if not line_input:
        print("No line center provided. Exiting.")
        sys.exit()
    try:
        rest_wave = float(line_input)
    except ValueError:
        print("Invalid numeric value for line center. Exiting.")
        sys.exit()

    # Step C: Collect all valid files in the folder
    # We'll just take all items that are not subfolders
    all_files = []
    for entry in os.listdir(data_dir):
        fullp = os.path.join(data_dir, entry)
        if os.path.isfile(fullp):
            all_files.append(fullp)

    if not all_files:
        print("No files found in directory.")
        sys.exit()

    # We'll store info => each file => (filename, epoch??, wave, flux, redchi, model_fit, wave_full, flux_full)
    # "epoch" isn't well-defined for real data, so let's just store "file"
    results = []

    # We'll define a region for single-voigt fit => ±15 Å around line center
    # But first, we do a ±30 Å search to refine the center by smoothing
    search_range = 30.0
    fit_window   = 15.0

    # We'll also define the "full" spectrum range we want to plot => 3900–4600
    PLOT_MIN = 3900
    PLOT_MAX = 4600

    for fpath in all_files:
        wave_in, flux_in = read_file(fpath)
        if wave_in is None or flux_in is None:
            continue

        # Ensure no NaNs, etc.
        flux_in = np.nan_to_num(flux_in, nan=1.0)
        # Possibly select the "full" range first
        mask_full = (wave_in >= PLOT_MIN) & (wave_in <= PLOT_MAX)
        wv_full   = wave_in[mask_full]
        fl_full   = flux_in[mask_full]

        if len(wv_full) < 10:
            # no data in 3900–4600 => skip
            continue

        # Next => line-centered fit
        # 1) Search region => rest_wave ± search_range
        search_min = rest_wave - search_range
        search_max = rest_wave + search_range
        mask_search = (wv_full >= search_min) & (wv_full <= search_max)
        wv_search = wv_full[mask_search]
        fl_search = fl_full[mask_search]
        if len(wv_search) < 5:
            # Can't do line detection
            continue

        # 2) find approximate center by smoothing
        center_est = find_line_center_smoothed(wv_search, fl_search, sigma=2.0, absorption=True)
        if center_est is None:
            center_est = rest_wave  # fallback

        # 3) define final fit window => ±15
        fwin_min = center_est - fit_window
        fwin_max = center_est + fit_window
        mask_fit = (wv_full >= fwin_min) & (wv_full <= fwin_max)
        wv_fit   = wv_full[mask_fit]
        fl_fit   = fl_full[mask_fit]
        if len(wv_fit) < 5:
            continue

        # Single-Voigt fit
        fitres = fit_single_voigt(wv_fit, fl_fit)
        results.append({
            'File'      : fpath,
            'redchi'    : fitres.redchi,
            'wave_full' : wv_full,   # entire 3900–4600 segment
            'flux_full' : fl_full,
            'fitres'    : fitres
        })

    if not results:
        print("No successful fits. Exiting.")
        sys.exit()

    # Step D: pick worst 4 => sorted by redchi descending
    results.sort(key=lambda x: x['redchi'], reverse=True)
    worst_4 = results[:4]

    print("\nFound these 4 worst fits by reduced chi-square:")
    for i, rec in enumerate(worst_4, start=1):
        print(f"{i}) File={os.path.basename(rec['File'])}, redchi={rec['redchi']:.4f}")

    # Step E: Overplot the *full* spectra 3900–4600 for these 4
    plt.figure(figsize=(10,6))

    for i, rec in enumerate(worst_4):
        # entire 3900–4600
        wv_f = rec['wave_full']
        fl_f = rec['flux_full']

        label = f"{os.path.basename(rec['File'])} (redchi={rec['redchi']:.2f})"
        plt.plot(wv_f, fl_f, alpha=0.8, label=label)

    plt.xlabel("Wavelength [Å]")
    plt.ylabel("Flux")
    plt.title("Worst 4 Single-Voigt Fits (Full Spectrum 3900–4600)")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.xlim(PLOT_MIN, PLOT_MAX)
    plt.tight_layout()
    plt.show()

    print("Done! Displayed worst 4 fits on a single plot from 3900–4600.")


if __name__ == "__main__":
    main()
