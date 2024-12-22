#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as fits
from scipy.interpolate import interp1d
from astropy.convolution import Gaussian1DKernel, convolve
from tkinter import Tk, filedialog
import sys
import pandas as pd

try:
    from spectres import spectres
except ImportError:
    def spectres(new_wavs, old_wavs, old_flux, *args, **kwargs):
        f = interp1d(old_wavs, old_flux, bounds_error=False, fill_value="extrapolate")
        return f(new_wavs)


# Mock functions for unsupported file types
def read_fits(infile):
    print(f"Reading FITS file: {infile}")
    with fits.open(infile) as hdul:
        wave = np.arange(len(hdul[0].data))
        flux = hdul[0].data
    return wave, flux


def read_ascii(infile, col0=0, col1=1, comment='#', SkipLines=0):
    print(f"Reading ASCII file: {infile}")
    try:
        spec = pd.read_csv(infile, header=None, delim_whitespace=True, comment=comment, skiprows=SkipLines).values
        wave = spec[:, col0]
        flux = spec[:, col1]
    except Exception as e:
        print(f"Error reading file: {e}")
        wave, flux = None, None
    return wave, flux


def read_file(infile, col0=0, col1=1, comment='#', SkipLines=0):
    ext = infile.split('.')[-1]
    if ext in ['fits', 'fit']:
        return read_fits(infile)
    else:
        return read_ascii(infile, col0, col1, comment, SkipLines)


# File selection
def select_files():
    root = Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select files to plot",
        filetypes=[("All files", "*.*")]
    )
    return list(file_paths)


# Parameters
Legend = True
Norm = False
SaveTxt = False

# Open file dialog
selected_files = select_files()

if not selected_files:
    print("No files selected. Exiting.")
    sys.exit()

fig, ax = plt.subplots()

# Process files
for infile in selected_files:
    wave_in, flux_in = read_file(infile)
    if wave_in is None or flux_in is None:
        continue

    flux_in = np.nan_to_num(flux_in, nan=1.0)

    if Norm:
        flux_in /= np.mean(flux_in)

    label = infile.split("/")[-1]
    ax.plot(wave_in, flux_in, linewidth=1.0, alpha=0.8, label=label)

# Spectral lines for O and B stars
lines_b_stars = {'He I 4471': 4471, 'Mg II 4481': 4481}
plt.draw()
ylim = ax.get_ylim()

for line_name, wavelength in lines_b_stars.items():
    ax.axvline(x=wavelength, color='red', linestyle='--', linewidth=0.5)
    ax.text(wavelength, ylim[1] * 0.98, line_name, rotation=90, fontsize=8, color='red')

# Finalize plot
ax.set_xlabel("Wavelength [Å]")
ax.set_ylabel("Flux")
if Legend:
    ax.legend()
plt.show()

# Save file
if SaveTxt:
    for infile in selected_files:
        wave_in, flux_in = read_file(infile)
        output_file = infile.split('/')[-1] + '_output.txt'
        np.savetxt(output_file, np.c_[wave_in, flux_in])
        print(f"Saved file: {output_file}")
