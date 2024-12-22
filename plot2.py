import matplotlib.pyplot as plt
import numpy as np
import astropy.io.fits as fits
from scipy import interpolate
from astropy.table import Table
import pandas as pd
import spectres
from astropy.convolution import Gaussian1DKernel
from astropy.convolution import convolve
from tkinter import Tk, filedialog

def read_file(infile, col0=0, col1=1, comment='#', SkipLines=0):
    ext = str(infile.split('.')[-1])

    if (ext == 'fits') or (ext ==  'fit'):
        wave, flux = read_fits(infile)
    elif (ext == 'gz'):
        wave, flux = read_tlusty(infile)
    elif (ext == 'dat' or ext == 'ascii' or ext == 'txt' or ext == 'nspec'):
        wave, flux = read_ascii(infile, col0=col0, col1=col1, comment=comment, SkipLines=SkipLines)
    elif (ext == 'tfits'):
        wave, flux = read_uvespop(infile)
    elif (ext == 'hfits'):
        wave, flux = read_hermes_normalized(infile)
    else:
        wave, flux = read_ascii(infile, col0=col0, col1=col1, comment=comment, SkipLines=SkipLines)
    return wave, flux

def read_ascii(infile, col0=0, col1=1, comment='#', SkipLines=0):
    print("%s: Input file is an ascii file." % infile)
    try:
        spec = (pd.read_csv(infile, header=None, delim_whitespace=True, comment=comment, skiprows=int(SkipLines))).values
    except:
        spec = (pd.read_csv(infile, header=None, delim_whitespace=True, comment=comment, skiprows=429)).values
    wave = spec[:,col0]
    flux = spec[:,col1]
    return wave, flux

# Define other helper functions (e.g., read_fits, read_tlusty, etc.) if necessary...

def select_files():
    root = Tk()
    root.withdraw()
    file_paths = filedialog.askopenfilenames(
        title="Select files to plot",
        filetypes=[("All files", "*.*"), ("Fits files", "*.fits"), ("ASCII files", "*.dat *.ascii *.txt")]
    )
    return list(file_paths)

fig, ax = plt.subplots()

Legend = True  # Enable legend by default
Binning = False
ScatterErr = False
Skip = False
SkipTwice = False
Norm = False
ReadMJD = False
Scatter = False
Degrade = False
SaveTxt = False
comment = '#'

# Open file dialog to select files
selected_files = select_files()

# If no files are selected, exit the script
if not selected_files:
    print("No files selected. Exiting.")
    sys.exit()

# Process the selected files
for infile in selected_files:
    if 'SkipLines' not in globals():
        SkipLines = 0
    col0, col1 = 0, 1  # Default column indices
    try:
        wave_in, flux_in = read_file(infile, col0=col0, col1=col1, comment=comment, SkipLines=SkipLines)
    except:
        wave_in, flux_in = read_file(infile, comment=comment, SkipLines=SkipLines)
    flux_in = np.nan_to_num(flux_in, 1.)
    try:
        wave_in = np.array(wave_in)
        flux_in = np.array(flux_in)
        wave_in = wave_in.astype(float)
        flux_in = flux_in.astype(float)
    except:
        wave_in, flux_in = np.loadtxt(infile, unpack=True)
    if len(flux_in) == 2:
        flux = flux_in[0]
        err = flux_in[1]
    else:
        flux = flux_in
    if Norm:
        flux /= np.mean(flux)
    if Binning:
        new_wavs = np.arange(wave_in[0], wave_in[-1], binsize)
        new_flux = spectres.spectres(new_wavs, wave_in, flux_in, spec_errs=None, fill=None, verbose=True)
        wave_in = np.copy(new_wavs)
        flux = np.copy(new_flux)
    if Degrade:
        lammid = np.average(wave_in)
        DlamRes = lammid/Resolving_Power
        new_waves = np.arange(wave_in[0], wave_in[-1], binsize)
        stdConv =  DlamRes/binsize /np.sqrt(2*np.log(2))/2.
        kernel = Gaussian1DKernel(stddev=stdConv)
        flux_binned = spectres.spectres(new_waves, wave_in, flux_in, spec_errs=None, fill=None, verbose=True)
        flux_degraded = convolve(flux_binned, kernel, normalize_kernel=True, boundary='extend')
        wave_in = np.copy(new_waves)
        flux = np.copy(flux_degraded)
    if ReadMJD:
        header = fits.getheader(infile)
        MJDObs = np.round(float(header['MJD-OBS']),1)
        name = MJDObs
    else:
        name = str(infile).split('.fits')[0]
        name = name.split('/')[-1]
    if ScatterErr:
        ax.errorbar(wave_in, flux, yerr=np.loadtxt(infile)[:,2], fmt='o', linewidth=1.0, alpha=0.8, label=name)
    elif Scatter:
        ax.scatter(wave_in, flux,  linewidth=1.0, alpha=0.8, label=name)
    else:
        ax.plot(wave_in, flux, linewidth=1.0, alpha=0.8, label=name)
if Legend:
    ax.legend()

# Adding lines based on prominence in O and B stars
# Wavelengths in angstroms
lines_o_stars = {
    'He II 3900': 3968,
    'He II 4200': 4199.83,
    'Hgamma/He II': 4340,
    'He II 4541': 4541,
}

lines_b_stars = {
    'Ca II 3934': 3965,   # Ca II H
    'He I 4010': 4010,    # Approximation
    'He I/II 4025': 4025, # Approximation
    'He I 4095': 4095,    # Approximation
    'Si IV 4089': 4089,   # Si IV and Hdelta/HeII group
    'Hdelta/He II': 4101, # Approximation for combined line
    'Si IV 4116': 4116,   # Si IV
    'He I 4385': 4385,    # Approximation
    'He I 4471': 4471,
    'Mg II 4481': 4481,
    'Si III 4550': 4550, # Approximation for combined line
}

lines_common = {
    'Hepsilon': 3970, # H_epsilon
}

# Plot the lines
for line_name, wavelength in lines_o_stars.items():
    ax.axvline(x=wavelength, color='blue', linestyle='--', linewidth=0.5)
    ax.text(wavelength, ax.get_ylim()[1], line_name, rotation=90, verticalalignment='bottom', fontsize=8, color='blue')

for line_name, wavelength in lines_b_stars.items():
    ax.axvline(x=wavelength, color='red', linestyle='--', linewidth=0.5)
    ax.text(wavelength, ax.get_ylim()[1], line_name, rotation=90, verticalalignment='bottom', fontsize=8, color='red')

for line_name, wavelength in lines_common.items():
    ax.axvline(x=wavelength, color='gray', linestyle='--', linewidth=0.5)
    ax.text(wavelength, ax.get_ylim()[1], line_name, rotation=90, verticalalignment='bottom', fontsize=8, color='gray')

ax.set_xlabel("Wavelength [A]")
ax.set_ylabel("Flux")
plt.show()

if SaveTxt:
    np.savetxt(infile + '.txt', np.c_[wave_in, flux])