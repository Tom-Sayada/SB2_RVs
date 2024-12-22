import os
import re
import numpy as np
import pandas as pd
from scipy.special import wofz, erf
from astropy.io import fits

c = 299792.458  # speed of light in km/s
min_width = 1e-5

def doppler_shift(wavelength, rv):
    return wavelength * (1 + rv / c)

def voigt_profile(x, amplitude, center, sigma, gamma):
    sigma = max(sigma, min_width)
    gamma = max(gamma, min_width)
    z = ((x - center) + 1j*gamma) / (sigma*np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma*np.sqrt(2*np.pi))

def skewed_voigt_profile(x, amplitude, center, sigma, gamma, skew):
    """
    Error-function based skew:
    Skewed_Voigt(x) = Voigt(x)*[1 + erf(skew*(x-center))]
    """
    sigma = max(sigma, min_width)
    gamma = max(gamma, min_width)
    z = ((x - center) + 1j*gamma) / (sigma*np.sqrt(2))
    base_voigt = amplitude * np.real(wofz(z)) / (sigma*np.sqrt(2*np.pi))

    skew_factor = 1 + erf(skew * (x - center))
    return base_voigt * skew_factor

def find_observation_files(data_directory):
    """
    Find observation files in data_directory with two patterns:
    1) Old ASCII pattern:
       obs_(\d+)_V1_.*_V2_.*.txt or obs_(\d+)_.*.txt or obs_(\d+)_.*.fit
       Extract epoch from group(1).
    2) New FITS pattern:
       BLOeM_F-NNN_EE_Combined.fits
       Extract epoch from group(3).
    Returns a list of (epoch, filepath).
    """
    patterns = [
        # Old patterns (ASCII or .fit):
        (r'obs_(\d+)_V1_.*_V2_.*', lambda x: int(x)),
        (r'obs_(\d+)\.txt', lambda x: int(x)),
        (r'obs_(\d+)_.*\.fit', lambda x: int(x)),
        # New FITS pattern:
        (r'BLOeM_(\d+)-(\d+)_(\d+)_Combined\.fits', lambda x: int(x))
    ]

    epoch_files = []
    for filename in os.listdir(data_directory):
        for pattern, epoch_extractor in patterns:
            match = re.match(pattern, filename)
            if match:
                # Decide which match group to use for "epoch"
                if 'BLOeM_' in filename and '_Combined.fits' in filename:
                    # new pattern => group(3) is epoch
                    epoch_str = match.group(3)
                else:
                    # old pattern => group(1)
                    epoch_str = match.group(1)

                epoch = epoch_extractor(epoch_str)
                filepath = os.path.join(data_directory, filename)
                epoch_files.append((epoch, filepath))
                break
    epoch_files.sort(key=lambda x: x[0])
    return epoch_files

def load_data_for_epoch(filepath):
    """
    Load observation data.
    If the file ends with .fits, try reading as FITS and
    expect columns 'WAVELENGTH' and 'SCI_NORM' (or adjust to your columns).
    Otherwise, treat as ASCII data (two columns: wavelength, flux).
    """
    if filepath.lower().endswith('.fits'):
        from astropy.io import fits
        try:
            with fits.open(filepath) as hdul:
                if len(hdul) < 2:
                    print(f"FITS file {filepath} has no extension with data. Falling back to ASCII.")
                    data = pd.read_csv(filepath, sep=r'\s+', names=['wavelength', 'flux'])
                    return data

                cols = hdul[1].columns.names
                print(f"Columns in {filepath}: {cols}")

                # Adjust these column names if your FITS has different ones
                if 'WAVELENGTH' in cols and 'SCI_NORM' in cols:
                    wave = hdul[1].data['WAVELENGTH']
                    flux = hdul[1].data['SCI_NORM']
                    data = pd.DataFrame({'wavelength': wave, 'flux': flux})
                    return data
                else:
                    print(f"WAVELENGTH/SCI_NORM not found in columns {cols} for {filepath}")
                    print("Falling back to ASCII reading.")
                    data = pd.read_csv(filepath, sep=r'\s+', names=['wavelength', 'flux'])
                    return data
        except OSError as e:
            print(f"Error reading FITS {filepath}: {e}")
            print("Falling back to ASCII.")
            data = pd.read_csv(filepath, sep=r'\s+', names=['wavelength', 'flux'])
            return data
        except UnicodeDecodeError as e:
            print(f"Unicode decode error reading {filepath} as ASCII fallback: {e}")
            raise
    else:
        # ASCII file
        data = pd.read_csv(filepath, sep=r'\s+', names=['wavelength', 'flux'])
        return data

def find_noise_regions(data, min_w, max_w, noise_window=5, min_noise_separation=5, max_noise_offset=100):
    def search_direction(direction):
        offset = min_noise_separation
        step = 1
        while offset <= max_noise_offset:
            if direction=='left':
                noise_center = min_w - offset
                if noise_center + noise_window/2 >= min_w:
                    offset += step
                    continue
            else:
                noise_center = max_w + offset
                if noise_center - noise_window/2 <= max_w:
                    offset += step
                    continue

            nm = noise_center - noise_window/2
            nM = noise_center + noise_window/2
            mask = (data['wavelength']>=nm)&(data['wavelength']<=nM)
            nf = data['flux'][mask].values
            if nf.size>0:
                meanf = np.mean(nf)
                stdf = np.std(nf)
                if abs(meanf-1.0)<0.02 and stdf>0:
                    return nm, nM, direction
            offset+=step
        return None

    left_res = search_direction('left')
    right_res = search_direction('right')
    noise_regs = []
    if left_res: noise_regs.append(left_res)
    if right_res: noise_regs.append(right_res)
    return noise_regs


def find_line_center(wave_array, flux_array, absorption=True):
    """
    Find the line center by locating the minimum flux (for absorption).
    If absorption=True, np.argmin.
    If absorption=False, might do np.argmax for emission lines.
    """
    if len(wave_array) == 0:
        return None
    if absorption:
        min_idx = np.argmin(flux_array)
    else:
        min_idx = np.argmax(flux_array)
    return wave_array[min_idx]
