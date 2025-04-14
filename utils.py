# utils.py - Part 1

"""
Utility functions for SB2 spectral analysis project.
Includes:
 - File I/O (FITS/text) with robust error handling
 - Wavelength/flux validation
 - Finding observation files
 - Doppler shifting
 - Noise region detection
 - Optional line-center-finding with smoothing (find_line_center_smoothed)
 - Interactive line selection for manual window definition
"""

import os
import re
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, List, Dict
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
c = 299792.458  # Speed of light in km/s
VALID_EXTENSIONS = ('.fits', '.fit', '.txt', '.dat')


class DataLoadError(Exception):
    """Custom exception for data loading errors"""
    pass


def validate_spectral_data(wavelength: np.ndarray, flux: np.ndarray) -> bool:
    """
    Validate basic spectral data requirements.
    - Equal array length
    - Non-empty
    - Finite values
    - Strictly increasing wavelength
    """
    if len(wavelength) != len(flux):
        raise ValueError("Wavelength and flux arrays must have the same length")
    if len(wavelength) == 0:
        raise ValueError("Empty wavelength array")
    if not np.all(np.isfinite(wavelength)) or not np.all(np.isfinite(flux)):
        raise ValueError("Non-finite values found in data")
    if not np.all(np.diff(wavelength) > 0):
        raise ValueError("Wavelength array must be strictly increasing")
    return True


def load_fits_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a FITS file with comprehensive error handling.
    If not valid or columns not found, fallback to text-file load.
    Extracts MJD if available in the header.
    """
    try:
        with fits.open(filepath) as hdul:
            # Try to extract MJD from header
            mjd_val = None
            if len(hdul) > 0:
                for mjd_key in ['MJD-OBS', 'MJD_OBS', 'MJD', 'MJDMID', 'MJD_MID', 'JD-MID']:
                    if mjd_key in hdul[0].header:
                        mjd_val = hdul[0].header[mjd_key]
                        # Convert JD to MJD if needed
                        if mjd_key.startswith('JD') and mjd_val > 2400000:
                            mjd_val -= 2400000.5
                        logger.info(f"Found {mjd_key}={mjd_val} in FITS header")
                        break

            if len(hdul) < 2:
                logger.warning("FITS file has <2 HDUs, attempting text read")
                df = load_text_data(filepath)
                if mjd_val is not None:
                    df.attrs['MJD'] = mjd_val
                return df

            wave_variants = ['WAVELENGTH', 'WAVE', 'LAMBDA', 'WAV']
            flux_variants = ['FLUX', 'SCI_NORM', 'FLUX_NORM', 'NORMALIZED_FLUX']

            wave_col = None
            flux_col = None

            for wv in wave_variants:
                if wv in hdul[1].columns.names:
                    wave_col = wv
                    break

            for fv in flux_variants:
                if fv in hdul[1].columns.names:
                    flux_col = fv
                    break

            if wave_col is None or flux_col is None:
                logger.warning("No standard columns in FITS, fallback to text")
                df = load_text_data(filepath)
                if mjd_val is not None:
                    df.attrs['MJD'] = mjd_val
                return df

            wave = hdul[1].data[wave_col]
            flux = hdul[1].data[flux_col]

            # ** Convert wave & flux to little-endian float64 to avoid endianness issues **
            wave = np.asarray(wave, dtype='<f8')
            flux = np.asarray(flux, dtype='<f8')

            validate_spectral_data(wave, flux)
            df = pd.DataFrame({'wavelength': wave, 'flux': flux})
            if mjd_val is not None:
                df.attrs['MJD'] = mjd_val
            return df

    except Exception as e:
        logger.error(f"Error reading FITS file {filepath}: {str(e)}")
        try:
            return load_text_data(filepath)
        except:
            raise DataLoadError(f"Failed to load file in any format: {str(e)}")


def load_text_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a text file with multiple attempts (different delimiters),
    numeric conversion, and basic validation.
    Also tries to extract MJD from comment lines (e.g. # MJD = 12345.678).
    """
    try:
        # First scan for MJD in headers/comments
        mjd_val = None
        try:
            with open(filepath, 'r') as f:
                for i, line in enumerate(f):
                    if i > 20:  # Only check first 20 lines
                        break
                    if line.startswith('#') or line.startswith('!'):
                        # Look for MJD = X.XXX or MJD: X.XXX or MJD_OBS = X.XXX
                        mjd_match = re.search(r'MJD[_\-]?(?:OBS|MID)?[\s=:]+(\d+\.?\d*)', line, re.IGNORECASE)
                        if mjd_match:
                            mjd_val = float(mjd_match.group(1))
                            logger.info(f"Found MJD={mjd_val} in file header")
                            break
                        # Also check for JD (Julian Date)
                        jd_match = re.search(r'JD[_\-]?(?:OBS|MID)?[\s=:]+(\d+\.?\d*)', line, re.IGNORECASE)
                        if jd_match:
                            jd_val = float(jd_match.group(1))
                            if jd_val > 2400000:  # Reasonable JD value
                                mjd_val = jd_val - 2400000.5  # Convert JD to MJD
                                logger.info(f"Converted JD={jd_val} to MJD={mjd_val}")
                                break
        except Exception as e:
            logger.debug(f"Error scanning for MJD in text file: {str(e)}")

        # Now parse the data
        for delimiter in [r'\s+', ',', '\t', ';']:
            try:
                # read as strings first
                data = pd.read_csv(filepath, sep=delimiter, header=None,
                                   engine='python', dtype=str)

                if len(data.columns) >= 2:
                    data = data.apply(pd.to_numeric, errors='coerce')
                    if data.iloc[:, 0].notna().any() and data.iloc[:, 1].notna().any():
                        data = data.dropna(subset=[0, 1])
                        if len(data) > 0:
                            data.columns = ['wavelength', 'flux'] + [
                                f'col_{i}' for i in range(2, len(data.columns))
                            ]
                            wave = data['wavelength'].values
                            flux = data['flux'].values
                            try:
                                validate_spectral_data(wave, flux)
                                df = data[['wavelength', 'flux']]
                                if mjd_val is not None:
                                    df.attrs['MJD'] = mjd_val
                                return df
                            except ValueError as ve:
                                logger.debug(f"Data validation failed: {str(ve)}")
                                continue
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                continue

        logger.warning(f"No valid data found in file {filepath}")
        df = pd.DataFrame()
        if mjd_val is not None:
            df.attrs['MJD'] = mjd_val
        return df

    except Exception as e:
        logger.error(f"Error reading text file {filepath}: {str(e)}")
        return pd.DataFrame()


def load_data_for_epoch(filepath: str) -> pd.DataFrame:
    """
    Main function to load spectral data from .fits or .txt/.dat.
    Returns an empty DataFrame on failure.
    DataFrame will have attrs['MJD'] set if MJD was found in the file.
    """
    if not os.path.exists(filepath):
        raise DataLoadError(f"File not found: {filepath}")

    file_ext = os.path.splitext(filepath.lower())[1]
    if file_ext not in VALID_EXTENSIONS:
        raise DataLoadError(f"Unsupported file extension: {file_ext}")

    try:
        if file_ext in ('.fits', '.fit'):
            return load_fits_data(filepath)
        else:
            return load_text_data(filepath)
    except DataLoadError as e:
        logger.error(str(e))
        return pd.DataFrame()
    except Exception as e:
        logger.error(f"Unexpected error loading {filepath}: {str(e)}")
        return pd.DataFrame()


def find_observation_files(data_directory: str) -> List[Tuple[int, str]]:
    """
    Search for observation files in 'data_directory' that match patterns
    like obs_123_V1_400_V2_500.txt, or obs_123.txt, etc.
    Return list of (epoch_number, filepath) sorted by epoch.
    """
    if not os.path.isdir(data_directory):
        raise DataLoadError(f"Invalid directory: {data_directory}")

    patterns = [
        (r'obs_(\d+)_V1_.*_V2_.*', lambda x: int(x)),
        (r'obs_(\d+)\.txt', lambda x: int(x)),
        (r'obs_(\d+)_.*\.fit', lambda x: int(x)),
        # specialized pattern example:
        (r'BLOeM_(\d+)-(\d+)_(\d+)_Combined\.fits', lambda x: int(x)),
    ]

    epoch_files = []
    for filename in os.listdir(data_directory):
        filepath = os.path.join(data_directory, filename)
        if not os.path.isfile(filepath):
            continue
        if not any(filepath.lower().endswith(ext) for ext in VALID_EXTENSIONS):
            continue

        for pattern, extractor in patterns:
            match = re.match(pattern, filename)
            if match:
                try:
                    if 'BLOeM_' in filename and '_Combined.fits' in filename:
                        ep_str = match.group(3)
                    else:
                        ep_str = match.group(1)
                    ep_val = extractor(ep_str)
                    epoch_files.append((ep_val, filepath))
                    break
                except (IndexError, ValueError) as e:
                    logger.warning(f"Failed to extract epoch from {filename}: {str(e)}")
                    continue

    if not epoch_files:
        raise DataLoadError(f"No valid observation files found in {data_directory}")

    epoch_files.sort(key=lambda x: x[0])
    logger.info(f"Found {len(epoch_files)} valid observation files in {data_directory}")
    return epoch_files


def doppler_shift(wavelength: np.ndarray, rv: float) -> np.ndarray:
    """
    new_wavelength = wavelength * (1 + rv / c).
    """
    try:
        return wavelength * (1 + rv / c)
    except Exception as e:
        logger.error(f"Error in doppler_shift: {str(e)}")
        raise


def find_noise_regions(data: pd.DataFrame,
                       line_min: float,
                       line_max: float,
                       noise_window: float = 5,
                       min_noise_separation: float = 5,
                       max_noise_offset: float = 100) -> List[Tuple[float, float, str]]:
    """
    Attempt to find continuum/noise regions on each side of [line_min, line_max].
    Return a list of (n_min, n_max, direction).
    """

    def search_direction(direction):
        offset = min_noise_separation
        step = 1.0
        while offset <= max_noise_offset:
            if direction == 'left':
                noise_center = line_min - (offset + noise_window / 2)
                if (noise_center + noise_window) >= (line_min - min_noise_separation):
                    offset += step
                    continue
            else:  # right
                noise_center = line_max + (offset + noise_window / 2)
                if (noise_center - noise_window) <= (line_max + min_noise_separation):
                    offset += step
                    continue

            nm = noise_center - noise_window / 2
            nM = noise_center + noise_window / 2
            mask = (data['wavelength'] >= nm) & (data['wavelength'] <= nM)
            nf = data['flux'][mask].values

            if nf.size > 0:
                meanf = np.mean(nf)
                stdf = np.std(nf)
                if abs(meanf - 1.0) < 0.02 and stdf > 0:
                    return (nm, nM, direction)

            offset += step
        return None

    noise_regs = []
    left_found = search_direction('left')
    if left_found:
        noise_regs.append(left_found)
    right_found = search_direction('right')
    if right_found:
        noise_regs.append(right_found)

    return noise_regs


def find_line_center(wave_array: np.ndarray,
                     flux_array: np.ndarray,
                     absorption: bool = True) -> Optional[float]:
    """
    Return the wavelength of min (for absorption) or max (for emission).
    """
    try:
        if len(wave_array) == 0:
            return None
        if absorption:
            idx = np.argmin(flux_array)
        else:
            idx = np.argmax(flux_array)
        return wave_array[idx]
    except Exception as e:
        logger.error(f"Error finding line center: {str(e)}")
        return None


def find_line_center_smoothed(wave_array: np.ndarray,
                              flux_array: np.ndarray,
                              sigma: float = 1.0,
                              absorption: bool = True) -> Optional[float]:
    """
    Smooth flux with a small Gaussian filter, then locate min (absorption) or max (emission).
    """
    if len(wave_array) < 3:
        return None
    sm_flux = gaussian_filter1d(flux_array, sigma=sigma)
    if absorption:
        idx = np.argmin(sm_flux)
    else:
        idx = np.argmax(sm_flux)
    if idx < 0 or idx >= len(wave_array):
        return None
    return wave_array[idx]


# utils.py - Part 2

def interactive_line_selection(df: pd.DataFrame,
                               spectral_lines: Dict,
                               epoch: int) -> Dict:
    """
    Allow user to interactively select line windows for each spectral line.

    Parameters:
    -----------
    df : pandas DataFrame
        The spectral data with 'wavelength' and 'flux' columns
    spectral_lines : dict
        Dictionary of spectral lines info
    epoch : int
        The epoch number

    Returns:
    --------
    dict
        Updated spectral_lines dictionary with user-selected windows
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import SpanSelector
    except ImportError:
        logger.error("Matplotlib is required for interactive line selection")
        return spectral_lines

    updated_lines = {}
    for ln_name, ln_info in spectral_lines.items():
        updated_lines[ln_name] = ln_info.copy()  # Make a copy to avoid modifying the original

        restw = ln_info['rest_wave']

        # Show a wider region around the expected line
        search_extra = 50.0
        search_min = restw - search_extra
        search_max = restw + search_extra
        mask = (df['wavelength'] >= search_min) & (df['wavelength'] <= search_max)

        if not mask.any():
            logger.warning(f"No data for line {ln_name} at {restw} - skipping")
            continue

        wv = df['wavelength'][mask].values
        fl = df['flux'][mask].values

        fig, ax = plt.figure(figsize=(12, 6)), plt.subplot(111)
        ax.plot(wv, fl, 'b-')
        ax.set_title(f"Epoch {epoch} - Select window for {ln_name} (rest λ = {restw})")
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("Flux")
        ax.axvline(restw, color='r', linestyle='--', label="Rest wavelength")
        plt.grid(alpha=0.3)
        plt.legend()

        # Vertical lines to mark the current window
        center = restw
        current_half_window = ln_info['window'] / 2
        left_line = ax.axvline(center - current_half_window, color='g', linestyle='-')
        right_line = ax.axvline(center + current_half_window, color='g', linestyle='-')
        window_text = ax.text(0.02, 0.95, f"Window: {ln_info['window']:.1f} Å",
                              transform=ax.transAxes, color='g')

        # Store the selection
        selection = {'center': center, 'window': ln_info['window']}

        def update_lines(min_val, max_val):
            center = (min_val + max_val) / 2
            window = max_val - min_val
            selection['center'] = center
            selection['window'] = window

            left_line.set_xdata([min_val, min_val])
            right_line.set_xdata([max_val, max_val])
            window_text.set_text(f"Window: {window:.1f} Å")
            plt.draw()

        # Create SpanSelector
        span = SpanSelector(
            ax, update_lines, 'horizontal', useblit=True,
            props=dict(alpha=0.3, facecolor='green'),
            button=1, minspan=2.0
        )

        plt.tight_layout()
        plt.show()

        # Update the spectral_lines dict with user selection
        updated_lines[ln_name]['window'] = selection['window']
        updated_lines[ln_name]['center_offset'] = selection['center'] - restw  # Store the offset from rest wavelength

        logger.info(f"Selected window for {ln_name}: center at λ = {selection['center']:.2f} " +
                    f"(offset: {selection['center'] - restw:.2f} Å), width = {selection['window']:.2f} Å")

    return updated_lines


def plot_preview_spectrum(df: pd.DataFrame,
                          title: str = "Full Spectrum Preview",
                          figsize: Tuple[int, int] = (12, 6)) -> None:
    """
    Plot the full spectrum as a quick preview.

    Parameters:
    -----------
    df : pandas DataFrame
        The spectral data with 'wavelength' and 'flux' columns
    title : str
        Plot title
    figsize : tuple
        Figure size (width, height) in inches
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.error("Matplotlib is required for plotting")
        return

    if df.empty or 'wavelength' not in df.columns or 'flux' not in df.columns:
        logger.error("Invalid dataframe for plotting")
        return

    plt.figure(figsize=figsize)
    plt.plot(df['wavelength'], df['flux'], 'b-')
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Flux")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def save_line_windows_to_file(spectral_lines: Dict,
                              filename: str = "line_windows.json") -> None:
    """
    Save the selected line windows to a JSON file.

    Parameters:
    -----------
    spectral_lines : dict
        Dictionary of spectral lines info with window and center_offset
    filename : str
        Output JSON filename
    """
    import json

    try:
        with open(filename, 'w') as f:
            json.dump(spectral_lines, f, indent=2)
        logger.info(f"Line window definitions saved to {filename}")
    except Exception as e:
        logger.error(f"Error saving line windows to {filename}: {e}")


def load_line_windows_from_file(filename: str) -> Dict:
    """
    Load line window definitions from a JSON file.

    Parameters:
    -----------
    filename : str
        Input JSON filename

    Returns:
    --------
    dict
        Dictionary of spectral lines info
    """
    import json

    try:
        with open(filename, 'r') as f:
            spectral_lines = json.load(f)
        logger.info(f"Loaded line window definitions from {filename}")
        return spectral_lines
    except Exception as e:
        logger.error(f"Error loading line windows from {filename}: {e}")
        return {}