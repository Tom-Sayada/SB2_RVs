"""
Utility functions for SB2 spectral analysis project.
Includes:
 - File I/O (FITS/text) with robust error handling
 - Wavelength/flux validation
 - Finding observation files
 - Doppler shifting
 - Voigt, Gaussian, and skewed profile variants
 - Noise region detection
 - Optional line-center-finding with smoothing (find_line_center_smoothed)
"""

import os
import re
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, Dict, List
from astropy.io import fits
from scipy.special import wofz, erf
from scipy.ndimage import gaussian_filter1d  # for smoothing in line-center detection

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
c = 299792.458  # Speed of light in km/s
min_width = 1e-5
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


def gaussian_profile(x: np.ndarray, amplitude: float, center: float,
                     sigma: float) -> np.ndarray:
    """
    Compute a Gaussian profile.
    amplitude < 0 => absorption line

    Args:
        x: Wavelength array
        amplitude: Profile amplitude (negative for absorption)
        center: Profile center wavelength
        sigma: Gaussian width parameter
    Returns:
        Profile flux values
    """
    try:
        sigma = max(sigma, min_width)
        return amplitude * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))
    except Exception as e:
        logger.error(f"Error in gaussian_profile: {str(e)}")
        raise


def skewed_gaussian_profile(x: np.ndarray, amplitude: float, center: float,
                            sigma: float, skew: float) -> np.ndarray:
    """
    Compute a skewed Gaussian profile by multiplying by (1 + erf(skew*(x-center))).
    amplitude < 0 => absorption line

    Args:
        x: Wavelength array
        amplitude: Profile amplitude (negative for absorption)
        center: Profile center wavelength
        sigma: Gaussian width parameter
        skew: Skewness parameter
    Returns:
        Profile flux values
    """
    try:
        sigma = max(sigma, min_width)
        base_gaussian = amplitude * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))
        skew_factor = 1 + erf(skew * (x - center))
        return base_gaussian * skew_factor
    except Exception as e:
        logger.error(f"Error in skewed_gaussian_profile: {str(e)}")
        raise


def voigt_profile(x: np.ndarray, amplitude: float, center: float,
                  sigma: float, gamma: float) -> np.ndarray:
    """
    Compute a Voigt profile using the Faddeeva function (wofz).
    amplitude < 0 => absorption line
    """
    try:
        sigma = max(sigma, min_width)
        gamma = max(gamma, min_width)
        z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
        return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
    except Exception as e:
        logger.error(f"Error in voigt_profile: {str(e)}")
        raise


def skewed_voigt_profile(x: np.ndarray, amplitude: float, center: float,
                         sigma: float, gamma: float, skew: float) -> np.ndarray:
    """
    Compute a skewed Voigt profile by multiplying by (1 + erf(skew*(x-center))).
    amplitude < 0 => absorption line
    """
    try:
        sigma = max(sigma, min_width)
        gamma = max(gamma, min_width)
        z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
        base_voigt = amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))
        skew_factor = 1 + erf(skew * (x - center))
        return base_voigt * skew_factor
    except Exception as e:
        logger.error(f"Error in skewed_voigt_profile: {str(e)}")
        raise


def load_fits_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a FITS file with comprehensive error handling.
    If the file is not valid or the necessary columns are not found,
    falls back to attempting text-file load.
    """
    try:
        with fits.open(filepath) as hdul:
            if len(hdul) < 2:
                logger.warning("FITS file has less than 2 HDUs, attempting to read as text")
                return load_text_data(filepath)

            # Try common column name variants
            wave_variants = ['WAVELENGTH', 'WAVE', 'LAMBDA', 'WAV']
            flux_variants = ['FLUX', 'SCI_NORM', 'FLUX_NORM', 'NORMALIZED_FLUX']

            wave_col = None
            flux_col = None

            for wave_name in wave_variants:
                if wave_name in hdul[1].columns.names:
                    wave_col = wave_name
                    break

            for flux_name in flux_variants:
                if flux_name in hdul[1].columns.names:
                    flux_col = flux_name
                    break

            if wave_col is None or flux_col is None:
                logger.warning("Standard column names not found in FITS, attempting to read as text")
                return load_text_data(filepath)

            wave = hdul[1].data[wave_col]
            flux = hdul[1].data[flux_col]

            validate_spectral_data(wave, flux)
            return pd.DataFrame({'wavelength': wave, 'flux': flux})

    except Exception as e:
        logger.error(f"Error reading FITS file {filepath}: {str(e)}")
        try:
            return load_text_data(filepath)
        except:
            raise DataLoadError(f"Failed to load file in any format: {str(e)}")


def load_text_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a text file with multiple attempts at delimiter parsing,
    numeric conversion, and basic validation.
    Returns an empty DataFrame if no valid data is found.
    """
    try:
        # Try different delimiters
        for delimiter in [r'\s+', ',', '\t', ';']:
            try:
                # Read as strings first to avoid conversion errors
                data = pd.read_csv(filepath, sep=delimiter, header=None,
                                   engine='python', dtype=str)

                if len(data.columns) >= 2:
                    # Try converting to numeric, replace errors with NaN
                    data = data.apply(pd.to_numeric, errors='coerce')

                    # Check if we have any valid numeric data
                    if data.iloc[:, 0].notna().any() and data.iloc[:, 1].notna().any():
                        # Drop rows with NaN in first two columns
                        data = data.dropna(subset=[0, 1])
                        if len(data) > 0:
                            # Rename columns
                            data.columns = ['wavelength', 'flux'] + [
                                f'col_{i}' for i in range(2, len(data.columns))
                            ]
                            wave = data['wavelength'].values
                            flux = data['flux'].values

                            try:
                                validate_spectral_data(wave, flux)
                                return data[['wavelength', 'flux']]
                            except ValueError as ve:
                                logger.debug(f"Data validation failed: {str(ve)}")
                                continue

            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                continue

        logger.warning(f"No valid data found in file {filepath}")
        return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error reading text file {filepath}: {str(e)}")
        return pd.DataFrame()


def load_data_for_epoch(filepath: str) -> pd.DataFrame:
    """
    Main function to load spectral data from any supported file format (.fits, .txt, .dat).
    Raises DataLoadError or returns an empty DataFrame if unsuccessful.
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
        return pd.DataFrame()  # Return empty DataFrame instead of raising error
    except Exception as e:
        logger.error(f"Unexpected error loading {filepath}: {str(e)}")
        return pd.DataFrame()


def find_observation_files(data_directory: str) -> List[Tuple[int, str]]:
    """
    Search for observation files in the given directory that match certain naming patterns
    (e.g., obs_123.txt, obs_999.fit, etc.). Return a list of (epoch_number, filepath) pairs,
    sorted by epoch.
    """
    if not os.path.isdir(data_directory):
        raise DataLoadError(f"Invalid directory: {data_directory}")

    patterns = [
        (r'obs_(\d+)_V1_.*_V2_.*', lambda x: int(x)),  # e.g. obs_123_V1_400_V2_500
        (r'obs_(\d+)\.txt', lambda x: int(x)),  # e.g. obs_123.txt
        (r'obs_(\d+)_.*\.fit', lambda x: int(x)),  # e.g. obs_123_something.fit
        (r'BLOeM_(\d+)-(\d+)_(\d+)_Combined\.fits', lambda x: int(x)),  # specialized pattern
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

    # Sort by epoch number
    epoch_files.sort(key=lambda x: x[0])
    logger.info(f"Found {len(epoch_files)} valid observation files in {data_directory}")

    return epoch_files


def doppler_shift(wavelength: np.ndarray, rv: float) -> np.ndarray:
    """
    Simple Doppler shift: shift wavelength by rv (km/s).
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
    Attempt to locate continuum 'noise' regions on both sides of the line range [line_min, line_max].
    Returns a list of (n_min, n_max, direction).

    Args:
        data: A DataFrame with columns 'wavelength' and 'flux'.
        line_min, line_max: The boundaries of the line window.
        noise_window: Width (in Å) of the noise sampling window.
        min_noise_separation: Minimum separation from the line window to the noise region.
        max_noise_offset: Maximum distance from the line window to search for noise regions.

    The function tries to find a small continuum region on each side of the line window
    that has flux ~1 with small std. If not found, returns an empty list or partial results.
    """

    def search_direction(direction):
        offset = min_noise_separation
        step = 1.0
        while offset <= max_noise_offset:
            if direction == 'left':
                # For left side, center = line_min - offset
                noise_center = line_min - (offset + noise_window / 2)
                # Make sure it doesn't overlap the line
                if (noise_center + noise_window) >= (line_min - min_noise_separation):
                    offset += step
                    continue
            else:
                # For right side, center = line_max + offset
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
                # A simplistic check for continuum ~1
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
    A simple function that returns the wavelength at which flux is min (for absorption)
    or max (for emission), using raw data. This can be susceptible to noise.
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


def find_line_center_smoothed(
        wave_array: np.ndarray,
        flux_array: np.ndarray,
        sigma: float = 1.0,
        absorption: bool = True
) -> Optional[float]:
    """
    Smooth the flux array with a small Gaussian filter
    and then locate the min (for absorption) or max (for emission).

    Args:
        wave_array: 1D array of wavelength values
        flux_array: 1D array of flux values
        sigma: Smoothing kernel width for gaussian_filter1d
        absorption: If True, locate a minimum. Otherwise, locate maximum.

    Returns:
        The wavelength of the line center, or None if invalid.
    """
    if len(wave_array) < 3:
        return None

    # Apply mild Gaussian smoothing
    sm_flux = gaussian_filter1d(flux_array, sigma=sigma)

    if absorption:
        idx = np.argmin(sm_flux)
    else:
        idx = np.argmax(sm_flux)

    if idx < 0 or idx >= len(wave_array):
        return None

    return wave_array[idx]