import os
import re
import numpy as np
import pandas as pd
import logging
from typing import Tuple, Optional, Dict, List, Union
from astropy.io import fits
from scipy.special import wofz, erf
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
from tqdm import tqdm

# Configure logging
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


class DataQualityError(Exception):
    """Custom exception for data quality issues"""
    pass


def validate_spectral_data(wavelength: np.ndarray, flux: np.ndarray,
                           threshold: float = 0.1) -> bool:
    """
    Validate basic spectral data requirements.
    """
    if len(wavelength) != len(flux):
        raise ValueError("Wavelength and flux arrays must have the same length")
    if len(wavelength) == 0:
        raise ValueError("Empty wavelength array")

    # Check for finite values
    valid_mask = np.isfinite(wavelength) & np.isfinite(flux)
    valid_fraction = np.sum(valid_mask) / len(wavelength)
    if valid_fraction < (1 - threshold):
        raise DataQualityError(f"Too many invalid points: {1 - valid_fraction:.1%}")

    # Check wavelength spacing
    dw = np.diff(wavelength[valid_mask])
    if not np.all(dw > 0):
        raise ValueError("Wavelength array must be strictly increasing")

    # Check for reasonable flux values
    flux_valid = flux[valid_mask]
    if np.any(flux_valid < 0) or np.any(flux_valid > 2):
        logger.warning("Flux values outside expected range [0, 2]")

    return True


def robust_normalize_flux(flux: np.ndarray, percentile: float = 98) -> np.ndarray:
    """
    Normalize flux using robust statistics.
    """
    norm_value = np.percentile(flux[np.isfinite(flux)], percentile)
    if norm_value <= 0 or not np.isfinite(norm_value):
        raise DataQualityError("Invalid normalization value")
    return flux / norm_value


def estimate_noise_level(flux: np.ndarray, window_size: int = 20) -> float:
    """
    Estimate noise level using MAD.
    """
    if len(flux) < window_size:
        return np.std(flux)

    flux_series = pd.Series(flux)
    rolling_mad = flux_series.rolling(window=window_size, center=True).apply(
        lambda x: np.median(np.abs(x - np.median(x)))
    )

    noise_level = np.nanmedian(rolling_mad) * 1.4826
    return max(noise_level, 1e-4)


def load_fits_data(filepath: str) -> pd.DataFrame:
    """
    Load data from a FITS file with enhanced error handling.
    """
    try:
        with fits.open(filepath) as hdul:
            if len(hdul) < 2:
                logger.warning("FITS file has less than 2 HDUs, attempting text load")
                return load_text_data(filepath)

            # Try common column name variants
            wave_variants = ['WAVELENGTH', 'WAVE', 'LAMBDA', 'WAV']
            flux_variants = ['FLUX', 'SCI_NORM', 'FLUX_NORM', 'NORMALIZED_FLUX']

            wave_col = None
            flux_col = None

            # First try exact matches
            for hdu in hdul[1:]:
                if not hasattr(hdu, 'columns'):
                    continue

                col_names = [col.name.upper() for col in hdu.columns]

                for wave_name in wave_variants:
                    if wave_name in col_names:
                        wave_col = wave_name
                        break

                for flux_name in flux_variants:
                    if flux_name in col_names:
                        flux_col = flux_name
                        break

                if wave_col and flux_col:
                    break

            # Try partial matches if needed
            if not (wave_col and flux_col):
                for hdu in hdul[1:]:
                    if not hasattr(hdu, 'columns'):
                        continue
                    col_names = [col.name.upper() for col in hdu.columns]

                    for col in col_names:
                        if not wave_col and any(v in col for v in wave_variants):
                            wave_col = col
                        if not flux_col and any(v in col for v in flux_variants):
                            flux_col = col
                    if wave_col and flux_col:
                        break

            if not (wave_col and flux_col):
                logger.warning("Standard columns not found, attempting text load")
                return load_text_data(filepath)

            # Extract data
            wave = hdul[1].data[wave_col]
            flux = hdul[1].data[flux_col]

            # Handle byte-string columns
            if wave.dtype.kind in ('S', 'U'):
                wave = wave.astype(float)
            if flux.dtype.kind in ('S', 'U'):
                flux = flux.astype(float)

            validate_spectral_data(wave, flux)

            df = pd.DataFrame({'wavelength': wave, 'flux': flux})
            df = df.sort_values('wavelength')
            df = df.dropna()

            return df

    except Exception as e:
        logger.error(f"Error reading FITS file {filepath}: {str(e)}")
        try:
            return load_text_data(filepath)
        except Exception as e2:
            raise DataLoadError(f"Failed to load file in any format: {str(e2)}")


def load_text_data(filepath: str) -> pd.DataFrame:
    """
    Load data from text file with enhanced error handling.
    """
    try:
        df = None
        # Try different delimiters
        for delimiter in [None, ',', '\t', ';', '|']:
            for skiprows in [0, 1, 2, 3]:
                try:
                    df = pd.read_csv(filepath,
                                     sep=delimiter,
                                     skiprows=skiprows,
                                     engine='python')

                    numeric_cols = df.select_dtypes(include=[np.number]).columns
                    if len(numeric_cols) >= 2:
                        df = df[numeric_cols[:2]]
                        df.columns = ['wavelength', 'flux']
                        break
                except:
                    continue
            if df is not None and len(df) > 0:
                break

        if df is None or len(df) == 0:
            logger.warning(f"No valid data found in file {filepath}")
            return pd.DataFrame()

        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna()

        if len(df) > 0:
            df = df.sort_values('wavelength')
            validate_spectral_data(df['wavelength'].values, df['flux'].values)
            return df[['wavelength', 'flux']]

        return pd.DataFrame()

    except Exception as e:
        logger.error(f"Error reading text file {filepath}: {str(e)}")
        return pd.DataFrame()


def load_data_for_epoch(filepath: str) -> pd.DataFrame:
    """
    Load spectral data from any supported file format.
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
    Find observation files and sort by epoch number.
    """
    if not os.path.isdir(data_directory):
        raise DataLoadError(f"Invalid directory: {data_directory}")

    patterns = [
        (r'obs_(\d+)_V1_.*_V2_.*', lambda x: int(x)),  # e.g. obs_123_V1_400_V2_500
        (r'obs_(\d+)\.txt', lambda x: int(x)),  # e.g. obs_123.txt
        (r'obs_(\d+)_.*\.fit', lambda x: int(x)),  # e.g. obs_123_something.fit
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


def find_line_center_smoothed(wave_array: np.ndarray,
                              flux_array: np.ndarray,
                              sigma: float = 1.0,
                              absorption: bool = True) -> Optional[float]:
    """
    Find line center using smoothed data to handle noise better.
    """
    try:
        if len(wave_array) < 3:
            return None

        # Apply Gaussian smoothing
        smoothed = gaussian_filter1d(flux_array, sigma=sigma)

        if absorption:
            # Find minima
            peaks, properties = find_peaks(-smoothed,
                                           prominence=0.02,
                                           width=3)
        else:
            # Find maxima
            peaks, properties = find_peaks(smoothed,
                                           prominence=0.02,
                                           width=3)

        if len(peaks) == 0:
            return None

        # Get the deepest/highest peak
        if absorption:
            best_idx = peaks[np.argmin(smoothed[peaks])]
        else:
            best_idx = peaks[np.argmax(smoothed[peaks])]

        return wave_array[best_idx]

    except Exception as e:
        logger.error(f"Error finding line center: {str(e)}")
        return None


def find_noise_regions(data: pd.DataFrame,
                       line_min: float,
                       line_max: float,
                       noise_window: float = 8.0,
                       min_separation: float = 5.0,
                       max_offset: float = 30.0) -> List[Tuple[float, float, str]]:
    """
    Find noise (continuum) regions around spectral line.

    Args:
        data: DataFrame with wavelength and flux columns
        line_min, line_max: Line window boundaries
        noise_window: Width of noise sampling window
        min_separation: Minimum separation from line window
        max_offset: Maximum distance to search for noise

    Returns:
        List of (start_wave, end_wave, direction) tuples
    """

    def evaluate_region(wave_range: Tuple[float, float],
                        direction: str) -> Optional[Tuple[float, float, str]]:
        """Evaluate a potential noise region"""
        mask = ((data['wavelength'] >= wave_range[0]) &
                (data['wavelength'] <= wave_range[1]))

        if not np.any(mask):
            return None

        flux_segment = data['flux'][mask].values
        if len(flux_segment) < 5:  # Minimum points needed
            return None

        # Robust statistics
        median = np.median(flux_segment)
        mad = np.median(np.abs(flux_segment - median))
        sigma = 1.4826 * mad  # Estimate of standard deviation

        # Criteria for good continuum region:
        # 1. Median near 1.0
        # 2. Low variation
        # 3. No strong trends
        if (0.97 <= median <= 1.03 and mad < 0.02):
            # Check for trends
            x = data['wavelength'][mask].values
            y = flux_segment
            coeffs = np.polyfit(x - x.mean(), y - y.mean(), 1)
            slope = coeffs[0]

            if abs(slope) < 0.001:  # No strong trend
                return (wave_range[0], wave_range[1], direction)

        return None

    noise_regions = []

    # Search on both sides
    for direction, sign in [('left', -1), ('right', 1)]:
        offset = min_separation
        while offset <= max_offset:
            if direction == 'left':
                region_min = line_min + sign * (offset + noise_window)
                region_max = line_min + sign * offset
            else:
                region_min = line_max + sign * offset
                region_max = line_max + sign * (offset + noise_window)

            region = evaluate_region((region_min, region_max), direction)
            if region is not None:
                noise_regions.append(region)
                break

            offset += noise_window / 2  # Overlap windows by 50%

    return noise_regions


def voigt_profile(x: np.ndarray,
                  amplitude: float,
                  center: float,
                  sigma: float,
                  gamma: float) -> np.ndarray:
    """
    Compute Voigt profile with enhanced numerical stability.
    """
    try:
        sigma = max(sigma, min_width)
        gamma = max(gamma, min_width)

        # Scale to avoid overflow
        z = ((x - center) + 1j * gamma) / (sigma * np.sqrt(2))
        z_clipped = np.clip(z.real, -100, 100) + 1j * np.clip(z.imag, -100, 100)

        profile = np.real(wofz(z_clipped)) / (sigma * np.sqrt(2 * np.pi))

        return amplitude * profile

    except Exception as e:
        logger.error(f"Error computing Voigt profile: {str(e)}")
        raise


def skewed_voigt_profile(x: np.ndarray,
                         amplitude: float,
                         center: float,
                         sigma: float,
                         gamma: float,
                         skew: float) -> np.ndarray:
    """
    Compute skewed Voigt profile with enhanced handling of edge cases.
    """
    try:
        # Skip skewing for very small skew values
        if abs(skew) < 1e-10:
            return voigt_profile(x, amplitude, center, sigma, gamma)

        base_profile = voigt_profile(x, amplitude, center, sigma, gamma)

        # Compute skewing factor with bounds checking
        skew_arg = skew * (x - center)
        # Clip to avoid numerical issues
        skew_arg = np.clip(skew_arg, -100, 100)
        skew_factor = 1 + erf(skew_arg)

        return base_profile * skew_factor

    except Exception as e:
        logger.error(f"Error computing skewed Voigt profile: {str(e)}")
        raise