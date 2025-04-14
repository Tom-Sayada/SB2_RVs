# line_profiles.py

import numpy as np
from scipy.special import wofz, erf

_min_width = 1e-5

def gaussian_profile(x, amplitude, center, sigma):
    """
    amplitude < 0 => absorption line
    """
    sigma = max(sigma, _min_width)
    return amplitude * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))

def skewed_gaussian_profile(x, amplitude, center, sigma, skew):
    """
    amplitude < 0 => absorption line
    """
    sigma = max(sigma, _min_width)
    base_gauss = amplitude * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))
    skew_factor = 1 + erf(skew * (x - center))
    return base_gauss * skew_factor

def voigt_profile(x, amplitude, center, sigma, gamma):
    """
    amplitude < 0 => absorption line
    """
    sigma = max(sigma, _min_width)
    gamma = max(gamma, _min_width)
    z = ((x - center) + 1j*gamma) / (sigma * np.sqrt(2))
    return amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))

def skewed_voigt_profile(x, amplitude, center, sigma, gamma, skew):
    """
    amplitude < 0 => absorption line
    """
    sigma = max(sigma, _min_width)
    gamma = max(gamma, _min_width)
    z = ((x - center) + 1j*gamma) / (sigma * np.sqrt(2))
    base_voigt = amplitude * np.real(wofz(z)) / (sigma * np.sqrt(2*np.pi))
    skew_factor = 1 + erf(skew * (x - center))
    return base_voigt * skew_factor
