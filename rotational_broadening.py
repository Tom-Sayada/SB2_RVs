import numpy as np
from scipy.ndimage import convolve1d


def apply_rotational_broadening(wave_array, flux_array, v_rot, epsilon=0.0):
    """
    Apply rotational broadening to a spectrum in linear wavelength scale.

    Parameters:
        wave_array (np.ndarray): Array of wavelengths in Angstroms
        flux_array (np.ndarray): Array of normalized flux values
        v_rot (float): Rotational velocity (v sin i) in km/s
        epsilon (float): Limb darkening coefficient (default is 0.0)

    Returns:
        np.ndarray: Rotationally broadened flux array
    """
    if v_rot <= 0:
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
    delta_log_lambda_vrot = v_rot / c

    # Calculate the number of points needed for the broadening kernel
    n_points = int(2 * delta_log_lambda_vrot / delta_log_lambda) + 1
    if n_points % 2 == 0:  # ensure odd number of points
        n_points += 1

    x = np.linspace(-v_rot, v_rot, n_points)
    kernel = np.zeros_like(x)

    # Calculate the rotational broadening kernel
    mask = np.abs(x) <= v_rot
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


# Example usage:
if __name__ == "__main__":
    # Example with a simple Gaussian spectral line
    wave = np.linspace(5000, 5010, 1000)  # wavelength array
    flux = 1.0 - 0.5 * np.exp(-(wave - 5005) ** 2 / 0.1)  # Gaussian line

    # Apply rotational broadening
    vsini = 30.0  # km/s
    broadened_flux = apply_rotational_broadening(wave, flux, vsini, epsilon=0.0)

    # Plot the results
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(wave, flux, 'b-', label='Original')
        plt.plot(wave, broadened_flux, 'r-', label=f'v sin i = {vsini} km/s')
        plt.xlabel('Wavelength (Å)')
        plt.ylabel('Normalized Flux')
        plt.legend()
        plt.grid(True)
        plt.show()
    except ImportError:
        print("Matplotlib not available for plotting")