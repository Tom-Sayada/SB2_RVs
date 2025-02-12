import numpy as np
from lmfit import Parameters
import scipy.ndimage as ndimage
from scipy.signal import find_peaks
from src.utils import voigt_profile

c_light = 299792.458


def get_line_weight(wavelength: float) -> float:
    """
    Get the weight for a spectral line based on its properties.
    Higher weight means less importance in the fit.
    """
    # Hydrogen lines get higher weights (less importance)
    h_lines = [4340.472, 4101.734, 3970.075]  # Add other H lines as needed

    for h_wave in h_lines:
        if abs(wavelength - h_wave) < 1.0:  # Within 1Å of a H line
            return 2.0  # Hydrogen lines get half the weight

    return 1.0  # Default weight for other lines


def setup_parameters(
        central_wavelengths,
        all_epochs,
        profile_type='sym',
        fit_baseline=False,
        initial_rvs=None  # Add optional initial RV estimates
):
    params = Parameters()

    # Set sensible bounds and initial values based on line type
    for line_id, cwv in central_wavelengths.items():
        # Identify line type
        is_hydrogen = any(str(cwv).startswith(h) for h in ['4340', '4861', '6563'])

        if is_hydrogen:
            # Hydrogen lines - broader, deeper profiles
            params.add(f'a1_{line_id}', value=-1.0, min=-5.0, max=-0.1)
            params.add(f'sigma1_{line_id}', value=2.0, min=0.1, max=5.0)
            params.add(f'gamma1_{line_id}', value=2.0, min=0.1, max=5.0)
            params.add(f'a2_{line_id}', value=-0.8, min=-5.0, max=-0.1)
            params.add(f'sigma2_{line_id}', value=2.0, min=0.1, max=5.0)
            params.add(f'gamma2_{line_id}', value=2.0, min=0.1, max=5.0)
        else:
            # Helium lines - narrower profiles
            params.add(f'a1_{line_id}', value=-0.4, min=-2.0, max=-0.05)
            params.add(f'sigma1_{line_id}', value=0.8, min=0.1, max=3.0)
            params.add(f'gamma1_{line_id}', value=0.8, min=0.1, max=3.0)
            params.add(f'a2_{line_id}', value=-0.3, min=-2.0, max=-0.05)
            params.add(f'sigma2_{line_id}', value=0.8, min=0.1, max=3.0)
            params.add(f'gamma2_{line_id}', value=0.8, min=0.1, max=3.0)

        if fit_baseline:
            if is_hydrogen:
                params.add(f'baseline_{line_id}', value=1.0, min=0.95, max=1.05)
            else:
                params.add(f'baseline_{line_id}', value=1.0, min=0.98, max=1.02)

        if profile_type == 'asym' and is_hydrogen:
            params.add(f'skew1_{line_id}', value=0.0, min=-1.0, max=1.0)
            params.add(f'skew2_{line_id}', value=0.0, min=-1.0, max=1.0)

    # RV parameters with smarter initialization
    if initial_rvs is None:
        initial_rvs = {ep: (0.0, 0.0) for ep in all_epochs}

    for ep in all_epochs:
        e = int(ep)
        rv1_init, rv2_init = initial_rvs.get(e, (0.0, 0.0))

        params.add(f'rv1_epoch{e}',
                   value=rv1_init,
                   min=-500.0,
                   max=500.0)

        params.add(f'rv2_epoch{e}',
                   value=rv2_init,
                   min=-500.0,
                   max=500.0)

    return params


def find_initial_rvs(wv, flux, rest_wv, min_depth=0.02):
    """Simple function to find potential RVs from line positions"""
    # Smooth the flux a bit to reduce noise impact
    smooth_flux = ndimage.gaussian_filter1d(flux, sigma=1.0)

    # Find local minima
    minima, _ = find_peaks(-smooth_flux)

    rv_candidates = []
    for m in minima:
        depth = 1.0 - smooth_flux[m]
        if depth > min_depth:
            wv_at_min = wv[m]
            # Convert to RV
            rv = (wv_at_min / rest_wv - 1.0) * c_light
            if -500 <= rv <= 500:  # Within valid RV range
                rv_candidates.append(rv)

    return rv_candidates


def estimate_initial_rvs(wavelengths_dict, fluxes_dict, epochs_dict, central_wavelengths):
    """Estimate initial RVs for each epoch"""
    initial_rvs = {}
    all_epochs = np.unique(np.concatenate([ep for ep in epochs_dict.values()]))

    for ep in all_epochs:
        rv_candidates = []
        for line_id in wavelengths_dict:
            rest_wv = central_wavelengths[line_id]
            idx = (epochs_dict[line_id] == ep)

            if not np.any(idx):
                continue

            wv = wavelengths_dict[line_id][idx]
            flux = fluxes_dict[line_id][idx]

            candidates = find_initial_rvs(wv, flux, rest_wv)
            rv_candidates.extend(candidates)

        if len(rv_candidates) >= 2:
            # Simple clustering - split at median
            rv_array = np.array(rv_candidates)
            median = np.median(rv_array)
            group1 = rv_array[rv_array <= median]
            group2 = rv_array[rv_array > median]

            rv1_est = np.mean(group1) if len(group1) > 0 else -50.0
            rv2_est = np.mean(group2) if len(group2) > 0 else 50.0

            initial_rvs[ep] = (rv1_est, rv2_est)
        else:
            initial_rvs[ep] = (0.0, 0.0)

    return initial_rvs


def grid_search_rvs(wv, flux, rest_wv, rv_range=(-500, 500), coarse_points=15, fine_points=15):
    """
    Enhanced grid search with physical constraints for real data.
    """
    # Normalize and prepare data
    flux = flux / np.median(flux[np.argsort(flux)[len(flux) // 2 - 5:len(flux) // 2 + 5]])
    min_flux = np.min(flux)
    min_loc = wv[np.argmin(flux)]

    # Find strongest absorption feature
    main_rv = (min_loc / rest_wv - 1.0) * 299792.458

    # Create asymmetric grids around main feature
    rv_grid1 = np.linspace(main_rv - 200, main_rv + 200, coarse_points)
    rv_grid2 = np.linspace(main_rv - 300, main_rv + 300, coarse_points)

    best_chisq = np.inf
    best_rvs = (0, 0)

    # More realistic line parameters for real data
    depth_ratios = [(0.6, 0.4), (0.7, 0.3), (0.5, 0.5)]  # Different component ratios
    widths = [0.8, 1.0, 1.2]  # Multiple widths to try

    # Coarse grid search with physical constraints
    for rv1 in rv_grid1:
        for rv2 in rv_grid2:
            # Physical constraints
            if abs(rv1 - rv2) < 30:  # Minimum RV separation
                continue
            if abs(rv1 - rv2) > 400:  # Maximum RV separation
                continue

            for depth_ratio in depth_ratios:
                for width in widths:
                    # Model components
                    shifted1 = wv * (1 + rv1 / 299792.458)
                    shifted2 = wv * (1 + rv2 / 299792.458)

                    # Deeper component
                    depth1 = min(0.4 * depth_ratio[0], 0.6)
                    model1 = 1.0 - depth1 * (1 / (1 + ((wv - shifted1) / (width)) ** 2))

                    # Shallower component
                    depth2 = min(0.4 * depth_ratio[1], 0.6)
                    model2 = 1.0 - depth2 * (1 / (1 + ((wv - shifted2) / (width)) ** 2))

                    # Combined model with proper blending
                    model = model1 * model2

                    # Weighted chi-square calculation
                    residuals = flux - model
                    weights = 1.0 / (flux + 0.1)  # More weight to absorption cores
                    chisq = np.sum(weights * residuals ** 2)

                    if chisq < best_chisq:
                        best_chisq = chisq
                        best_rvs = (rv1, rv2)

    # Fine grid around best result
    rv1_best, rv2_best = best_rvs
    fine_range = 30.0  # Smaller range for fine search

    fine_grid1 = np.linspace(rv1_best - fine_range, rv1_best + fine_range, fine_points)
    fine_grid2 = np.linspace(rv2_best - fine_range, rv2_best + fine_range, fine_points)

    # Fine search with best parameters from coarse grid
    for rv1 in fine_grid1:
        for rv2 in fine_grid2:
            if abs(rv1 - rv2) < 30 or abs(rv1 - rv2) > 400:
                continue

            shifted1 = wv * (1 + rv1 / 299792.458)
            shifted2 = wv * (1 + rv2 / 299792.458)

            model1 = 1.0 - depth1 * (1 / (1 + ((wv - shifted1) / (width)) ** 2))
            model2 = 1.0 - depth2 * (1 / (1 + ((wv - shifted2) / (width)) ** 2))
            model = model1 * model2

            residuals = flux - model
            weights = 1.0 / (flux + 0.1)
            chisq = np.sum(weights * residuals ** 2)

            if chisq < best_chisq:
                best_chisq = chisq
                best_rvs = (rv1, rv2)

    return sorted(best_rvs)  # Returns (more negative, more positive)

def residuals(
        params,
        wavelengths_dict,
        fluxes_dict,
        epochs_dict,
        uncertainties_dict,
        central_wavelengths,
        profile_type='sym',
        weighted=True,
        fit_baseline=False
):
    from src.model_builder import compute_full_model

    model_flux = compute_full_model(
        params,
        wavelengths_dict,
        epochs_dict,
        central_wavelengths,
        profile_type=profile_type
    )

    all_residuals = []
    for line_id in wavelengths_dict:
        if len(wavelengths_dict[line_id]) == 0:
            continue

        obs = fluxes_dict[line_id]
        mod = model_flux[line_id]
        unc = uncertainties_dict[line_id]

        # Apply line-specific weighting
        weight = get_line_weight(central_wavelengths[line_id])
        unc = unc * weight  # Increase uncertainty for H lines

        if weighted:
            all_residuals.append((obs - mod) / unc)
        else:
            all_residuals.append(obs - mod)

    return np.concatenate(all_residuals)
