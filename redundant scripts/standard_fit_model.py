#!/usr/bin/env python3
# standard_fit_model.py

"""
Implements a "standard" SB2 model using Voigt profiles (by default) with optional baseline
and possible initial-guess logic. Also provides a 'residuals' function that can apply
Gaussian or Lorentzian weighting near line cores, if desired.

This file is typically used by older standard-fitting routines; if your main code
calls model_builder.py's residuals, you may not need this. Otherwise, copy/paste
this as a fully functional script for standard SB2 fits with line-core weighting.
"""

import numpy as np
from lmfit import Parameters
import scipy.ndimage as ndimage
from scipy.signal import find_peaks
from src.utils import voigt_profile, min_width
from src.model_builder import compute_full_model, _compute_rv1_rv2

c_light = 299792.458


def get_line_weight(wavelength: float) -> float:
    """
    Return a per-line scaling factor.
    Example: If near certain hydrogen lines, increase the uncertainty
    (thus weighting them less). Otherwise return 1.0.

    Adjust as needed for your science.
    """
    h_lines = [4340.472, 4101.734, 3970.075]  # Add other H lines as needed

    for h_wave in h_lines:
        # Within 1Å of a H line => weight=2.0 means we double the uncertainty,
        # effectively reducing its importance by factor ~2.
        if abs(wavelength - h_wave) < 1.0:
            return 2.0

    return 1.0


def setup_parameters(
        central_wavelengths,
        all_epochs,
        profile_type='sym',
        fit_baseline=False,
        initial_rvs=None
):
    """
    Build an lmfit.Parameters() for a standard SB2 fit (two Voigt components per line).
    If 'profile_type' is 'asym' for hydrogen lines, it allows skew1/skew2.

    If 'initial_rvs' is provided, it should be a dict: { epoch -> (rv1_init, rv2_init) }.
    """
    params = Parameters()

    # Set up line shape parameters
    for line_id, cwv in central_wavelengths.items():
        # Identify if it's a hydrogen line
        is_hydrogen = any(str(cwv).startswith(h) for h in ['4340', '4861', '6563'])

        if is_hydrogen:
            # broader lines
            params.add(f'a1_{line_id}', value=-1.0, min=-5.0, max=-0.1)
            params.add(f'sigma1_{line_id}', value=2.0, min=0.1, max=5.0)
            params.add(f'gamma1_{line_id}', value=2.0, min=0.1, max=5.0)
            params.add(f'a2_{line_id}', value=-0.8, min=-5.0, max=-0.1)
            params.add(f'sigma2_{line_id}', value=2.0, min=0.1, max=5.0)
            params.add(f'gamma2_{line_id}', value=2.0, min=0.1, max=5.0)
        else:
            # helium or other
            params.add(f'a1_{line_id}', value=-0.4, min=-2.0, max=-0.05)
            params.add(f'sigma1_{line_id}', value=0.8, min=0.1, max=3.0)
            params.add(f'gamma1_{line_id}', value=0.8, min=0.1, max=3.0)
            params.add(f'a2_{line_id}', value=-0.3, min=-2.0, max=-0.05)
            params.add(f'sigma2_{line_id}', value=0.8, min=0.1, max=3.0)
            params.add(f'gamma2_{line_id}', value=0.8, min=0.1, max=3.0)

        # optional baseline
        if fit_baseline:
            if is_hydrogen:
                params.add(f'baseline_{line_id}', value=1.0, min=0.95, max=1.05)
            else:
                params.add(f'baseline_{line_id}', value=1.0, min=0.98, max=1.02)

        # optional skew if 'asym' profile for hydrogen
        if profile_type == 'asym' and is_hydrogen:
            params.add(f'skew1_{line_id}', value=0.0, min=-1.0, max=1.0)
            params.add(f'skew2_{line_id}', value=0.0, min=-1.0, max=1.0)

    # Possibly use user-provided initial RVs
    if initial_rvs is None:
        initial_rvs = {ep: (0.0, 0.0) for ep in all_epochs}

    # Add rv1_epoch, rv2_epoch
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
    """
    A simple function to guess the radial velocity by finding
    local minima in a smoothed flux array.
    We require the line depth > min_depth to consider it a real line.

    Returns a list of candidate RVs.
    """
    # Gaussian-smooth the flux
    smooth_flux = ndimage.gaussian_filter1d(flux, sigma=1.0)

    # Find local minima
    minima_idx, _ = find_peaks(-smooth_flux)

    rv_candidates = []
    for m in minima_idx:
        if m < 0 or m >= len(smooth_flux):
            continue

        depth = 1.0 - smooth_flux[m]  # how deep below continuum ~1
        if depth >= min_depth:
            wv_min = wv[m]
            # convert to RV = (obs/rest - 1)*c
            rv = (wv_min / rest_wv - 1.0) * c_light
            if -500 <= rv <= 500:
                rv_candidates.append(rv)

    return rv_candidates


def estimate_initial_rvs(wavelengths_dict, fluxes_dict, epochs_dict, central_wavelengths):
    """
    Example approach to find roughly where the lines are for each epoch:
     - For each line, we find local minima => convert to RV
     - We cluster them around two main peaks => star1, star2
     - Return a dict { epoch: (rv1, rv2) } for each epoch
    """
    initial_rvs = {}
    all_ep = set()
    for lid in epochs_dict:
        all_ep.update(epochs_dict[lid].tolist())
    all_ep = sorted(all_ep)

    for ep in all_ep:
        rv_candidates = []
        for line_id in wavelengths_dict:
            idx = (epochs_dict[line_id] == ep)
            if not np.any(idx):
                continue

            wv = wavelengths_dict[line_id][idx]
            fl = fluxes_dict[line_id][idx]
            restwv = central_wavelengths[line_id]
            cands = find_initial_rvs(wv, fl, restwv, min_depth=0.02)
            rv_candidates.extend(cands)

        if len(rv_candidates) < 2:
            # fallback
            initial_rvs[ep] = (0.0, 0.0)
        else:
            rv_array = np.array(rv_candidates)
            median_rv = np.median(rv_array)
            group1 = rv_array[rv_array <= median_rv]
            group2 = rv_array[rv_array > median_rv]

            rv1_est = np.mean(group1) if len(group1) else -50.0
            rv2_est = np.mean(group2) if len(group2) else 50.0
            initial_rvs[ep] = (rv1_est, rv2_est)

    return initial_rvs


def grid_search_rvs(wv, flux, rest_wv, rv_range=(-500, 500), coarse_points=15, fine_points=15):
    """
    A more advanced grid-search approach for RV,
    typically not used if we have good initial guesses.

    Returns sorted rv1, rv2
    """
    # Example placeholder logic (some line-shape iteration).
    # In practice, you'd do a real grid search.
    # We'll keep a minimal version that just returns (0,0).
    return (0.0, 0.0)


##################################
# NEW: Gaussian/Lorentz weighting
##################################
def _peak_weight_gaussian(wv, centers, amp, width):
    """
    Build a weighting array = 1 + amp * sum( Gaussian( (wv-c_j)/width ) ).
    If amp=0 => returns all ones => no weighting.
    """
    w = np.ones_like(wv)
    if amp <= 0 or width <= 0:
        return w
    for c in centers:
        dist = w - c
        dist = wv - c  # correction above
        w += amp * np.exp(-0.5 * (dist / width)**2)
    return w


def _peak_weight_lorentz(wv, centers, amp, width):
    """
    Build weighting array = 1 + amp * sum( Lorentz( (wv-c_j)/width ) ).
    If amp=0 => returns all ones => no weighting.
    """
    w = np.ones_like(wv)
    if amp <= 0 or width <= 0:
        return w
    for c in centers:
        dist = wv - c
        w += amp / (1.0 + (dist / width)**2)
    return w


def residuals(
        params,
        wavelengths_dict,
        fluxes_dict,
        epochs_dict,
        uncertainties_dict,
        central_wavelengths,
        profile_type='sym',
        weighted=True,
        fit_baseline=False,
        # Additional arguments for line-core weighting:
        peak_weight_amp=0.0,
        peak_weight_width=1.0,
        peak_weight_kind='gaussian'
):
    """
    The standard approach residual function. Uses compute_full_model(...) from
    src.model_builder, then optionally applies a "peak weighting" near star1, star2 line centers
    for each epoch.

    By default, peak_weight_amp=0 => no extra weighting, so code reverts to old uniform logic.
    """
    # Build the model
    model_flux = compute_full_model(
        params,
        wavelengths_dict,
        epochs_dict,
        central_wavelengths,
        profile_type=profile_type
        # (In your code, you might also specify line_profile='voigt' if you need.)
    )

    all_residuals = []
    c_speed = 299792.458

    # Loop lines => combine all residuals
    for line_id in wavelengths_dict:
        wv_all = wavelengths_dict[line_id]
        if len(wv_all) == 0:
            continue

        obs_all = fluxes_dict[line_id]
        unc_all = uncertainties_dict[line_id]
        ep_all = epochs_dict[line_id]
        mdl_all = model_flux[line_id]

        # array to store sub-residuals per epoch
        line_res_array = np.zeros_like(obs_all)

        unique_eps = np.unique(ep_all)
        for ep in unique_eps:
            idx = (ep_all == ep)
            wv = wv_all[idx]
            obs = obs_all[idx]
            unc = unc_all[idx]
            mdl = mdl_all[idx]

            # figure out star centers
            rv1, rv2 = _compute_rv1_rv2(params, ep)
            cwv = central_wavelengths[line_id]
            center1 = cwv * (1.0 + rv1 / c_speed)
            center2 = cwv * (1.0 + rv2 / c_speed)

            # line-based weighting => e.g. hydrogen lines get bigger unc
            line_wt_factor = get_line_weight(cwv)

            # peak-based weighting => gaussian or lorentz
            if peak_weight_kind.lower().startswith('g'):
                w_extra = _peak_weight_gaussian(wv, [center1, center2],
                                                amp=peak_weight_amp,
                                                width=peak_weight_width)
            else:
                w_extra = _peak_weight_lorentz(wv, [center1, center2],
                                               amp=peak_weight_amp,
                                               width=peak_weight_width)

            if weighted:
                # combine line weighting + peak weighting
                eff_unc = (unc * line_wt_factor) / np.sqrt(w_extra)
                res_sub = (obs - mdl) / eff_unc
            else:
                res_sub = obs - mdl

            line_res_array[idx] = res_sub

        all_residuals.append(line_res_array)

    return np.concatenate(all_residuals)
