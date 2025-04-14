# model_builder.py

import numpy as np
import logging
from typing import Dict, Optional
from lmfit import Parameters
from collections import defaultdict

# Import line-profile functions from your new file:
from src.line_profiles import (
    _min_width,
    voigt_profile, skewed_voigt_profile,
    gaussian_profile, skewed_gaussian_profile
)

logger = logging.getLogger(__name__)

class ModelError(Exception):
    """Custom exception for model computation errors"""
    pass

def _compute_rv1_rv2(params, epoch):
    """
    Decide how to compute rv1, rv2 for a given epoch.
    If 'ratio' and 'v_sys' are in params, we use:
       (rv1 + v_sys) = - ratio * (rv2 + v_sys)
    Otherwise, fallback to explicit rv1_epochN, rv2_epochN.
    """
    ekey2 = f'rv2_epoch{epoch}'
    ekey1 = f'rv1_epoch{epoch}'

    if ('ratio' in params) and ('v_sys' in params) and (ekey2 in params):
        ratio_val = params['ratio'].value
        v_sys = params['v_sys'].value
        rv2 = params[ekey2].value
        # rv1 = - ratio*(rv2 + v_sys) - v_sys
        rv1 = -ratio_val * (rv2 + v_sys) - v_sys
        return rv1, rv2
    else:
        rv1 = params[ekey1].value if ekey1 in params else 0.0
        rv2 = params[ekey2].value if ekey2 in params else 0.0
        return rv1, rv2


def setup_parameters(
        central_wavelengths,
        all_epochs,
        profile_type='sym',
        fit_baseline=False,
        line_profile='voigt'
):
    """
    Default parameter-setup function for the 'standard' approach.
    (If you want ratio-based, see ratio_fit_model.py)

    Adds a1, sigma1, gamma1 (voigt only), a2, sigma2, gamma2 (voigt only),
    optional baseline, optional skew to each line. Then adds rv1_epochN, rv2_epochN for each epoch.
    """
    params = Parameters()

    # Set sensible bounds and initial values
    for line_id, cwv in central_wavelengths.items():
        # Identify line type
        is_hydrogen = any(str(cwv).startswith(h) for h in ['4340', '4101', '4861', '6563'])

        if is_hydrogen:
            # H lines => broader, deeper
            params.add(f'a1_{line_id}', value=-1.0, min=-5.0, max=-0.01)
            params.add(f'sigma1_{line_id}', value=2.0, min=0.01, max=2.0)
            params.add(f'a2_{line_id}', value=-0.8, min=-5.0, max=-0.01)
            params.add(f'sigma2_{line_id}', value=2.0, min=0.01, max=2.0)
            if line_profile == 'voigt':
                params.add(f'gamma1_{line_id}', value=1.0, min=0.05, max=4.0)
                params.add(f'gamma2_{line_id}', value=1.0, min=0.05, max=4.0)
        else:
            # Helium lines => narrower
            params.add(f'a1_{line_id}', value=-0.4, min=-1.0, max=-0.005)
            params.add(f'sigma1_{line_id}', value=0.8, min=0.001, max=4.0)
            params.add(f'a2_{line_id}', value=-0.3, min=-1.0, max=-0.005)
            params.add(f'sigma2_{line_id}', value=0.8, min=0.001, max=4.0)
            if line_profile == 'voigt':
                params.add(f'gamma1_{line_id}', value=0.8, min=0.001, max=4.0)
                params.add(f'gamma2_{line_id}', value=0.8, min=0.001, max=4.0)

        if fit_baseline:
            if is_hydrogen:
                params.add(f'baseline_{line_id}', value=1.0, min=0.95, max=1.05)
            else:
                params.add(f'baseline_{line_id}', value=1.0, min=0.98, max=1.02)

        if profile_type == 'asym':
            params.add(f'skew1_{line_id}', value=0.0, min=-1.0, max=1.0)
            params.add(f'skew2_{line_id}', value=0.0, min=-1.0, max=1.0)

    # Add rv1/rv2 for each epoch
    for ep in all_epochs:
        e = int(ep)
        params.add(f'rv1_epoch{e}', value=0.0, min=-500, max=500)
        params.add(f'rv2_epoch{e}', value=0.0, min=-500, max=500)

    return params


def compute_full_model(
        params,
        wavelengths_dict,
        epochs_dict,
        central_wavelengths,
        profile_type='sym',
        line_profile='voigt'
):
    """
    Build model flux for each line_id, summing star1 + star2.
    """
    c_speed = 299792.458
    model_fluxes = {}

    # Initialize output arrays with baseline
    for line_id in wavelengths_dict:
        wv_arr = wavelengths_dict[line_id]
        if len(wv_arr) == 0:
            model_fluxes[line_id] = np.array([])
            continue

        bkey = f'baseline_{line_id}'
        base_val = params[bkey].value if (bkey in params) else 1.0
        model_fluxes[line_id] = np.full_like(wv_arr, base_val)

    # Gather all epochs from the data
    try:
        all_ep = np.unique(np.concatenate([epochs_dict[lid] for lid in epochs_dict]))
    except Exception as e:
        raise ModelError(f"Error processing epochs: {str(e)}")

    # For each epoch, compute line profiles
    for ep in all_ep:
        e = int(ep)
        rv1, rv2 = _compute_rv1_rv2(params, e)

        for line_id in wavelengths_dict:
            idx = (epochs_dict[line_id] == e)
            if not np.any(idx):
                continue

            wv = wavelengths_dict[line_id][idx]
            cwv = central_wavelengths[line_id]

            # line-shape params
            a1 = params[f'a1_{line_id}'].value
            s1 = params[f'sigma1_{line_id}'].value
            a2 = params[f'a2_{line_id}'].value
            s2 = params[f'sigma2_{line_id}'].value

            g1 = g2 = None
            if line_profile == 'voigt':
                try:
                    g1 = params[f'gamma1_{line_id}'].value
                    g2 = params[f'gamma2_{line_id}'].value
                except KeyError:
                    raise ModelError(f"Missing gamma parameters for Voigt profile on line {line_id}")

            # skew
            skew1 = params.get(f'skew1_{line_id}', None)
            skew1 = skew1.value if skew1 else 0.0
            skew2 = params.get(f'skew2_{line_id}', None)
            skew2 = skew2.value if skew2 else 0.0

            center1 = cwv * (1 + rv1 / c_speed)
            center2 = cwv * (1 + rv2 / c_speed)

            # build star1
            if line_profile == 'gaussian':
                if abs(skew1) > 1e-8:
                    prof1 = skewed_gaussian_profile(wv, -abs(a1), center1, s1, skew1)
                else:
                    prof1 = gaussian_profile(wv, -abs(a1), center1, s1)

                if abs(skew2) > 1e-8:
                    prof2 = skewed_gaussian_profile(wv, -abs(a2), center2, s2, skew2)
                else:
                    prof2 = gaussian_profile(wv, -abs(a2), center2, s2)
            else:
                # voigt
                if g1 is None or g2 is None:
                    raise ModelError("Gamma parameters required for Voigt profile")
                if abs(skew1) > 1e-8:
                    prof1 = skewed_voigt_profile(wv, -abs(a1), center1, s1, g1, skew1)
                else:
                    prof1 = voigt_profile(wv, -abs(a1), center1, s1, g1)

                if abs(skew2) > 1e-8:
                    prof2 = skewed_voigt_profile(wv, -abs(a2), center2, s2, g2, skew2)
                else:
                    prof2 = voigt_profile(wv, -abs(a2), center2, s2, g2)

            model_fluxes[line_id][idx] += (prof1 + prof2)

    return model_fluxes


def get_star_components(params, line_id, wv_array, cwave, rv1_val, rv2_val,
                        profile_type='sym', line_profile='voigt'):
    """
    Return star1_flux, star2_flux individually for plotting.
    """
    base_val = 1.0
    bkey = f'baseline_{line_id}'
    if bkey in params:
        base_val = params[bkey].value

    a1 = params[f'a1_{line_id}'].value
    s1 = params[f'sigma1_{line_id}'].value
    a2 = params[f'a2_{line_id}'].value
    s2 = params[f'sigma2_{line_id}'].value

    g1 = g2 = None
    if line_profile == 'voigt':
        g1 = params[f'gamma1_{line_id}'].value
        g2 = params[f'gamma2_{line_id}'].value

    skew1 = params.get(f'skew1_{line_id}', None)
    skew1 = skew1.value if skew1 else 0.0
    skew2 = params.get(f'skew2_{line_id}', None)
    skew2 = skew2.value if skew2 else 0.0

    center1 = cwave * (1 + rv1_val / 299792.458)
    center2 = cwave * (1 + rv2_val / 299792.458)

    if line_profile == 'gaussian':
        if abs(skew1) > 1e-8:
            prof1 = skewed_gaussian_profile(wv_array, -abs(a1), center1, s1, skew1)
        else:
            prof1 = gaussian_profile(wv_array, -abs(a1), center1, s1)

        if abs(skew2) > 1e-8:
            prof2 = skewed_gaussian_profile(wv_array, -abs(a2), center2, s2, skew2)
        else:
            prof2 = gaussian_profile(wv_array, -abs(a2), center2, s2)
    else:
        # voigt
        if abs(skew1) > 1e-8:
            prof1 = skewed_voigt_profile(wv_array, -abs(a1), center1, s1, g1, skew1)
        else:
            prof1 = voigt_profile(wv_array, -abs(a1), center1, s1, g1)

        if abs(skew2) > 1e-8:
            prof2 = skewed_voigt_profile(wv_array, -abs(a2), center2, s2, g2, skew2)
        else:
            prof2 = voigt_profile(wv_array, -abs(a2), center2, s2, g2)

    star1_flux = base_val + prof1
    star2_flux = base_val + prof2
    return star1_flux, star2_flux


def _peak_weight_gaussian(wv, centers, amp, width):
    """
    Returns a multiplicative weight array based on a sum of Gaussian peaks
    around each center. w[i] = 1 + amp * sum( exp(-(wv[i]-c_j)^2/(2*width^2)) ).
    If amp=0 => returns all ones => no weighting.
    """
    w = np.ones_like(wv)
    if amp <= 0 or width <= 0:
        return w
    for c in centers:
        dist = wv - c
        w += amp * np.exp(-0.5 * (dist / width)**2)
    return w

def _peak_weight_lorentz(wv, centers, amp, width):
    """
    Returns a multiplicative weight array based on a sum of Lorentzian peaks
    around each center. w[i] = 1 + amp * sum( 1/[1 + ((wv[i]-c_j)/width)^2] ).
    If amp=0 => returns all ones => no weighting.
    """
    w = np.ones_like(wv)
    if amp <= 0 or width <= 0:
        return w
    for c in centers:
        dist = wv - c
        w += amp / (1.0 + (dist/width)**2)
    return w


def residuals(
        params,
        wavelengths_dict,
        fluxes_dict,
        epochs_dict,
        uncertainties_dict,
        central_wavelengths,
        profile_type='sym',
        line_profile='voigt',
        weighted=True,
        peak_weight_amp=1.0,
        peak_weight_width=0.8,
        peak_weight_kind='gaussian'
):
    """
    Calculate residuals for the standard approach. Optionally apply extra
    weighting near line cores (star1 + star2) using Gaussian or Lorentz profiles.
    """
    mod_flux = compute_full_model(
        params,
        wavelengths_dict,
        epochs_dict,
        central_wavelengths,
        profile_type=profile_type,
        line_profile=line_profile
    )

    all_residuals = []
    c_speed = 299792.458

    for line_id in wavelengths_dict:
        if len(wavelengths_dict[line_id]) == 0:
            continue

        wv_all = wavelengths_dict[line_id]
        obs_all = fluxes_dict[line_id]
        unc_all = uncertainties_dict[line_id]
        ep_all = epochs_dict[line_id]

        mdl_all = mod_flux[line_id]
        res_array = np.zeros_like(obs_all)

        unique_eps = np.unique(ep_all)
        for ep in unique_eps:
            idx = (ep_all == ep)
            wv = wv_all[idx]
            obs = obs_all[idx]
            mdl = mdl_all[idx]
            unc = unc_all[idx]

            rv1, rv2 = _compute_rv1_rv2(params, ep)
            cwv = central_wavelengths[line_id]
            center1 = cwv * (1 + rv1/c_speed)
            center2 = cwv * (1 + rv2/c_speed)

            # Build peak weighting
            if peak_weight_kind.lower().startswith('g'):
                w_extra = _peak_weight_gaussian(wv, [center1, center2],
                                                amp=peak_weight_amp,
                                                width=peak_weight_width)
            else:
                w_extra = _peak_weight_lorentz(wv, [center1, center2],
                                               amp=peak_weight_amp,
                                               width=peak_weight_width)

            if weighted:
                eff_unc = unc / np.sqrt(w_extra)
                res_sub = (obs - mdl)/eff_unc
            else:
                res_sub = obs - mdl

            res_array[idx] = res_sub

        all_residuals.append(res_array)

    return np.concatenate(all_residuals)
