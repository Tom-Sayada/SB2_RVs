import numpy as np
import logging
from typing import Dict, Union, Tuple, Optional
from lmfit import Parameters
from src.utils import voigt_profile, skewed_voigt_profile, min_width

logger = logging.getLogger(__name__)

class ModelError(Exception):
    """Custom exception for model computation errors"""
    pass

def setup_parameters(
        central_wavelengths,
        all_epochs,
        profile_type='sym',
        fit_baseline=False
):
    """
    Default parameter-setup function for the 'standard' approach.
    (If you want ratio-based, see ratio_fit_model.py)
    """
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

    # By default, for the standard approach, we add rv1/rv2 for each epoch
    for ep in all_epochs:
        e = int(ep)
        params.add(f'rv1_epoch{e}', value=0.0, min=-500, max=500)
        params.add(f'rv2_epoch{e}', value=0.0, min=-500, max=500)

    return params


def _compute_rv1_rv2(params, epoch):
    """
    Helper: Decide how to compute rv1, rv2 for a given epoch,
    depending on whether ratio & v_sys are in the parameter set.

    If ratio and v_sys exist:
       (rv1 + v_sys) = - ratio * (rv2 + v_sys).
    Else fallback to the standard approach: just read rv1_epochN, rv2_epochN.
    """
    ekey2 = f'rv2_epoch{epoch}'
    ekey1 = f'rv1_epoch{epoch}'

    # Check if ratio-based approach is in use
    if ('ratio' in params) and ('v_sys' in params) and (ekey2 in params):
        ratio_val = params['ratio'].value
        v_sys = params['v_sys'].value
        rv2 = params[ekey2].value
        # (rv1 + v_sys) = - ratio*(rv2 + v_sys) => rv1 = - ratio*(rv2 + v_sys) - v_sys
        rv1 = -ratio_val*(rv2 + v_sys) - v_sys
        return rv1, rv2
    else:
        # Standard approach
        rv1 = params[ekey1].value if ekey1 in params else 0.0
        rv2 = params[ekey2].value if ekey2 in params else 0.0
        return rv1, rv2


def compute_full_model(
        params,
        wavelengths_dict,
        epochs_dict,
        central_wavelengths,
        profile_type='sym'
):
    """
    Build model flux for each line_id, summing star1 + star2,
    either standard or ratio-based, depending on the presence
    of 'ratio' & 'v_sys' in the parameter set.
    """
    c_speed = 299792.458
    model_fluxes = {}

    # Initialize output arrays
    for line_id in wavelengths_dict:
        wv_arr = wavelengths_dict[line_id]
        if len(wv_arr) == 0:
            model_fluxes[line_id] = np.array([])
            continue

        # Baseline if present
        bkey = f'baseline_{line_id}'
        base_val = params[bkey].value if (bkey in params) else 1.0
        model_fluxes[line_id] = np.full_like(wv_arr, base_val)

    # Gather all epochs from the data
    try:
        all_ep = np.unique(np.concatenate([epochs_dict[lid] for lid in epochs_dict]))
    except Exception as e:
        raise ModelError(f"Error processing epochs: {str(e)}")

    # For each epoch, compute Doppler shifts
    for ep in all_ep:
        e = int(ep)

        rv1, rv2 = _compute_rv1_rv2(params, e)

        # Process each line
        for line_id in wavelengths_dict:
            idx = (epochs_dict[line_id] == e)
            if not np.any(idx):
                continue

            wv = wavelengths_dict[line_id][idx]
            cwv = central_wavelengths[line_id]

            # Get line parameters
            a1 = params[f'a1_{line_id}'].value
            s1 = params[f'sigma1_{line_id}'].value
            g1 = params[f'gamma1_{line_id}'].value
            a2 = params[f'a2_{line_id}'].value
            s2 = params[f'sigma2_{line_id}'].value
            g2 = params[f'gamma2_{line_id}'].value

            # Skew, if present
            skew1 = params.get(f'skew1_{line_id}', None)
            skew1 = skew1.value if skew1 else 0.0
            skew2 = params.get(f'skew2_{line_id}', None)
            skew2 = skew2.value if skew2 else 0.0

            # Doppler shifted centers
            center1 = cwv * (1 + rv1 / c_speed)
            center2 = cwv * (1 + rv2 / c_speed)

            # Build profiles
            try:
                if abs(skew1) > 1e-8:
                    prof1 = skewed_voigt_profile(wv, -abs(a1), center1, s1, g1, skew1)
                else:
                    prof1 = voigt_profile(wv, -abs(a1), center1, s1, g1)

                if abs(skew2) > 1e-8:
                    prof2 = skewed_voigt_profile(wv, -abs(a2), center2, s2, g2, skew2)
                else:
                    prof2 = voigt_profile(wv, -abs(a2), center2, s2, g2)
            except Exception as ee:
                raise ModelError(f"Error computing profiles for {line_id}, epoch={ep}: {str(ee)}")

            model_fluxes[line_id][idx] += (prof1 + prof2)

    return model_fluxes


def get_star_components(params, line_id, wv_array, cwave, rv1_val, rv2_val, profile_type='sym'):
    """Return star1_flux, star2_flux individually for plotting."""
    base_val = 1.0
    bkey = f'baseline_{line_id}'
    if bkey in params:
        base_val = params[bkey].value

    # Get line parameters
    a1 = params[f'a1_{line_id}'].value
    s1 = params[f'sigma1_{line_id}'].value
    g1 = params[f'gamma1_{line_id}'].value
    a2 = params[f'a2_{line_id}'].value
    s2 = params[f'sigma2_{line_id}'].value
    g2 = params[f'gamma2_{line_id}'].value

    skew1 = params.get(f'skew1_{line_id}', None)
    skew1 = skew1.value if skew1 else 0.0
    skew2 = params.get(f'skew2_{line_id}', None)
    skew2 = skew2.value if skew2 else 0.0

    center1 = cwave * (1 + rv1_val / 299792.458)
    center2 = cwave * (1 + rv2_val / 299792.458)

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


def residuals(
        params,
        wavelengths_dict,
        fluxes_dict,
        epochs_dict,
        uncertainties_dict,
        central_wavelengths,
        profile_type='sym',
        weighted=True
):
    """
    Calculate residuals for fitting (both standard & ratio-based).
    """
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

        if weighted:
            all_residuals.append((obs - mod) / unc)
        else:
            all_residuals.append(obs - mod)

    return np.concatenate(all_residuals)
