# model_builder_weighted.py

import numpy as np
import logging
from typing import Dict, Optional, Any
from lmfit import Parameters
from collections import defaultdict

# Import line-profile functions:
from src.line_profiles import (
    voigt_profile, skewed_voigt_profile,
    gaussian_profile, skewed_gaussian_profile
)

logger = logging.getLogger(__name__)

# Global dictionary to store current epoch weights for iteration
CURRENT_EPOCH_WEIGHTS = {}


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


def separation_weight(rv1, rv2, min_weight=0.2, max_sep=400.0):
    """
    Compute weight based on RV separation.
    Returns a value between min_weight and 1.0.
    - For small separations (near blends), returns min_weight
    - For large separations (≥ max_sep), returns 1.0
    - Linear ramp in between
    """
    sep = abs(rv1 - rv2)
    if sep >= max_sep:
        return 1.0
    # Linear ramp from min_weight to 1.0
    weight = min_weight + (1.0 - min_weight) * (sep / max_sep)
    return max(min_weight, min(weight, 1.0))


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


def setup_parameters(
        central_wavelengths,
        all_epochs,
        profile_type='sym',
        fit_baseline=False,
        line_profile='voigt'
):
    """
    Default parameter-setup function for standard approach with separation weighting.
    Sets up a1, sigma1, gamma1, a2, sigma2, gamma2, and rv1_epochN, rv2_epochN.
    """
    params = Parameters()

    # Set sensible bounds and initial values
    for line_id, cwv in central_wavelengths.items():
        # Identify line type
        is_hydrogen = any(str(cwv).startswith(h) for h in ['4340', '4861', '6563'])

        if is_hydrogen:
            # H lines => broader, deeper
            params.add(f'a1_{line_id}', value=-1.0, min=-20.0, max=-0.1)
            params.add(f'sigma1_{line_id}', value=2.0, min=0.1, max=5.0)
            params.add(f'a2_{line_id}', value=-0.8, min=-20.0, max=-0.1)
            params.add(f'sigma2_{line_id}', value=2.0, min=0.1, max=5.0)
            if line_profile == 'voigt':
                params.add(f'gamma1_{line_id}', value=2.0, min=0.1, max=5.0)
                params.add(f'gamma2_{line_id}', value=2.0, min=0.1, max=5.0)
        else:
            # Helium lines => narrower
            params.add(f'a1_{line_id}', value=-0.4, min=-20.0, max=-0.05)
            params.add(f'sigma1_{line_id}', value=0.8, min=0.1, max=3.0)
            params.add(f'a2_{line_id}', value=-0.3, min=-20.0, max=-0.05)
            params.add(f'sigma2_{line_id}', value=0.8, min=0.1, max=3.0)
            if line_profile == 'voigt':
                params.add(f'gamma1_{line_id}', value=0.8, min=0.1, max=3.0)
                params.add(f'gamma2_{line_id}', value=0.8, min=0.1, max=3.0)

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
        min_weight=0.2,
        max_sep=400.0
):
    """
    Calculate residuals for the weighted standard approach.
    Uses global CURRENT_EPOCH_WEIGHTS if available, or calculates weights on the fly.
    """
    # Compute model fluxes
    mod_flux = compute_full_model(
        params,
        wavelengths_dict,
        epochs_dict,
        central_wavelengths,
        profile_type=profile_type,
        line_profile=line_profile
    )

    all_residuals = []

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

            # Get RVs for this epoch
            rv1, rv2 = _compute_rv1_rv2(params, ep)

            # Use pre-stored weight if available, otherwise calculate
            if ep in CURRENT_EPOCH_WEIGHTS:
                sep_factor = CURRENT_EPOCH_WEIGHTS[ep]
            else:
                sep_factor = separation_weight(rv1, rv2, min_weight=min_weight, max_sep=max_sep)

            # Apply the weight to uncertainties
            if weighted:
                # Adjust uncertainties based on separation weight:
                # - Small weight (near blend) => increase uncertainty (downweight in fit)
                # - Large weight (well separated) => keep uncertainty as is
                eff_unc = unc / np.sqrt(sep_factor)
                res_sub = (obs - mdl) / eff_unc
            else:
                res_sub = obs - mdl

            res_array[idx] = res_sub

        all_residuals.append(res_array)

    return np.concatenate(all_residuals)


def iterate_with_weights(
        params,
        minimizer,
        all_epochs,
        min_weight=0.2,
        max_sep=400.0,
        max_iterations=3,
        verbose=True
):
    """
    Iteratively optimize with RV separation-based weighting.

    Process:
    1. Start with initial parameters
    2. Run fit with current weights
    3. Update weights based on new RVs
    4. Repeat until convergence or max_iterations reached

    Returns:
        Final minimizer result
    """
    global CURRENT_EPOCH_WEIGHTS

    current_params = params.copy()
    prev_chi2 = float('inf')

    for iteration in range(max_iterations):
        # Update epoch weights based on current RVs
        CURRENT_EPOCH_WEIGHTS = {}
        for ep in all_epochs:
            rv1, rv2 = _compute_rv1_rv2(current_params, ep)
            weight = separation_weight(rv1, rv2, min_weight=min_weight, max_sep=max_sep)
            CURRENT_EPOCH_WEIGHTS[ep] = weight
            if verbose:
                print(f"  Epoch {ep}: RV1={rv1:.2f}, RV2={rv2:.2f}, sep={abs(rv1 - rv2):.2f}, weight={weight:.3f}")

        # Run minimization with these weights
        if verbose:
            print(f"Iteration {iteration + 1}/{max_iterations}: Running minimization with updated weights...")

        # Update minimizer parameters
        for param_name, param in current_params.items():
            if param_name in minimizer.params:
                minimizer.params[param_name].value = param.value

        # Run minimization
        result = minimizer.minimize(method='leastsq', params=minimizer.params)
        current_params = result.params

        # Check convergence
        chi2 = sum(result.residual ** 2)
        if verbose:
            print(f"  Chi² = {chi2:.2f} (prev: {prev_chi2:.2f})")

        if abs(chi2 - prev_chi2) < 0.01 * prev_chi2:
            if verbose:
                print(f"  Converged after {iteration + 1} iterations (change < 1%)")
            break

        prev_chi2 = chi2

    # Final output
    if verbose:
        print("Final weights by epoch:")
        for ep in sorted(CURRENT_EPOCH_WEIGHTS.keys()):
            rv1, rv2 = _compute_rv1_rv2(current_params, ep)
            print(
                f"  Epoch {ep}: RV1={rv1:.2f}, RV2={rv2:.2f}, sep={abs(rv1 - rv2):.2f}, weight={CURRENT_EPOCH_WEIGHTS[ep]:.3f}")

    return result