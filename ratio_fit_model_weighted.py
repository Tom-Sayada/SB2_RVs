# ratio_fit_model_weighted.py

"""
Weighted ratio + v_sys approach:
  (rv1 + v_sys) = - ratio * (rv2 + v_sys)

Stores rv2_epochN only. We compute rv1 on the fly from ratio, v_sys, and rv2.

**Difference** from the original ratio_fit_model.py:
 - We introduce an epoch-dependent weighting factor based on |rv1 - rv2|.
 - We use the global CURRENT_EPOCH_WEIGHTS dictionary to store weights per epoch.
 - We integrate with the iterate_with_weights function from model_builder_weighted.
"""

import numpy as np
from lmfit import Parameters
from typing import Dict, Optional

# Import the weighted model builder so we don't conflict with the old file
from src.model_builder_weighted import (
    compute_full_model,
    _compute_rv1_rv2,
    separation_weight
)

# Global dictionary to store current weights
CURRENT_EPOCH_WEIGHTS = {}


def setup_parameters(
        central_wavelengths,
        all_epochs,
        profile_type='sym',
        fit_baseline=False,
        line_profile='voigt',
        initial_rvs=None
):
    """
    Setup parameters for the ratio-constrained approach.
    """
    if initial_rvs is None:
        initial_rvs = {}

    params = Parameters()

    for line_id, cwv in central_wavelengths.items():
        is_hydrogen = any(str(cwv).startswith(x) for x in ['4340', '4861', '6563'])

        if is_hydrogen:
            params.add(f'a1_{line_id}', value=-1.0, min=-5.0, max=-0.1)
            params.add(f'sigma1_{line_id}', value=2.0, min=1e-5, max=5.0)
            params.add(f'a2_{line_id}', value=-0.8, min=-5.0, max=-0.1)
            params.add(f'sigma2_{line_id}', value=2.0, min=1e-5, max=5.0)
            if line_profile == 'voigt':
                params.add(f'gamma1_{line_id}', value=2.0, min=1e-5, max=5.0)
                params.add(f'gamma2_{line_id}', value=2.0, min=1e-5, max=5.0)
        else:
            params.add(f'a1_{line_id}', value=-0.4, min=-2.0, max=-0.05)
            params.add(f'sigma1_{line_id}', value=0.8, min=1e-5, max=3.0)
            params.add(f'a2_{line_id}', value=-0.3, min=-2.0, max=-0.05)
            params.add(f'sigma2_{line_id}', value=0.8, min=1e-5, max=3.0)
            if line_profile == 'voigt':
                params.add(f'gamma1_{line_id}', value=0.8, min=1e-5, max=3.0)
                params.add(f'gamma2_{line_id}', value=0.8, min=1e-5, max=3.0)

        if fit_baseline:
            if is_hydrogen:
                params.add(f'baseline_{line_id}', value=1.0, min=0.95, max=1.05)
            else:
                params.add(f'baseline_{line_id}', value=1.0, min=0.98, max=1.02)

        if profile_type in ('asym', 'skewed'):
            params.add(f'skew1_{line_id}', value=0.0, min=-1.0, max=1.0)
            params.add(f'skew2_{line_id}', value=0.0, min=-1.0, max=1.0)

    # ratio, v_sys
    params.add('ratio', value=1.0, min=0.01, max=20.0)
    params.add('v_sys', value=0.0, min=-500.0, max=500.0)

    # rv2 for each epoch
    for ep in all_epochs:
        epval = int(ep)
        rv2_init = initial_rvs.get(epval, 0.0)
        params.add(f'rv2_epoch{epval}', value=rv2_init, min=-500.0, max=500.0)

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
    Weighted residual function for ratio approach, with separation-based weighting.

    Uses pre-computed weights from the CURRENT_EPOCH_WEIGHTS global dictionary if available,
    otherwise calculates weights on the fly.

    If weighted=True, we apply:
    1. Normal 1/sigma weighting
    2. Additional factor from separation_weight(rv1, rv2) to downweight near-blend epochs

    Parameters:
        min_weight: Minimum weight for blended epochs (0-1, default 0.2)
        max_sep: RV separation in km/s above which weight=1.0 (default 400)
    """
    # Access the global weights dictionary
    global CURRENT_EPOCH_WEIGHTS

    # Compute model fluxes
    mod_flux = compute_full_model(
        params,
        wavelengths_dict,
        epochs_dict,
        central_wavelengths,
        profile_type=profile_type,
        line_profile=line_profile
    )

    all_res = []

    for line_id in wavelengths_dict:
        wv_all = wavelengths_dict[line_id]
        if len(wv_all) == 0:
            continue

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
            unc = unc_all[idx]
            mdl = mdl_all[idx]

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

        all_res.append(res_array)

    return np.concatenate(all_res)


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
        result = minimizer.minimize(method='leastsq')
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