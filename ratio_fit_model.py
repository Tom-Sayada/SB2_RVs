# ratio_fit_model.py

"""
Ratio + v_sys approach:
  (rv1 + v_sys) = - ratio * (rv2 + v_sys)

Stores rv2_epochN only. We compute rv1 on the fly from ratio, v_sys, and rv2.
"""

import numpy as np
from lmfit import Parameters
from typing import Optional
# We import compute_full_model from model_builder; that uses line_profiles under the hood.
from src.model_builder import compute_full_model, _compute_rv1_rv2, ModelError

# If you need line_profiles directly (e.g. custom partial fits),
# you could import them. But typically we rely on model_builder.
from src.line_profiles import (
    gaussian_profile,
    skewed_gaussian_profile,
    voigt_profile,
    skewed_voigt_profile
)

def get_line_weight(wavelength: float) -> float:
    """
    If you want H lines to weigh differently, do so here.
    E.g., near H lines => weight=2.0, else 1.0
    """
    h_lines = [4340.472, 4101.734, 3970.075]  # example lines
    for h_wave in h_lines:
        if abs(wavelength - h_wave) < 1.0:
            return 2.0
    return 1.0


def _peak_weight_gaussian(wv, centers, amp, width):
    w = np.ones_like(wv)
    if amp <= 0 or width <= 0:
        return w
    for c in centers:
        dist = wv - c
        w += amp * np.exp(-0.5 * (dist / width)**2)
    return w

def _peak_weight_lorentz(wv, centers, amp, width):
    w = np.ones_like(wv)
    if amp <= 0 or width <= 0:
        return w
    for c in centers:
        dist = wv - c
        w += amp / (1.0 + (dist / width)**2)
    return w


def setup_parameters(
        central_wavelengths,
        all_epochs,
        profile_type='sym',
        fit_baseline=False,
        line_profile='voigt',
        initial_rvs=None
):
    """
    Create an lmfit.Parameters() set for the ratio-based approach.

    - ratio, v_sys: global
    - rv2_epoch{N}: one for each epoch
    - each line_id: a1, sigma1, gamma1..., a2, sigma2, gamma2..., optional baseline, optional skew
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
    params.add('ratio', value=1.0, min=0.6, max=1.5)
    params.add('v_sys', value=150.0, min=100.0, max=200.0)

    # rv2 for each epoch
    for ep in all_epochs:
        epval = int(ep)
        rv2_init = initial_rvs.get(epval, 0.0)
        params.add(f'rv2_epoch{epval}', value=rv2_init, min=-100.0, max=300.0)

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
        fit_baseline=False,
        peak_weight_amp=1.0,
        peak_weight_width=1.0,
        peak_weight_kind='gaussian'
):
    """
    Main residual function for the ratio approach.
    Uses compute_full_model(...) from model_builder, which enforces:
        (rv1 + v_sys) = - ratio*(rv2 + v_sys).
    """
    mod_flux = compute_full_model(
        params,
        wavelengths_dict,
        epochs_dict,
        central_wavelengths,
        profile_type=profile_type,
        line_profile=line_profile
    )

    all_res = []
    c_speed = 299792.458

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

            rv1, rv2 = _compute_rv1_rv2(params, ep)
            cwv = central_wavelengths[line_id]
            center1 = cwv*(1 + rv1/c_speed)
            center2 = cwv*(1 + rv2/c_speed)

            # line-based weighting => e.g. H lines
            base_line_wt = get_line_weight(cwv)

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

        all_res.append(res_array)

    return np.concatenate(all_res)
