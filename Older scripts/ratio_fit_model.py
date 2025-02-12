# ratio_fit_model.py

"""
ratio_fit_model.py

Ratio + v_sys approach:
  (rv1 + v_sys) = - ratio * (rv2 + v_sys)

We store rv2_epoch only. 'initial_rvs' => float rv2_init.
"""

import numpy as np
from lmfit import Parameters
from src.utils import voigt_profile, skewed_voigt_profile, min_width
from src.model_builder import compute_full_model

def get_line_weight(wavelength: float) -> float:
    """
    If you want H lines to weigh differently, do so here.
    """
    h_lines = [4340.472, 4101.734, 3970.075]
    for h_wave in h_lines:
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
    if initial_rvs is None:
        initial_rvs = {}

    params = Parameters()

    for line_id, cwv in central_wavelengths.items():
        is_hydrogen = any(str(cwv).startswith(h) for h in ['4340','4861','6563'])
        if is_hydrogen:
            params.add(f'a1_{line_id}', value=-1.0, min=-5.0, max=-0.1)
            params.add(f'sigma1_{line_id}', value=2.0, min=min_width, max=5.0)
            params.add(f'gamma1_{line_id}', value=2.0, min=min_width, max=5.0)
            params.add(f'a2_{line_id}', value=-0.8, min=-5.0, max=-0.1)
            params.add(f'sigma2_{line_id}', value=2.0, min=min_width, max=5.0)
            params.add(f'gamma2_{line_id}', value=2.0, min=min_width, max=5.0)
        else:
            params.add(f'a1_{line_id}', value=-0.4, min=-2.0, max=-0.05)
            params.add(f'sigma1_{line_id}', value=0.8, min=min_width, max=3.0)
            params.add(f'gamma1_{line_id}', value=0.8, min=min_width, max=3.0)
            params.add(f'a2_{line_id}', value=-0.3, min=-2.0, max=-0.05)
            params.add(f'sigma2_{line_id}', value=0.8, min=min_width, max=3.0)
            params.add(f'gamma2_{line_id}', value=0.8, min=min_width, max=3.0)

        if fit_baseline:
            if is_hydrogen:
                params.add(f'baseline_{line_id}', value=1.0, min=0.95, max=1.05)
            else:
                params.add(f'baseline_{line_id}', value=1.0, min=0.98, max=1.02)

        if profile_type in ('asym','skewed'):
            params.add(f'skew1_{line_id}', value=0.0, min=-1.0, max=1.0)
            params.add(f'skew2_{line_id}', value=0.0, min=-1.0, max=1.0)

    # ratio, v_sys
    params.add('ratio', value=1.0, min=0.01, max=20.0)
    params.add('v_sys', value=0.0, min=-500.0, max=500.0)

    # Only store rv2 per epoch
    for ep in all_epochs:
        e = int(ep)
        rv2_init = initial_rvs.get(e, 0.0)
        params.add(f'rv2_epoch{e}', value=rv2_init, min=-500.0, max=500.0)

    return params

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
    mod_flux = compute_full_model(
        params,
        wavelengths_dict,
        epochs_dict,
        central_wavelengths,
        profile_type=profile_type
    )

    all_res = []
    for line_id in wavelengths_dict:
        wv = wavelengths_dict[line_id]
        obs = fluxes_dict[line_id]
        mdl = mod_flux[line_id]
        unc = uncertainties_dict[line_id]
        if len(obs) == 0:
            continue

        wave_center = central_wavelengths[line_id]
        lw = get_line_weight(wave_center)
        if weighted:
            unc_eff = unc * lw
            res = (obs - mdl)/unc_eff
        else:
            res = obs - mdl

        all_res.append(res)

    return np.concatenate(all_res)
