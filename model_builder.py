import numpy as np
from src.utils import voigt_profile, skewed_voigt_profile

def compute_full_model(params, wavelengths_dict, epochs_dict, central_wavelengths):
    """
    Build the model flux for each line & epoch by summing star1 & star2 Voigt profiles.
    Checks if 'ratio' is in the params for ratio-based approach (else standard).
    If skew parameters exist (skew1_lineID, skew2_lineID), uses skewed_voigt_profile.
    """

    c_speed = 299792.458  # km/s
    model_fluxes = {}

    # Gather all unique epochs used
    all_epochs = np.unique(np.concatenate([ep for ep in epochs_dict.values()]))

    has_ratio = ('ratio' in params)
    ratio_val = params['ratio'].value if has_ratio else None

    # Initialize each line’s flux to 1.0
    for line_id, wv_array in wavelengths_dict.items():
        model_fluxes[line_id] = np.ones_like(wv_array)

    # Fill the model
    for ep in all_epochs:
        ep_int = int(ep)

        # Handle ratio-based or standard RV
        if has_ratio:
            rv2 = params[f'rv2_epoch{ep_int}'].value
            rv1 = - ratio_val * rv2
        else:
            rv1 = params[f'rv1_epoch{ep_int}'].value
            rv2 = params[f'rv2_epoch{ep_int}'].value

        for line_id in wavelengths_dict:
            idx = (epochs_dict[line_id] == ep_int)
            if not np.any(idx):
                continue

            wvs = wavelengths_dict[line_id][idx]
            cwv = central_wavelengths[line_id]

            a1 = params[f'a1_{line_id}'].value
            s1 = params[f'sigma1_{line_id}'].value
            g1 = params[f'gamma1_{line_id}'].value
            a2 = params[f'a2_{line_id}'].value
            s2 = params[f'sigma2_{line_id}'].value
            g2 = params[f'gamma2_{line_id}'].value

            # Safely check for skew parameters
            skew1_key = f'skew1_{line_id}'
            skew2_key = f'skew2_{line_id}'

            if skew1_key in params:
                skew1_val = params[skew1_key].value
            else:
                skew1_val = 0.0

            if skew2_key in params:
                skew2_val = params[skew2_key].value
            else:
                skew2_val = 0.0

            # Doppler-shifted centers
            sc1 = cwv * (1 + rv1 / c_speed)
            sc2 = cwv * (1 + rv2 / c_speed)

            # Build star1 profile
            if abs(skew1_val) > 1e-8:
                prof1 = skewed_voigt_profile(wvs, -abs(a1), sc1, s1, g1, skew1_val)
            else:
                prof1 = voigt_profile(wvs, -abs(a1), sc1, s1, g1)

            # Build star2 profile
            if abs(skew2_val) > 1e-8:
                prof2 = skewed_voigt_profile(wvs, -abs(a2), sc2, s2, g2, skew2_val)
            else:
                prof2 = voigt_profile(wvs, -abs(a2), sc2, s2, g2)

            model_fluxes[line_id][idx] += (prof1 + prof2)

    return model_fluxes
