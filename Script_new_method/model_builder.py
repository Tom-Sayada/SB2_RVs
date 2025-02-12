import numpy as np
import logging
from typing import Dict, Tuple, Optional
from lmfit import Parameters
from src.utils import voigt_profile, skewed_voigt_profile

logger = logging.getLogger(__name__)

class ModelError(Exception):
    """Custom exception for model computation errors"""
    pass

def build_output_subfolder_name(
    fit_type: str,
    profile_type: str,
    use_weighted: bool,
    fit_baseline: bool
) -> str:
    """
    Build a standardized output subfolder name, e.g.
    'standard_sym_weighted_nobaseline_fit_results'
    """
    w_str = "weighted" if use_weighted else "unweighted"
    b_str = "baseline" if fit_baseline else "nobaseline"
    return f"{fit_type}_{profile_type}_{w_str}_{b_str}_fit_results"


def setup_parameters(
        central_wavelengths: Dict[str, float],
        all_epochs: list,
        profile_type: str='sym',
        fit_baseline: bool=False
) -> Parameters:
    """
    Basic param setup, no star-swap logic
    """
    from lmfit import Parameters
    params = Parameters()

    for line_id, cwv in central_wavelengths.items():
        is_hydrogen = any(str(cwv).startswith(h) for h in ['4340','4861','6563'])
        if is_hydrogen:
            params.add(f'a1_{line_id}', value=-0.8, min=-3.0, max=-0.05)
            params.add(f'sigma1_{line_id}', value=1.5, min=0.2, max=8.0)
            params.add(f'gamma1_{line_id}', value=1.5, min=0.2, max=8.0)
            params.add(f'a2_{line_id}', value=-0.6, min=-3.0, max=-0.05)
            params.add(f'sigma2_{line_id}', value=1.5, min=0.2, max=8.0)
            params.add(f'gamma2_{line_id}', value=1.5, min=0.2, max=8.0)
        else:
            params.add(f'a1_{line_id}', value=-0.3, min=-2.0, max=-0.02)
            params.add(f'sigma1_{line_id}', value=0.6, min=0.1, max=5.0)
            params.add(f'gamma1_{line_id}', value=0.6, min=0.1, max=5.0)
            params.add(f'a2_{line_id}', value=-0.25, min=-2.0, max=-0.02)
            params.add(f'sigma2_{line_id}', value=0.6, min=0.1, max=5.0)
            params.add(f'gamma2_{line_id}', value=0.6, min=0.1, max=5.0)

        if fit_baseline:
            if is_hydrogen:
                params.add(f'baseline_{line_id}', value=1.0, min=0.95, max=1.05)
            else:
                params.add(f'baseline_{line_id}', value=1.0, min=0.97, max=1.03)

        if profile_type=='asym':
            params.add(f'skew1_{line_id}', value=0.0, min=-2.0, max=2.0)
            params.add(f'skew2_{line_id}', value=0.0, min=-2.0, max=2.0)

    for ep in all_epochs:
        e = int(ep)
        params.add(f'rv1_epoch{e}', value=0.0, min=-500, max=500)
        params.add(f'rv2_epoch{e}', value=0.0, min=-500, max=500)

    return params


def compute_model_line(params: Parameters,
                       line_id: str,
                       wavelength: np.ndarray,
                       rv1: float,
                       rv2: float,
                       rest_wavelength: float,
                       profile_type: str='sym'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Combine star1 + star2 flux for a single line
    """
    try:
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

        c_speed = 299792.458
        center1 = rest_wavelength*(1+ rv1/c_speed)
        center2 = rest_wavelength*(1+ rv2/c_speed)

        baseline = params.get(f'baseline_{line_id}', None)
        base_val = baseline.value if baseline else 1.0

        if profile_type=='asym' and (abs(skew1)>1e-8 or abs(skew2)>1e-8):
            prof1 = skewed_voigt_profile(wavelength, -abs(a1), center1, s1, g1, skew1)
            prof2 = skewed_voigt_profile(wavelength, -abs(a2), center2, s2, g2, skew2)
        else:
            prof1 = voigt_profile(wavelength, -abs(a1), center1, s1, g1)
            prof2 = voigt_profile(wavelength, -abs(a2), center2, s2, g2)

        star1_flux = base_val + prof1
        star2_flux = base_val + prof2
        combined = base_val + prof1 + prof2
        return combined, star1_flux, star2_flux

    except Exception as e:
        logger.error(f"Error computing line {line_id}: {e}")
        raise ModelError(f"Failed to compute model line {line_id}")


def get_star_components(params: Parameters,
                        line_id: str,
                        wv_array: np.ndarray,
                        rest_wv: float,
                        rv1: float,
                        rv2: float,
                        profile_type: str='sym'
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return star1_flux, star2_flux for plotting, no mid-fit swap
    (Used by plot_results.py: so it can draw star1 & star2 lines)
    """
    try:
        # same approach as compute_model_line, but we only return star1_flux, star2_flux
        from src.utils import voigt_profile, skewed_voigt_profile
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

        c_speed = 299792.458
        center1 = rest_wv*(1+ rv1/c_speed)
        center2 = rest_wv*(1+ rv2/c_speed)

        baseline = params.get(f'baseline_{line_id}', None)
        base_val = baseline.value if baseline else 1.0

        if profile_type=='asym' and (abs(skew1)>1e-8 or abs(skew2)>1e-8):
            prof1 = skewed_voigt_profile(wv_array, -abs(a1), center1, s1, g1, skew1)
            prof2 = skewed_voigt_profile(wv_array, -abs(a2), center2, s2, g2, skew2)
        else:
            prof1 = voigt_profile(wv_array, -abs(a1), center1, s1, g1)
            prof2 = voigt_profile(wv_array, -abs(a2), center2, s2, g2)

        star1_flux = base_val + prof1
        star2_flux = base_val + prof2
        return star1_flux, star2_flux

    except Exception as e:
        logger.error(f"Error in get_star_components: {str(e)}")
        raise ModelError(f"Failed to compute star components for {line_id}")


def compute_full_model(params: Parameters,
                       wavelengths_dict: Dict[str, np.ndarray],
                       epochs_dict: Dict[str, np.ndarray],
                       central_wavelengths: Dict[str, float],
                       profile_type: str='sym'
) -> Dict[str, np.ndarray]:
    """
    Build the multi-line model across all epochs, no star-swap
    """
    model_fluxes = {}

    import numpy as np
    try:
        all_epochs = np.unique(np.concatenate([epochs_dict[lid] for lid in epochs_dict]))

        # init arrays
        for line_id in wavelengths_dict:
            wv_arr = wavelengths_dict[line_id]
            if len(wv_arr)==0:
                model_fluxes[line_id] = np.array([])
                continue
            baseline = params.get(f'baseline_{line_id}', None)
            base_val = baseline.value if baseline else 1.0
            model_fluxes[line_id] = np.full_like(wv_arr, base_val)

        for epoch in all_epochs:
            rv1_key = f'rv1_epoch{int(epoch)}'
            rv2_key = f'rv2_epoch{int(epoch)}'
            rv1 = params[rv1_key].value
            rv2 = params[rv2_key].value

            for line_id in wavelengths_dict:
                idx = (epochs_dict[line_id]==epoch)
                if not np.any(idx):
                    continue

                wv = wavelengths_dict[line_id][idx]
                rest_wv = central_wavelengths[line_id]

                combined, star1, star2 = compute_model_line(
                    params, line_id, wv, rv1, rv2, rest_wv, profile_type
                )
                model_fluxes[line_id][idx] = combined

        return model_fluxes

    except Exception as e:
        logger.error(f"Error in full model computation: {str(e)}")
        raise ModelError("Failed to compute full model")


def residuals(
        params: Parameters,
        wavelengths_dict: Dict[str, np.ndarray],
        fluxes_dict: Dict[str, np.ndarray],
        epochs_dict: Dict[str, np.ndarray],
        uncertainties_dict: Dict[str, np.ndarray],
        central_wavelengths: Dict[str, float],
        profile_type: str='sym',
        weighted: bool=True
) -> np.ndarray:
    """
    Compute residual vector for the entire dataset, no star-swap
    """
    import numpy as np
    from src.standard_fit_model import analyze_line_quality, get_line_weight

    try:
        model_flux = compute_full_model(
            params,
            wavelengths_dict,
            epochs_dict,
            central_wavelengths,
            profile_type=profile_type
        )

        all_resids = []
        for line_id in wavelengths_dict:
            if len(wavelengths_dict[line_id])==0:
                continue
            obs = fluxes_dict[line_id]
            mod = model_flux[line_id]
            unc = uncertainties_dict[line_id]

            # line weighting
            wave_center = central_wavelengths[line_id]
            line_quality = analyze_line_quality(wavelengths_dict[line_id], obs, unc)
            w_factor = get_line_weight(wave_center, line_quality)

            if weighted:
                unc_eff= unc*w_factor
                core_weight= 1.0 + 2.0*(1.0 - obs)
                core_weight= np.clip(core_weight,1.0,3.0)
                unc_eff /= core_weight
                resid = (obs - mod)/unc_eff
            else:
                resid = (obs - mod)*w_factor

            # clip outliers
            resid = np.clip(resid, -5, 5)
            all_resids.append(resid)

        return np.concatenate(all_resids)
    except Exception as e:
        logger.error(f"Error computing residuals: {str(e)}")
        raise ModelError("Failed to compute residuals")