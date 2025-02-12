import numpy as np
from lmfit import Parameters
import logging
from typing import Dict, List, Tuple, Optional
from scipy.stats import mode
from scipy.optimize import minimize
from sklearn.linear_model import TheilSenRegressor

from src.utils import (
    voigt_profile,
    skewed_voigt_profile,
    estimate_noise_level,
    robust_normalize_flux
)
# If you use line quality or other weighting from standard_fit_model:
from src.standard_fit_model import analyze_line_quality, get_line_weight


logger = logging.getLogger(__name__)


class RatioModelError(Exception):
    """Custom exception for ratio-based model computation errors"""
    pass


def get_line_weight_ratio(wavelength: float,
                          flux_quality: float = 1.0,
                          ratio_uncertainty: Optional[float] = 0.0) -> float:
    """
    Weighted lines with ratio-based penalty.

    If ratio_uncertainty is None, treat it as 0.0 to avoid TypeError.
    """
    if ratio_uncertainty is None:
        ratio_uncertainty = 0.0

    base_weight = 1.0
    h_lines = [4340.472, 4101.734, 3970.075, 4861.35, 6562.81]
    he_lines = [4026.0, 4471.5, 4921.9, 5875.6]

    # If it's near a hydrogen line, lower base weight
    for h_wave in h_lines:
        if abs(wavelength - h_wave) < 1.0:
            base_weight = 0.5
            break

    # If it's near a helium line, raise base weight
    for he_wave in he_lines:
        if abs(wavelength - he_wave) < 1.0:
            base_weight = 1.5
            break

    # flux_quality from 0..1 => map to 0.5..1.0
    quality_factor = 0.5 + 0.5 * flux_quality

    # ratio_uncertainty is typically param['ratio'].stderr
    # If it's large, we reduce the weight
    ratio_factor = np.exp(-2.0 * ratio_uncertainty)

    return base_weight * quality_factor * ratio_factor


def estimate_ratio_vsys(standard_params: Dict[str, Tuple[float, float, float]],
                        min_epochs: int = 3) -> Tuple[float, float, Dict[int, float]]:
    """
    Derive ratio & v_sys from standard-fit results
    """
    try:
        rv_dict = {}
        for name, (val, _, _) in standard_params.items():
            if name.startswith("rv1_epoch"):
                ep = int(name.replace("rv1_epoch",""))
                rv_dict.setdefault(ep, [None, None])[0] = val
            elif name.startswith("rv2_epoch"):
                ep = int(name.replace("rv2_epoch",""))
                rv_dict.setdefault(ep, [None, None])[1] = val

        if len(rv_dict) < min_epochs:
            logger.warning(f"Too few epochs ({len(rv_dict)}) for ratio estimation")
            return 1.0, 0.0, {ep: (vals[1] if vals[1] else 0.0)
                              for ep, vals in rv_dict.items()}

        rv1_list = []
        rv2_list = []
        for ep, (r1, r2) in sorted(rv_dict.items()):
            if r1 is not None and r2 is not None:
                rv1_list.append(r1)
                rv2_list.append(r2)

        if len(rv1_list) < min_epochs:
            logger.warning("Not enough paired RV1/RV2 to estimate ratio => default=1.0")
            return 1.0, 0.0, {ep: (vals[1] if vals[1] else 0.0)
                              for ep, vals in rv_dict.items()}

        X = np.array(rv2_list).reshape(-1, 1)
        y = np.array(rv1_list)
        X_with_intercept = np.column_stack([X, np.ones_like(X)])
        reg = TheilSenRegressor(random_state=42)
        reg.fit(X_with_intercept, y)

        slope = reg.coef_[0]
        intercept = reg.intercept_

        ratio_est = -slope  # from rv1 = intercept + slope*rv2 => ratio = -slope
        if abs(ratio_est + 1) < 1e-8:
            ratio_est = 1.0
            v_sys_est = 0.0
        else:
            v_sys_est = - intercept / (ratio_est + 1)
        ratio_est = np.clip(ratio_est, 0.01, 20.0)

        # Initialize rv2 values
        rv2_init_dict = {}
        for ep, (r1, r2) in rv_dict.items():
            if r2 is not None:
                rv2_init_dict[ep] = r2
            elif r1 is not None:
                rv2_init_dict[ep] = -(r1 + v_sys_est) / ratio_est - v_sys_est
            else:
                rv2_init_dict[ep] = 0.0

        return ratio_est, v_sys_est, rv2_init_dict

    except Exception as e:
        logger.error(f"Error in ratio estimation: {str(e)}")
        return 1.0, 0.0, {}


def setup_parameters(
        central_wavelengths: Dict[str, float],
        all_epochs: List[int],
        profile_type: str = 'sym',
        fit_baseline: bool = False,
        initial_rvs: Optional[Dict[int, float]] = None,
        initial_state: Optional[Dict[str, Tuple[float, float, float]]] = None,
        ratio_estimate: float = 1.0,
        vsys_estimate: float = 0.0
) -> Parameters:
    """
    Parameter setup for ratio-based fitting.
    We handle tuple -> float so LMFIT doesn't see a tuple.
    """
    from lmfit import Parameters

    def safe_val(dic, key, default):
        """Extract a float from dic[key] which might be (val, min, max)."""
        val = dic.get(key, default)
        if isinstance(val, tuple):
            val = val[0]
        return val

    params = Parameters()
    init_vals = initial_state or {}

    for line_id, cwv in central_wavelengths.items():
        is_hydrogen = any(str(cwv).startswith(h) for h in ['4340', '4861', '6563'])

        a1_init = safe_val(init_vals, f'a1_{line_id}', -0.8 if is_hydrogen else -0.3)
        a2_init = safe_val(init_vals, f'a2_{line_id}', -0.6 if is_hydrogen else -0.25)

        if is_hydrogen:
            # broader lines for H
            params.add(f'a1_{line_id}', value=a1_init, min=-3.0, max=-0.05)
            params.add(f'sigma1_{line_id}',
                       value=safe_val(init_vals, f'sigma1_{line_id}',1.5),
                       min=0.2, max=8.0)
            params.add(f'gamma1_{line_id}',
                       value=safe_val(init_vals, f'gamma1_{line_id}',1.5),
                       min=0.2, max=8.0)

            params.add(f'a2_{line_id}', value=a2_init, min=-3.0, max=-0.05)
            params.add(f'sigma2_{line_id}',
                       value=safe_val(init_vals, f'sigma2_{line_id}',1.5),
                       min=0.2, max=8.0)
            params.add(f'gamma2_{line_id}',
                       value=safe_val(init_vals, f'gamma2_{line_id}',1.5),
                       min=0.2, max=8.0)

            if fit_baseline:
                base_init = safe_val(init_vals, f'baseline_{line_id}',1.0)
                params.add(f'baseline_{line_id}', value=base_init, min=0.95, max=1.05)
        else:
            # narrower lines for He
            params.add(f'a1_{line_id}', value=a1_init, min=-2.0, max=-0.02)
            params.add(f'sigma1_{line_id}',
                       value=safe_val(init_vals, f'sigma1_{line_id}',0.6),
                       min=0.1, max=5.0)
            params.add(f'gamma1_{line_id}',
                       value=safe_val(init_vals, f'gamma1_{line_id}',0.6),
                       min=0.1, max=5.0)

            params.add(f'a2_{line_id}', value=a2_init, min=-2.0, max=-0.02)
            params.add(f'sigma2_{line_id}',
                       value=safe_val(init_vals, f'sigma2_{line_id}',0.6),
                       min=0.1, max=5.0)
            params.add(f'gamma2_{line_id}',
                       value=safe_val(init_vals, f'gamma2_{line_id}',0.6),
                       min=0.1, max=5.0)

            if fit_baseline:
                base_init = safe_val(init_vals, f'baseline_{line_id}',1.0)
                params.add(f'baseline_{line_id}', value=base_init, min=0.97, max=1.03)

        if profile_type=='asym':
            sk1_init = safe_val(init_vals, f'skew1_{line_id}',0.0)
            params.add(f'skew1_{line_id}', value=sk1_init, min=-2.0, max=2.0)
            sk2_init = safe_val(init_vals, f'skew2_{line_id}',0.0)
            params.add(f'skew2_{line_id}', value=sk2_init, min=-2.0, max=2.0)

    # add ratio param
    ratio_init = safe_val(init_vals, 'ratio', ratio_estimate)
    ratio_init = np.clip(ratio_init,0.01,20.0)
    params.add('ratio', value=ratio_init, min=0.01, max=20.0)

    # add v_sys param
    vsys_init = safe_val(init_vals, 'v_sys', vsys_estimate)
    params.add('v_sys', value=vsys_init, min=-500.0, max=500.0)

    # add rv2 param for each epoch
    initial_rvs = initial_rvs or {}
    for ep in all_epochs:
        e = int(ep)
        rv2_init = initial_rvs.get(e, 0.0)
        rv2_init = safe_val(init_vals, f'rv2_epoch{e}', rv2_init)
        params.add(f'rv2_epoch{e}', value=rv2_init, min=-500.0, max=500.0)

    return params


def fit_resampled_data(initial_params: Parameters,
                       wavelengths_dict: Dict[str, np.ndarray],
                       fluxes_dict: Dict[str, np.ndarray],
                       epochs_dict: Dict[str, np.ndarray],
                       uncertainties_dict: Dict[str, np.ndarray],
                       central_wavelengths: Dict[str, float]) -> Optional[Parameters]:
    """
    For bootstrap resampling
    """
    from lmfit import Minimizer

    try:
        # copy param structure
        params = Parameters()
        for name, param in initial_params.items():
            params.add(name, value=param.value,
                       min=param.min, max=param.max)

        from src.ratio_fit_model import residuals
        minimizer = Minimizer(
            residuals,
            params,
            fcn_args=(wavelengths_dict, fluxes_dict, epochs_dict,
                      uncertainties_dict, central_wavelengths),
            fcn_kws={'profile_type':'sym', 'weighted':True}
        )
        result = minimizer.minimize(method='leastsq', max_nfev=100)
        if result.success:
            return result.params
        else:
            return None
    except Exception as e:
        logger.error(f"Error in bootstrap fit: {str(e)}")
        return None


def estimate_parameter_uncertainties(params: Parameters,
                                     wavelengths_dict: Dict[str, np.ndarray],
                                     fluxes_dict: Dict[str, np.ndarray],
                                     epochs_dict: Dict[str, np.ndarray],
                                     uncertainties_dict: Dict[str, np.ndarray],
                                     central_wavelengths: Dict[str, float]) -> Dict[str, float]:
    """
    Basic bootstrap for ratio fit
    """
    n_bootstrap = 100
    param_samples = {name: [] for name in params.keys()}

    try:
        for _ in range(n_bootstrap):
            flux_resampled = {}
            for line_id in fluxes_dict:
                flux = fluxes_dict[line_id]
                unc = uncertainties_dict[line_id]
                noise = np.random.normal(0,1,size=len(flux))*unc
                flux_resampled[line_id] = flux + noise

            result_params = fit_resampled_data(params, wavelengths_dict,
                                               flux_resampled,
                                               epochs_dict, uncertainties_dict,
                                               central_wavelengths)
            if result_params is not None:
                for name in params.keys():
                    param_samples[name].append(result_params[name].value)

        uncertainties = {}
        for name in params.keys():
            if len(param_samples[name])>0:
                uncertainties[name] = np.std(param_samples[name])
            else:
                uncertainties[name] = 0.0
        return uncertainties
    except Exception as e:
        logger.error(f"Error in uncertainty estimation: {str(e)}")
        return {name: 0.0 for name in params.keys()}


def compute_rv1_from_ratio(rv2: float, ratio: float, v_sys: float) -> float:
    """
    rv1 = - ratio*(rv2 + v_sys) - v_sys
    """
    return -ratio*(rv2 + v_sys) - v_sys


def compute_model_components(params: Parameters,
                             wavelength: np.ndarray,
                             rv2: float,
                             rest_wavelength: float,
                             line_id: str,
                             profile_type: str='sym'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (combined, star1_flux, star2_flux) for ratio-based approach
    """
    try:
        # ratio param might not have a .stderr, so we ignore that here
        ratio = params['ratio'].value
        v_sys = params['v_sys'].value
        rv1 = compute_rv1_from_ratio(rv2, ratio, v_sys)

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

        c_speed=299792.458
        center1 = rest_wavelength*(1+ rv1/c_speed)
        center2 = rest_wavelength*(1+ rv2/c_speed)

        baseline = params.get(f'baseline_{line_id}', None)
        base_val = baseline.value if baseline else 1.0

        try:
            if profile_type=='asym' and (abs(skew1)>1e-8 or abs(skew2)>1e-8):
                prof1 = skewed_voigt_profile(wavelength, -abs(a1), center1, s1, g1, skew1)
                prof2 = skewed_voigt_profile(wavelength, -abs(a2), center2, s2, g2, skew2)
            else:
                prof1 = voigt_profile(wavelength, -abs(a1), center1, s1, g1)
                prof2 = voigt_profile(wavelength, -abs(a2), center2, s2, g2)
        except Exception as e:
            logger.error(f"Profile comp error: {str(e)}")
            prof1 = -abs(a1)*np.exp(-0.5*((wavelength-center1)/s1)**2)
            prof2 = -abs(a2)*np.exp(-0.5*((wavelength-center2)/s2)**2)

        star1_flux = base_val + prof1
        star2_flux = base_val + prof2
        combined_flux= base_val + prof1 + prof2
        return combined_flux, star1_flux, star2_flux
    except Exception as e:
        logger.error(f"Error in model computation: {str(e)}")
        raise RatioModelError(f"Failed to compute model components: {str(e)}")


def compute_full_model(
        params: Parameters,
        wavelengths_dict: Dict[str, np.ndarray],
        epochs_dict: Dict[str, np.ndarray],
        central_wavelengths: Dict[str, float],
        profile_type: str='sym'
) -> Dict[str, np.ndarray]:
    """
    Build ratio-based model across lines + epochs
    """
    import numpy as np
    model_fluxes = {}

    try:
        for line_id in wavelengths_dict:
            wv_arr = wavelengths_dict[line_id]
            if len(wv_arr)==0:
                model_fluxes[line_id] = np.array([])
                continue
            baseline = params.get(f'baseline_{line_id}', None)
            base_val = baseline.value if baseline else 1.0
            model_fluxes[line_id] = np.full_like(wv_arr, base_val)

        all_epochs = np.unique(np.concatenate(list(epochs_dict.values())))

        for epoch in all_epochs:
            rv2_key = f'rv2_epoch{int(epoch)}'
            if rv2_key not in params:
                continue
            rv2_val = params[rv2_key].value

            for line_id in wavelengths_dict:
                idx = (epochs_dict[line_id]==epoch)
                if not np.any(idx):
                    continue
                wv = wavelengths_dict[line_id][idx]
                rest_wv = central_wavelengths[line_id]

                combined, star1, star2 = compute_model_components(
                    params, wv, rv2_val, rest_wv, line_id, profile_type
                )
                model_fluxes[line_id][idx] = combined

        return model_fluxes
    except Exception as e:
        logger.error(f"Error in full model computation: {str(e)}")
        raise RatioModelError(f"Failed to compute full model: {str(e)}")


def residuals(
        params: Parameters,
        wavelengths_dict: Dict[str, np.ndarray],
        fluxes_dict: Dict[str, np.ndarray],
        epochs_dict: Dict[str, np.ndarray],
        uncertainties_dict: Dict[str, np.ndarray],
        central_wavelengths: Dict[str, float],
        profile_type: str='sym',
        weighted: bool=True,
        fit_baseline: bool=False
) -> np.ndarray:
    """
    Residual function for ratio-based approach, ignoring star-swap mid-fit
    """
    import numpy as np
    try:
        model_flux = compute_full_model(
            params,
            wavelengths_dict,
            epochs_dict,
            central_wavelengths,
            profile_type
        )

        all_residuals = []
        for line_id in wavelengths_dict:
            if len(wavelengths_dict[line_id])==0:
                continue

            obs = fluxes_dict[line_id]
            mod = model_flux[line_id]
            unc = uncertainties_dict[line_id]

            # line quality
            line_quality = analyze_line_quality(wavelengths_dict[line_id],
                                                obs, unc)
            wave_center = central_wavelengths[line_id]

            # ratio param might not have a valid .stderr => fallback to 0.0
            ratio_stderr = params['ratio'].stderr
            if ratio_stderr is None:
                ratio_stderr = 0.0

            weight = get_line_weight_ratio(wave_center, line_quality, ratio_stderr)

            if weighted:
                unc_eff = unc * weight
                core_weight= 1.0 + 2.0*(1.0 - obs)
                core_weight= np.clip(core_weight,1.0,3.0)
                unc_eff /= core_weight
                residual = (obs - mod)/unc_eff
            else:
                residual = (obs - mod)*weight

            # clip outliers
            residual= np.clip(residual, -5,5)
            all_residuals.append(residual)

        return np.concatenate(all_residuals)

    except Exception as e:
        logger.error(f"Error calculating residuals: {str(e)}")
        raise RatioModelError(f"Failed to calculate residuals: {str(e)}")


def check_component_consistency(component_checks: Dict[str, Dict]) -> None:
    """
    Optional cross-check if you like, but no star-swap mid-fit
    """
    import numpy as np
    try:
        depth_ratios= []
        for ck, data in component_checks.items():
            d1= 1.0 - data['star1_depth']
            d2= 1.0 - data['star2_depth']
            if d1>0 and d2>0:
                depth_ratios.append(d1/d2)
        if depth_ratios:
            ratio_std= np.std(depth_ratios)
            if ratio_std>0.3:
                logger.warning("Large variation in depth ratio across lines/epochs")
    except Exception as e:
        logger.error(f"Error in check_component_consistency: {str(e)}")