import numpy as np
from lmfit import Parameters
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, savgol_filter
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.cluster import KMeans

from src.utils import (
    voigt_profile,
    skewed_voigt_profile,
    estimate_noise_level,
    robust_normalize_flux
)

logger = logging.getLogger(__name__)

class ModelError(Exception):
    """Custom exception for model computation errors"""
    pass


def get_line_weight(wavelength: float, flux_quality: float = 1.0) -> float:
    """
    Weighted lines based on line type + data quality
    """
    base_weight = 1.0
    h_lines = [4340.472, 4101.734, 3970.075, 4861.35, 6562.81]
    he_lines = [4026.0, 4471.5, 4921.9, 5875.6]

    for h_wave in h_lines:
        if abs(wavelength - h_wave) < 1.0:
            base_weight = 0.7
            break

    for he_wave in he_lines:
        if abs(wavelength - he_wave) < 1.0:
            base_weight = 1.2
            break

    # flux_quality is from 0..1, scale to 0.5..1
    quality_factor = 0.5 + 0.5 * flux_quality
    return base_weight * quality_factor


def analyze_line_quality(wavelength: np.ndarray,
                         flux: np.ndarray,
                         uncertainty: np.ndarray) -> float:
    """
    Evaluate data quality for the line region
    """
    try:
        snr = np.median(1.0 / uncertainty)
        snr_score = np.clip(snr / 50.0, 0, 1)

        dw = np.diff(wavelength)
        sampling_regularity = 1.0 - np.std(dw) / np.mean(dw)

        completeness = np.sum(np.isfinite(flux)) / len(flux)

        cont_var = np.std(flux[flux > np.median(flux)])
        cont_score = np.exp(-cont_var * 10)

        quality = (0.4 * snr_score +
                   0.2 * sampling_regularity +
                   0.2 * completeness +
                   0.2 * cont_score)
        return np.clip(quality, 0, 1)

    except Exception as e:
        logger.warning(f"Error in line quality analysis: {str(e)}")
        return 0.5


def setup_parameters(
        central_wavelengths: Dict[str, float],
        all_epochs: List[int],
        profile_type: str = 'sym',
        fit_baseline: bool = False,
        initial_state: Optional[Dict] = None
) -> Parameters:
    """
    Basic param setup for a two-Voigt model, no star-swap logic
    """
    from lmfit import Parameters
    params = Parameters()
    init_vals = initial_state or {}

    for line_id, cwv in central_wavelengths.items():
        is_hydrogen = any(str(cwv).startswith(h) for h in ['4340', '4861', '6563'])
        a1_init = init_vals.get(f'a1_{line_id}', -0.8 if is_hydrogen else -0.3)
        a2_init = init_vals.get(f'a2_{line_id}', -0.6 if is_hydrogen else -0.25)

        if is_hydrogen:
            # broader allowed bounds for H lines
            params.add(f'a1_{line_id}', value=a1_init, min=-3.0, max=-0.05)
            params.add(f'sigma1_{line_id}',
                       value=init_vals.get(f'sigma1_{line_id}', 1.5),
                       min=0.2, max=8.0)
            params.add(f'gamma1_{line_id}',
                       value=init_vals.get(f'gamma1_{line_id}', 1.5),
                       min=0.2, max=8.0)

            params.add(f'a2_{line_id}', value=a2_init, min=-3.0, max=-0.05)
            params.add(f'sigma2_{line_id}',
                       value=init_vals.get(f'sigma2_{line_id}', 1.5),
                       min=0.2, max=8.0)
            params.add(f'gamma2_{line_id}',
                       value=init_vals.get(f'gamma2_{line_id}', 1.5),
                       min=0.2, max=8.0)
        else:
            # narrower bounds for He lines
            params.add(f'a1_{line_id}', value=a1_init, min=-2.0, max=-0.02)
            params.add(f'sigma1_{line_id}',
                       value=init_vals.get(f'sigma1_{line_id}', 0.6),
                       min=0.1, max=5.0)
            params.add(f'gamma1_{line_id}',
                       value=init_vals.get(f'gamma1_{line_id}', 0.6),
                       min=0.1, max=5.0)

            params.add(f'a2_{line_id}', value=a2_init, min=-2.0, max=-0.02)
            params.add(f'sigma2_{line_id}',
                       value=init_vals.get(f'sigma2_{line_id}', 0.6),
                       min=0.1, max=5.0)
            params.add(f'gamma2_{line_id}',
                       value=init_vals.get(f'gamma2_{line_id}', 0.6),
                       min=0.1, max=5.0)

        if fit_baseline:
            base_init = init_vals.get(f'baseline_{line_id}', 1.0)
            if is_hydrogen:
                params.add(f'baseline_{line_id}', value=base_init,
                           min=0.95, max=1.05)
            else:
                params.add(f'baseline_{line_id}', value=base_init,
                           min=0.97, max=1.03)

        if profile_type == 'asym':
            # optional skew
            skew1_init = init_vals.get(f'skew1_{line_id}', 0.0)
            params.add(f'skew1_{line_id}', value=skew1_init, min=-2.0, max=2.0)
            skew2_init = init_vals.get(f'skew2_{line_id}', 0.0)
            params.add(f'skew2_{line_id}', value=skew2_init, min=-2.0, max=2.0)

    # add rv1, rv2 for each epoch
    for ep in all_epochs:
        e = int(ep)
        rv1_key = f'rv1_epoch{e}'
        rv2_key = f'rv2_epoch{e}'

        rv1_init = init_vals.get(rv1_key, 0.0)
        rv2_init = init_vals.get(rv2_key, 0.0)

        params.add(rv1_key, value=rv1_init, min=-500, max=500)
        params.add(rv2_key, value=rv2_init, min=-500, max=500)

    return params


def find_initial_rvs(wv: np.ndarray,
                     flux: np.ndarray,
                     rest_wv: float,
                     min_depth: float = 0.02,
                     max_peaks: int = 4) -> List[float]:
    """
    Multi-scale approach to guess initial RVs
    """
    try:
        rv_candidates = set()

        # 1) Gaussian smoothing
        for sigma in [0.5, 1.0, 2.0]:
            smoothed = gaussian_filter1d(flux, sigma=sigma)
            peaks, props = find_peaks(-smoothed, prominence=min_depth,
                                      width=3, distance=5)
            for peak, prom in zip(peaks, props['prominences']):
                if prom>min_depth:
                    wv_peak = wv[peak]
                    rv = (wv_peak/rest_wv - 1.0)*299792.458
                    if -500<=rv<=500:
                        rv_candidates.add(rv)

        # 2) Savitzky-Golay
        for window in [5,7,9]:
            if len(flux)>window:
                smoothed = savgol_filter(flux, window, 3)
                peaks, props = find_peaks(-smoothed, prominence=min_depth,
                                          width=3, distance=5)
                for peak, prom in zip(peaks, props['prominences']):
                    if prom>min_depth:
                        wv_peak = wv[peak]
                        rv = (wv_peak/rest_wv - 1.0)*299792.458
                        if -500<=rv<=500:
                            rv_candidates.add(rv)

        # 3) derivative-based
        smoothed = gaussian_filter1d(flux, sigma=1.0)
        deriv = np.gradient(smoothed)
        deriv2= np.gradient(deriv)
        zero_crosses = np.where(np.diff(np.signbit(deriv)))[0]
        for zc in zero_crosses:
            if zc+1<len(deriv2) and deriv2[zc]<0:
                depth = 1.0 - smoothed[zc]
                if depth>min_depth:
                    rv = (wv[zc]/rest_wv -1.0)*299792.458
                    if -500<=rv<=500:
                        rv_candidates.add(rv)

        rv_list = sorted(rv_candidates, key=abs)
        return rv_list[:max_peaks]

    except Exception as e:
        logger.error(f"Error in initial RV estimation: {str(e)}")
        return [0.0]


def grid_search_rvs(wv: np.ndarray,
                    flux: np.ndarray,
                    rest_wv: float,
                    rv_range: Tuple[float,float]=(-500,500),
                    initial_rvs: Optional[List[float]]=None) -> Tuple[float,float]:
    """
    Coarse + refined grid search for 2-component
    """
    def evaluate_model(rv1: float, rv2: float) -> float:
        shift1 = rest_wv*(1+rv1/299792.458)
        shift2 = rest_wv*(1+rv2/299792.458)
        sigma=1.0 if rest_wv>4400 else 1.5
        prof1= np.exp(-0.5*((wv-shift1)/sigma)**2)
        prof2= np.exp(-0.5*((wv-shift2)/sigma)**2)

        def residual(x):
            return np.sum((flux-(1 + x[0]*prof1 + x[1]*prof2))**2)

        res = minimize(residual, x0=[-0.2,-0.2], bounds=[(-1,0),(-1,0)])
        return residual(res.x)

    if initial_rvs:
        rv_points = np.array(initial_rvs)
        grid_ranges=[]
        for rv in rv_points:
            grid_ranges.extend([rv-50, rv+50])
        if len(grid_ranges)<2:
            grid_ranges = [rv_range[0], rv_range[1]]
    else:
        grid_ranges= [rv_range[0], rv_range[1]]

    best_rv1,best_rv2 = 0.0,0.0
    best_score = float('inf')

    for iteration in range(3):
        if iteration==0:
            n_points=20
            rv_grid= np.linspace(min(grid_ranges), max(grid_ranges), n_points)
        else:
            range_width=100.0/(2**iteration)
            rv_grid= np.linspace(best_rv1-range_width, best_rv1+range_width, 20)

        for rv1 in rv_grid:
            for rv2 in rv_grid:
                if abs(rv1-rv2)<30:
                    continue
                score = evaluate_model(rv1, rv2)
                if score<best_score:
                    best_score=score
                    best_rv1,best_rv2= rv1,rv2

    if best_rv1>best_rv2:
        best_rv1,best_rv2= best_rv2,best_rv1
    return best_rv1,best_rv2


def compute_model_line(params: Parameters,
                       line_id: str,
                       wavelength: np.ndarray,
                       rv1: float,
                       rv2: float,
                       rest_wavelength: float,
                       profile_type: str='sym') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Single line's 2-Voigt combination
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
            logger.error(f"Profile comp error for {line_id}: {str(e)}")
            prof1 = np.exp(-0.5*((wavelength-center1)/s1)**2)*a1
            prof2 = np.exp(-0.5*((wavelength-center2)/s2)**2)*a2

        star1_flux = base_val + prof1
        star2_flux = base_val + prof2
        combined_flux = base_val + prof1 + prof2
        return combined_flux, star1_flux, star2_flux

    except Exception as e:
        logger.error(f"Error in model computation for {line_id}: {str(e)}")
        raise ModelError(f"Failed to compute model for {line_id}")


def compute_full_model(
        params: Parameters,
        wavelengths_dict: Dict[str, np.ndarray],
        epochs_dict: Dict[str, np.ndarray],
        central_wavelengths: Dict[str, float],
        profile_type: str='sym'
) -> Dict[str, np.ndarray]:
    """
    Build the multi-line model across all epochs, no star-swap
    """
    import numpy as np
    model_fluxes= {}

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

        # fill model
        for epoch in all_epochs:
            rv1_key = f'rv1_epoch{int(epoch)}'
            rv2_key = f'rv2_epoch{int(epoch)}'
            rv1 = params[rv1_key].value
            rv2 = params[rv2_key].value

            for line_id in wavelengths_dict:
                idx = (epochs_dict[line_id]==epoch)
                if not np.any(idx):
                    continue
                wv= wavelengths_dict[line_id][idx]
                restwv= central_wavelengths[line_id]

                combined, star1, star2= compute_model_line(params, line_id,
                                                           wv, rv1, rv2,
                                                           restwv, profile_type)
                model_fluxes[line_id][idx] = combined

        return model_fluxes

    except Exception as e:
        logger.error(f"Error in full model: {str(e)}")
        raise ModelError("Failed to build full model")


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
    Residual function with no star-swap mid-fit
    """
    import numpy as np
    try:
        model_flux = compute_full_model(params, wavelengths_dict, epochs_dict,
                                        central_wavelengths, profile_type)
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
                # more weight to line core
                core_weight= 1.0 + 2.0*(1.0 - obs)
                core_weight= np.clip(core_weight,1.0,3.0)
                unc_eff /= core_weight
                resid= (obs - mod)/unc_eff
            else:
                resid= (obs - mod)*w_factor

            resid= np.clip(resid, -5,5)
            all_resids.append(resid)

        return np.concatenate(all_resids)

    except Exception as e:
        logger.error(f"Error computing residuals: {str(e)}")
        raise ModelError("Failed to compute residuals")