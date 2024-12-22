import numpy as np
from lmfit import Parameters
from src.voigt import voigt_profile  # if needed, else define voigt_profile here

c = 299792.458
min_width = 1e-5

def get_initial_rv_guesses(epoch, n_epochs):
    phase = epoch / n_epochs * 2 * np.pi
    rv1 = 50 * np.sin(phase)
    rv2 = -rv1
    return rv1, rv2

def doppler_shift(wavelength, rv):
    return wavelength * (1 + rv / c)

def setup_parameters(central_wavelengths, all_epochs):
    params = Parameters()
    # line params
    for line_id in central_wavelengths:
        params.add(f'a1_{line_id}', value=-0.3, min=-1.0, max=0)
        params.add(f'sigma1_{line_id}', value=0.5, min=min_width, max=3)
        params.add(f'gamma1_{line_id}', value=0.5, min=min_width, max=3)
        params.add(f'a2_{line_id}', value=-0.2, min=-1.0, max=0)
        params.add(f'sigma2_{line_id}', value=0.5, min=min_width, max=3)
        params.add(f'gamma2_{line_id}', value=0.5, min=min_width, max=3)

    # rv params
    for epoch in all_epochs:
        epoch_int = int(epoch)
        rv1_init, rv2_init = get_initial_rv_guesses(epoch_int, len(all_epochs))
        params.add(f'rv1_epoch{epoch_int}', value=rv1_init, min=-400, max=400)
        params.add(f'rv2_epoch{epoch_int}', value=rv2_init, min=-400, max=400)

    return params

def global_model(params, wavelengths_dict, epochs_dict, central_wavelengths):
    model_fluxes = {line_id: np.ones_like(wavelengths)
                    for line_id, wavelengths in wavelengths_dict.items()}

    all_epochs = np.unique(np.concatenate([epochs for epochs in epochs_dict.values()]))

    for epoch in all_epochs:
        epoch_int = int(epoch)
        rv1 = params[f'rv1_epoch{epoch_int}'].value
        rv2 = params[f'rv2_epoch{epoch_int}'].value

        for line_id in wavelengths_dict:
            idx = epochs_dict[line_id] == epoch
            if np.any(idx):
                wavelengths = wavelengths_dict[line_id][idx]
                cw = central_wavelengths[line_id]
                shifted_center1 = doppler_shift(cw, rv1)
                shifted_center2 = doppler_shift(cw, rv2)

                a1 = params[f'a1_{line_id}'].value
                sigma1 = params[f'sigma1_{line_id}'].value
                gamma1 = params[f'gamma1_{line_id}'].value
                a2 = params[f'a2_{line_id}'].value
                sigma2 = params[f'sigma2_{line_id}'].value
                gamma2 = params[f'gamma2_{line_id}'].value

                profile1 = voigt_profile(wavelengths, -abs(a1), shifted_center1, sigma1, gamma1)
                profile2 = voigt_profile(wavelengths, -abs(a2), shifted_center2, sigma2, gamma2)

                model_fluxes[line_id][idx] += profile1 + profile2

    return model_fluxes

def calculate_line_weights(fluxes_dict):
    weights = {}
    for line_id, fluxes in fluxes_dict.items():
        line_depth = 1.0 - np.min(fluxes)
        weights[line_id] = max(line_depth, 0.1)
    mean_w = np.mean(list(weights.values()))
    return {lid: w/mean_w for lid,w in weights.items()}

def residuals(params, wavelengths_dict, fluxes_dict, epochs_dict,
              uncertainties_dict, central_wavelengths, fit_progress):
    if fit_progress is not None:
        fit_progress.iteration += 1

    line_weights = calculate_line_weights(fluxes_dict)
    model_fluxes = global_model(params, wavelengths_dict, epochs_dict, central_wavelengths)

    all_residuals = []
    for line_id in wavelengths_dict:
        weighted_unc = uncertainties_dict[line_id]/line_weights[line_id]
        res_line = (fluxes_dict[line_id]-model_fluxes[line_id])/weighted_unc
        all_residuals.append(res_line)

    res = np.concatenate(all_residuals)
    return res
