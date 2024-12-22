import os
import time
import numpy as np
from tqdm import tqdm
import argparse
from lmfit import minimize, report_fit

from src.utils import (
    find_observation_files,
    load_data_for_epoch,
    find_noise_regions,
    find_line_center
)
from src.ratio_fit_model import setup_parameters, residuals
from src.plot_results import report_fit_results

class FitProgress:
    def __init__(self):
        self.iteration = 0
        self.start_time = None

def main():
    # Base directory
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    parser = argparse.ArgumentParser(description="Run ratio-constrained fit on observation data.")
    parser.add_argument("--data_dir", type=str, help="Directory containing observation files.",
                        default=os.path.join(base_dir, 'data', 'obs'))
    parser.add_argument("--output_dir", type=str, help="Directory to save output results.",
                        default=os.path.join(base_dir, 'output', 'ratio_constrained_results'))
    parser.add_argument("--profile_type", type=str, help="Profile type: 'sym' or 'skewed'",
                        default='sym')
    parser.add_argument("--lines", type=str, help="Comma-separated list of line names to fit",
                        default=None)
    parser.add_argument("--unweighted", action='store_true',
                        help="Use unweighted residuals instead of weighted by uncertainties.")
    args = parser.parse_args()

    data_directory = args.data_dir
    output_directory = args.output_dir
    profile_type = args.profile_type
    use_weighted = not args.unweighted
    os.makedirs(output_directory, exist_ok=True)

    # Lines info for ratio constrained
    # You can tweak the 'window' values as needed
    all_lines_info = {
        'He4471': {'wavelength': 4471.5, 'window': 20.0},
        'He4026': {'wavelength': 4026.0, 'window': 20.0},
        'He4388': {'wavelength': 4388.0, 'window': 20.0},
    }

    if args.lines:
        selected_lines = [l.strip() for l in args.lines.split(',') if l.strip()]
        for line in selected_lines:
            if line not in all_lines_info:
                raise ValueError(f"Selected line '{line}' not recognized.")
        spectral_lines = {ln: all_lines_info[ln] for ln in selected_lines}
    else:
        spectral_lines = all_lines_info

    print("Performing Ratio-Constrained Fit on lines:", list(spectral_lines.keys()))
    epoch_files = find_observation_files(data_directory)
    if not epoch_files:
        raise ValueError("No observation files found in the specified directory.")

    # Prepare data containers
    wavelengths_line = {}
    fluxes_line = {}
    uncertainties_line = {}
    epochs_line = {}
    wavelengths_plot = {}
    fluxes_plot = {}
    epochs_plot = {}
    noise_regions = {}
    central_wavelengths = {}
    windows = {}

    # Initialize containers for each line
    for name, info in spectral_lines.items():
        line_id = f'line_{int(info["wavelength"]*10)}'
        central_wavelengths[line_id] = info['wavelength']
        windows[line_id] = info['window']
        for dct in [
            wavelengths_line, fluxes_line, uncertainties_line, epochs_line,
            wavelengths_plot, fluxes_plot, epochs_plot
        ]:
            dct[line_id] = []
        noise_regions[line_id] = {}

    print("Loading epochs...")
    for (epoch, filepath) in tqdm(epoch_files, desc="Loading data"):
        data = load_data_for_epoch(filepath)
        if len(data) == 0:
            continue

        # For each line, gather a BROAD region that includes the line + potential noise
        for line_id, cwave in central_wavelengths.items():
            w = windows[line_id]

            # 1) Broad region for searching line center
            broad_factor = 3.0  # or 2.0, 4.0, etc.
            broad_search_w = broad_factor * w
            broad_min = cwave - broad_search_w/2
            broad_max = cwave + broad_search_w/2

            mask_broad = (data['wavelength'] >= broad_min) & (data['wavelength'] <= broad_max)
            wv_broad = data['wavelength'][mask_broad].values
            fv_broad = data['flux'][mask_broad].values
            if len(wv_broad) == 0:
                continue

            # Attempt to find line center
            center_detected = find_line_center(wv_broad, fv_broad, absorption=True)
            if center_detected is None:
                continue

            # 2) Define final_min, final_max ~ [center ± w/2]
            final_min = center_detected - w/2
            final_max = center_detected + w/2

            # 3) Detect noise region(s)
            nrs = find_noise_regions(data, final_min, final_max)
            # Possibly expand final_min/final_max to include noise edges
            for (nm, nM, direction) in (nrs or []):
                if nm < final_min:
                    final_min = nm
                if nM > final_max:
                    final_max = nM

            # Subset data in [final_min, final_max]
            mask_final = (data['wavelength'] >= final_min) & (data['wavelength'] <= final_max)
            wv = data['wavelength'][mask_final].values
            fv = data['flux'][mask_final].values
            if len(wv) == 0:
                continue

            # Compute sigma
            sigma_values = []
            if nrs:
                for (nm, nM, direction) in nrs:
                    nm_mask = (data['wavelength'] >= nm) & (data['wavelength'] <= nM)
                    noise_flux = data['flux'][nm_mask]
                    sigma_values.append(noise_flux.std())
            sigma_epoch = np.mean(sigma_values) if sigma_values else 0.02

            # Store data arrays
            wavelengths_line[line_id].append(wv)
            fluxes_line[line_id].append(fv)
            uncertainties_line[line_id].append(np.full_like(wv, sigma_epoch))
            epochs_line[line_id].append(np.full_like(wv, epoch, dtype=int))

            # For plotting
            wavelengths_plot[line_id].append(wv)
            fluxes_plot[line_id].append(fv)
            epochs_plot[line_id].append(np.full_like(wv, epoch, dtype=int))

            # Keep track of noise regions for plotting
            noise_regions[line_id][epoch] = nrs if nrs else []

    # Concatenate
    for dct in [
        wavelengths_line, fluxes_line, uncertainties_line, epochs_line,
        wavelengths_plot, fluxes_plot, epochs_plot
    ]:
        for lid in central_wavelengths:
            if len(dct[lid])>0:
                dct[lid] = np.concatenate(dct[lid])
            else:
                dct[lid] = np.array([])

    all_epochs = np.unique(np.concatenate(
        [ep for ep in epochs_line.values() if len(ep)>0])) if len(epochs_line)>0 else []

    print(f"Found {len(all_epochs)} unique epochs.")

    # Setup ratio fit
    params = setup_parameters(central_wavelengths, all_epochs, profile_type)
    fit_progress = FitProgress()
    fit_progress.start_time = time.time()

    print("Performing single-stage leastsq fit for ratio-constrained model...")
    result = minimize(
        residuals, params,
        args=(
            wavelengths_line, fluxes_line, epochs_line, uncertainties_line,
            central_wavelengths, profile_type, fit_progress, use_weighted
        ),
        method='leastsq',
        max_nfev=10000,
        ftol=1e-6,
        xtol=1e-6,
        calc_covar=True
    )

    print("\nFit Report (Ratio-Constrained):")
    report_fit(result)

    # Summaries
    results_df, chi_square_df = report_fit_results(
        result,
        wavelengths_line, fluxes_line, uncertainties_line,
        epochs_line, central_wavelengths,
        noise_regions, windows,
        wavelengths_plot, fluxes_plot, epochs_plot,
        output_directory
    )

if __name__ == "__main__":
    main()




