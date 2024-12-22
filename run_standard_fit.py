import os
import time
import numpy as np
from tqdm import tqdm
from lmfit import minimize, report_fit
import argparse

from src.utils import (
    find_observation_files,
    load_data_for_epoch,
    find_noise_regions,
    find_line_center
)
from src.standard_fit_model import setup_parameters, residuals
from src.plot_results import report_fit_results

class FitProgress:
    def __init__(self):
        self.iteration = 0
        self.start_time = None

def main():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    parser = argparse.ArgumentParser(description="Run standard fit on observation data.")
    parser.add_argument("--data_dir", type=str, help="Directory containing observation files.",
                        default=os.path.join(base_dir, 'data', 'obs'))
    parser.add_argument("--output_dir", type=str, help="Directory to save output results.",
                        default=os.path.join(base_dir, 'output', 'standard_fit_results'))
    parser.add_argument("--profile_type", type=str, help="Profile type: 'sym' or 'skewed'", default='sym')
    parser.add_argument("--lines", type=str, help="Comma-separated list of line names to fit", default=None)
    parser.add_argument("--unweighted", action='store_true',
                        help="If set, use unweighted residuals instead of weighted.")
    args = parser.parse_args()

    data_directory = args.data_dir
    output_directory = args.output_dir
    profile_type = args.profile_type
    use_weighted = not args.unweighted

    os.makedirs(output_directory, exist_ok=True)

    # Define all available lines
    all_lines_info = {
        'He4471': {'wavelength': 4471.5, 'window': 20.0},
        'He4026': {'wavelength': 4026.0, 'window': 20.0},
        'He4388': {'wavelength': 4388.0, 'window': 20.0},
    }

    # Select lines if user specified
    if args.lines:
        selected_lines = [l.strip() for l in args.lines.split(',') if l.strip()]
        for line in selected_lines:
            if line not in all_lines_info:
                raise ValueError(f"Selected line '{line}' not recognized.")
        spectral_lines = {ln: all_lines_info[ln] for ln in selected_lines}
    else:
        spectral_lines = all_lines_info

    print("Using lines:", list(spectral_lines.keys()))
    print("Searching for observation files in:", data_directory)
    epoch_files = find_observation_files(data_directory)
    if not epoch_files:
        raise ValueError("No observation files found in the specified directory.")

    print(f"Found {len(epoch_files)} files.")

    # Data containers
    wavelengths_line = {}
    fluxes_line = {}
    uncertainties_line = {}
    epochs_line = {}
    noise_regions_dict = {}
    central_wavelengths = {}
    windows = {}
    wavelengths_plot = {}
    fluxes_plot = {}
    epochs_plot = {}

    # Prepare containers for each line
    for name, info in spectral_lines.items():
        line_id = f'line_{int(info["wavelength"]*10)}'
        central_wavelengths[line_id] = info['wavelength']
        windows[line_id] = info['window']
        for dct in [wavelengths_line, fluxes_line, uncertainties_line, epochs_line,
                    wavelengths_plot, fluxes_plot, epochs_plot]:
            dct[line_id] = []
        noise_regions_dict[line_id] = {}

    print("Loading epochs...")
    for (epoch, filepath) in tqdm(epoch_files, desc="Loading data"):
        data = load_data_for_epoch(filepath)
        if len(data) == 0:
            print(f"No valid data in {filepath}, skipping...")
            continue

        # For each spectral line, gather a BROAD region that includes:
        # - The line window
        # - Potential noise region(s)
        for line_id, cwave in central_wavelengths.items():
            w = windows[line_id]

            # 1) We do a broad search for the line center
            #    E.g. we can do a 2x or 3x window if you like
            broad_factor = 3.0
            broad_search_w = broad_factor * w
            broad_min = cwave - broad_search_w/2
            broad_max = cwave + broad_search_w/2

            # Mask the data in [broad_min, broad_max]
            mask_broad = (data['wavelength'] >= broad_min) & (data['wavelength'] <= broad_max)
            wv_broad = data['wavelength'][mask_broad].values
            fv_broad = data['flux'][mask_broad].values
            if len(wv_broad) == 0:
                continue

            # Attempt to find line center within that broad region
            center_detected = find_line_center(wv_broad, fv_broad, absorption=True)
            if center_detected is None:
                continue

            # 2) Now define the final range for actually storing data:
            #    Instead of forcibly clipping to [center - w/2, center + w/2],
            #    we define an even broader region to include noise if needed.
            #    We'll detect noise to see how far it extends.

            final_min = center_detected - w/2
            final_max = center_detected + w/2

            # Find noise region(s) in the entire broad range:
            # We'll do so in the entire broad_min...broad_max region.
            nrs = find_noise_regions(data, final_min, final_max)
            # nrs might contain left or right noise edges outside [final_min, final_max].
            # We'll combine them to define a bigger range.

            # If there's noise outside the line window, let's expand final_min/final_max accordingly
            # We collect the min edge from any noise if it's less than final_min,
            # or the max edge from any noise if it's greater than final_max.
            # This ensures we keep data from noise region.
            for (nm, nM, direction) in (nrs or []):
                if nm < final_min:
                    final_min = nm
                if nM > final_max:
                    final_max = nM

            # Now we have final_min/final_max that covers line window + noise region
            mask_final = (data['wavelength'] >= final_min) & (data['wavelength'] <= final_max)
            wv_final = data['wavelength'][mask_final].values
            fv_final = data['flux'][mask_final].values
            if len(wv_final) == 0:
                continue

            # Estimate noise sigma from the noise region flux
            sigma_values = []
            if nrs:
                for (nm, nM, direction) in nrs:
                    nm_mask = (data['wavelength'] >= nm) & (data['wavelength'] <= nM)
                    noise_flux = data['flux'][nm_mask]
                    sigma_values.append(noise_flux.std())
            sigma_epoch = np.mean(sigma_values) if sigma_values else 0.02  # or 0.025, etc.

            # Store data
            wavelengths_line[line_id].append(wv_final)
            fluxes_line[line_id].append(fv_final)
            uncertainties_line[line_id].append(np.full_like(wv_final, sigma_epoch))
            epochs_line[line_id].append(np.full_like(wv_final, epoch, dtype=int))

            # For plotting (same arrays if you like)
            wavelengths_plot[line_id].append(wv_final)
            fluxes_plot[line_id].append(fv_final)
            epochs_plot[line_id].append(np.full_like(wv_final, epoch, dtype=int))

            # Store the noise regions we found for this epoch
            noise_regions_dict[line_id][epoch] = nrs if nrs else []

    # Concatenate arrays for each line
    for line_id in central_wavelengths:
        for dct in [wavelengths_line, fluxes_line, uncertainties_line, epochs_line,
                    wavelengths_plot, fluxes_plot, epochs_plot]:
            if len(dct[line_id])>0:
                dct[line_id] = np.concatenate(dct[line_id])
            else:
                dct[line_id] = np.array([])

    # Collect all unique epochs
    all_epochs = np.unique(np.concatenate(
        [ep for ep in epochs_line.values() if len(ep)>0])) if len(epochs_line)>0 else []

    print(f"Found {len(all_epochs)} unique epochs.")

    # Fit
    from src.standard_fit_model import setup_parameters, residuals
    params = setup_parameters(central_wavelengths, all_epochs, profile_type)

    fit_progress = FitProgress()
    fit_progress.start_time = time.time()

    print("Performing single-stage leastsq fit...")
    result = minimize(
        residuals, params,
        args=(wavelengths_line, fluxes_line, epochs_line, uncertainties_line,
              central_wavelengths, profile_type, fit_progress, use_weighted),
        method='leastsq', max_nfev=10000, ftol=1e-6, xtol=1e-6, calc_covar=True
    )

    print("\nFit Report:")
    report_fit(result)

    # Summaries and plots
    from src.plot_results import report_fit_results
    results_df, chi_square_df = report_fit_results(
        result,
        wavelengths_line, fluxes_line, uncertainties_line,
        epochs_line, central_wavelengths,
        noise_regions_dict, windows,
        wavelengths_plot, fluxes_plot, epochs_plot,
        output_directory
    )

if __name__ == "__main__":
    main()
