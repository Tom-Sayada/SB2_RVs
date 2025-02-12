#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from lmfit import Minimizer
import matplotlib.pyplot as plt
from datetime import datetime

from src.utils import find_observation_files, load_data_for_epoch
from src.model_builder import setup_parameters, residuals


def run_single_test(
        wavelengths_dict,
        fluxes_dict,
        epochs_dict,
        uncertainties_dict,
        central_wavelengths,
        niter,
        temperature,
        stepsize,
        method='basinhopping'
):
    """Run a single optimization test with given parameters"""

    # Setup initial parameters
    params = setup_parameters(
        central_wavelengths=central_wavelengths,
        all_epochs=np.unique(np.concatenate([epochs_dict[lid] for lid in epochs_dict]))
    )

    # Create minimizer
    minimizer = Minimizer(
        residuals,
        params,
        fcn_args=(wavelengths_dict, fluxes_dict, epochs_dict,
                  uncertainties_dict, central_wavelengths),
        fcn_kws={'weighted': True}
    )

    try:
        # Basin-hopping stage
        result_bh = minimizer.minimize(
            method=method,
            niter=niter,
            T=temperature,
            stepsize=stepsize,
            minimizer_kwargs={'method': 'L-BFGS-B'}
        )

        # Refinement stage
        final_result = minimizer.minimize(
            method='leastsq',
            params=result_bh.params
        )

        return {
            'success': final_result.success,
            'chi_square': final_result.chisqr,
            'red_chi': final_result.redchi,
            'nfev': final_result.nfev,
            'error': None
        }

    except Exception as e:
        return {
            'success': False,
            'chi_square': np.inf,
            'red_chi': np.inf,
            'nfev': 0,
            'error': str(e)
        }


def test_optimization_params(data_dir, output_dir=None):
    """
    Test different optimization parameters and save results
    """
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        output_dir = f"optimization_test_results_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    # Load data
    epoch_files = find_observation_files(data_dir)
    if not epoch_files:
        raise ValueError(f"No observation files found in {data_dir}")

    # Example spectral lines
    spectral_lines = {
        'He4471': {'rest_wave': 4471.5, 'window': 20.0},
        'He4026': {'rest_wave': 4026.0, 'window': 20.0},
        'He4388': {'rest_wave': 4388.0, 'window': 20.0},
    }

    # Build data arrays (simplified for testing)
    wavelengths_dict = {}
    fluxes_dict = {}
    epochs_dict = {}
    uncertainties_dict = {}

    for ep, filepath in epoch_files[:3]:  # Use first 3 epochs for speed
        df = load_data_for_epoch(filepath)
        if df.empty:
            continue

        for ln_name, ln_info in spectral_lines.items():
            line_id = f"line_{int(ln_info['rest_wave'] * 10)}"

            # Basic windowing
            rest_wave = ln_info['rest_wave']
            half_window = ln_info['window'] / 2
            mask = (df['wavelength'] >= rest_wave - half_window) & \
                   (df['wavelength'] <= rest_wave + half_window)

            if not np.any(mask):
                continue

            wv = df['wavelength'][mask].values
            fl = df['flux'][mask].values

            wavelengths_dict.setdefault(line_id, []).append(wv)
            fluxes_dict.setdefault(line_id, []).append(fl)
            epochs_dict.setdefault(line_id, []).append(np.full_like(wv, ep))
            uncertainties_dict.setdefault(line_id, []).append(np.full_like(wv, 0.02))

    # Flatten arrays
    for d in [wavelengths_dict, fluxes_dict, epochs_dict, uncertainties_dict]:
        for k in d:
            if len(d[k]) > 0:
                d[k] = np.concatenate(d[k])
            else:
                d[k] = np.array([])

    # Map line ID -> rest wavelength
    central_wavelengths = {
        f"line_{int(v['rest_wave'] * 10)}": v['rest_wave']
        for v in spectral_lines.values()
    }

    # Parameter grid
    param_grid = {
        'niter': [5, 10, 15],
        'temperature': [1.0, 5.0, 10.0],
        'stepsize': [0.1, 0.3, 0.5]
    }

    # Run tests
    results = []
    total_tests = np.prod([len(v) for v in param_grid.values()])

    with tqdm(total=total_tests) as pbar:
        for niter in param_grid['niter']:
            for temp in param_grid['temperature']:
                for step in param_grid['stepsize']:
                    # Run 3 times for each combination
                    for trial in range(3):
                        result = run_single_test(
                            wavelengths_dict,
                            fluxes_dict,
                            epochs_dict,
                            uncertainties_dict,
                            central_wavelengths,
                            niter=niter,
                            temperature=temp,
                            stepsize=step
                        )

                        results.append({
                            'niter': niter,
                            'temperature': temp,
                            'stepsize': step,
                            'trial': trial,
                            **result
                        })

                    pbar.update(1)

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Save detailed results
    results_file = os.path.join(output_dir, 'optimization_results.csv')
    df_results.to_csv(results_file, index=False)

    # Compute statistics per parameter combination
    stats = df_results.groupby(['niter', 'temperature', 'stepsize']).agg({
        'success': 'mean',
        'chi_square': ['mean', 'std'],
        'red_chi': ['mean', 'std'],
        'nfev': 'mean'
    }).round(3)

    stats_file = os.path.join(output_dir, 'optimization_statistics.csv')
    stats.to_csv(stats_file)

    # Create visualization
    plt.figure(figsize=(15, 10))

    success_pivot = df_results.pivot_table(
        values='success',
        index='temperature',
        columns=['niter', 'stepsize'],
        aggfunc='mean'
    )

    plt.imshow(success_pivot, aspect='auto', cmap='RdYlGn')
    plt.colorbar(label='Success Rate')

    # Add labels
    plt.title('Optimization Success Rate by Parameters')
    plt.xlabel('(niter, stepsize)')
    plt.ylabel('Temperature')

    # Add text annotations
    for i in range(success_pivot.shape[0]):
        for j in range(success_pivot.shape[1]):
            plt.text(j, i, f"{success_pivot.iloc[i, j]:.2f}",
                     ha='center', va='center')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'optimization_heatmap.png'))
    plt.close()

    return df_results, stats


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    results, stats = test_optimization_params(args.data_dir, args.output_dir)
    print("\nOptimization test complete. Best combinations:")
    print(stats.sort_values(('success', 'mean'), ascending=False).head())