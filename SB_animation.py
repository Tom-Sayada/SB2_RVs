#!/usr/bin/env python3

"""
animate_spectrum.py

Creates an animation of line segments around specified spectral lines
(across multiple observation epochs) by doppler-shifting each line
to align with its rest wavelength. Works for both simulation-like data
and real fits/txt observation files using your existing 'utils.py'.

Usage:
  python animate_spectrum.py
  -> Will prompt you to pick a folder containing observation files
     (like 'BLOeM_4-059_01_Combined.fits' or 'obs_123.txt', etc.).
  -> Then it loads them in ascending epoch order using utils.find_observation_files,
     extracts line segments for lines in 'lines_info', finds line centers,
     shifts them to rest_wavelength, and animates them.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tkinter as tk
from tkinter import filedialog

# Import from your SB2 project
try:
    from src.utils import find_observation_files, load_data_for_epoch
except ImportError as e:
    print("Could not import find_observation_files or load_data_for_epoch from src.utils.")
    print("Make sure your project structure / PYTHONPATH is correct.")
    raise e

###############################################################################
# Define which lines you want to animate. Adjust or add more as you like.
###############################################################################
lines_info = {
    'He4471': {'rest_wavelength': 4471.5, 'window': 20.0},
    'He4026': {'rest_wavelength': 4026.0, 'window': 20.0},
    'He4388': {'rest_wavelength': 4388.0, 'window': 20.0},
    'Hg4340': {'rest_wavelength': 4340.472, 'window': 20.0},
    'Hd4101': {'rest_wavelength': 4101.734, 'window': 20.0},
}


###############################################################################
# A helper function to find the line center in a small sub-region
# We assume absorption => min flux
###############################################################################
def find_line_center(wave, flux, rest_wl, search_window=5.0):
    """
    Look for min flux in [rest_wl - search_window, rest_wl + search_window].
    Return the wavelength of that min flux. If no data, fallback to rest_wl.
    """
    mask = (wave >= rest_wl - search_window) & (wave <= rest_wl + search_window)
    if not np.any(mask):
        return rest_wl  # fallback

    subw = wave[mask]
    subf = flux[mask]

    idx_min = np.argmin(subf)
    return subw[idx_min]


###############################################################################
# This function extracts the line windows for each epoch & line,
# finds the line center, and shifts so that line center => rest_wavelength.
###############################################################################
def extract_shifted_line_data(observations, lines_info):
    """
    observations: list of (wavelength array, flux array) per epoch
    lines_info: dict line_name => {'rest_wavelength': X, 'window': Y}

    Returns:
      line_data = { line_name: [ (shifted_wave, flux) for each epoch ] }
    """
    line_data = {}
    for line_name, line_params in lines_info.items():
        line_data[line_name] = []

    for (wave_all, flux_all) in observations:
        # For each epoch, for each line
        for line_name, line_params in lines_info.items():
            rest_wl = line_params['rest_wavelength']
            window = line_params['window']

            # subset wave, flux in [rest_wl - window, rest_wl + window]
            mask = (wave_all >= rest_wl - window) & (wave_all <= rest_wl + window)
            if not np.any(mask):
                # no data => store None
                line_data[line_name].append((None, None))
                continue

            subw = wave_all[mask]
            subf = flux_all[mask]

            # find observed center
            obs_center = find_line_center(subw, subf, rest_wl, search_window=5.0)

            # shift wave array so that obs_center => rest_wl
            shift = rest_wl - obs_center
            shifted_wave = subw + shift
            # store
            line_data[line_name].append((shifted_wave, subf))

    return line_data


###############################################################################
# The main animation function
###############################################################################
def animate_spectrum(folder_path, lines_info, output_gif=None):
    # 1) find observation files
    try:
        obs_files = find_observation_files(folder_path)
    except Exception as e:
        print(f"Error searching for observation files in {folder_path}: {e}")
        return

    if not obs_files:
        print("No observation files found.")
        return

    print(f"Found {len(obs_files)} observation files in {folder_path}")

    # 2) load data for each epoch in ascending order
    observations = []
    for (ep, fpath) in sorted(obs_files, key=lambda x: x[0]):
        df = load_data_for_epoch(fpath)
        if df.empty:
            print(f"Warning: epoch {ep} => no data.")
            # store None?
            observations.append((np.array([]), np.array([])))
            continue

        # convert to numpy arrays
        wave = df['wavelength'].values
        flux = df['flux'].values
        observations.append((wave, flux))

    if not observations:
        print("No loaded data => exit.")
        return

    # 3) shift lines
    line_data = extract_shifted_line_data(observations, lines_info)

    # 4) Build the animation figure with subplots => one per line
    n_lines = len(lines_info)
    fig, axes = plt.subplots(nrows=1, ncols=n_lines, figsize=(5 * n_lines, 5))
    if n_lines == 1:
        axes = [axes]

    # pre-define line artists
    line_artists = {}
    line_names = list(lines_info.keys())

    # For each subplot/line
    for ax, line_name in zip(axes, line_names):
        # find the first non-None entry for x-limits
        wave_example, flux_example = None, None
        for (w, f) in line_data[line_name]:
            if w is not None and len(w) > 0:
                wave_example = w
                flux_example = f
                break
        if wave_example is None:
            # no data => just skip
            ax.set_title(f"{line_name} (no data)")
            continue

        # set axis limits
        rest_wl = lines_info[line_name]['rest_wavelength']
        window = lines_info[line_name]['window']
        ax.set_xlim(rest_wl - window, rest_wl + window)
        # approximate flux min, max from the example
        fmin = np.min(flux_example)
        fmax = np.max(flux_example)
        ax.set_ylim(fmin * 0.50, fmax * 1.05 if fmax > 0 else 1.1)
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("Flux")
        ax.set_title(f"{line_name}", fontsize=12)

        # create a line artist
        (lref,) = ax.plot([], [], color='b', lw=2)
        line_artists[line_name] = lref

    def init_func():
        # set everything to empty
        for ln in line_names:
            if ln in line_artists:
                line_artists[ln].set_data([], [])
        return list(line_artists.values())

    def update_func(frame):
        # each line => update data from line_data
        for ln in line_names:
            w, f = line_data[ln][frame]
            if w is not None and f is not None and ln in line_artists:
                line_artists[ln].set_data(w, f)
        return list(line_artists.values())

    ani = animation.FuncAnimation(fig, update_func, frames=len(observations),
                                  init_func=init_func, interval=800, blit=True)

    if output_gif:
        ani.save(output_gif, writer='pillow', fps=6)
        print(f"Animation saved => {output_gif}")
    else:
        plt.show()


###############################################################################
def main():
    print("Select a folder containing observation files (FITS or text).")
    root = tk.Tk()
    root.withdraw()
    folder_path = filedialog.askdirectory(title="Select folder with obs_... or BLOeM_... files")
    root.destroy()

    if not folder_path:
        print("No folder selected => exit.")
        sys.exit(1)

    # Example usage: animate the lines, optionally ask user for a GIF name
    out_gif = None
    # out_gif = os.path.join(folder_path, "spectra_animation.gif")  # if you want a default

    animate_spectrum(folder_path, lines_info, output_gif=out_gif)


if __name__ == "__main__":
    main()
