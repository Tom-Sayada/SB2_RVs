#!/usr/bin/env python3

import os
import sys
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import curve_fit
import astropy.io.fits as fits
from tkinter import Tk, filedialog
from matplotlib.backend_bases import KeyEvent

matplotlib.use("TkAgg")  # often more reliable for key events on Mac

clight = 2.99792458e5
WHAT_TO_READ = 'FITS'  # or 'TXT'

def gaussian(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def find_line_center(wavelengths, flux, line_wavelength, window):
    mask = ((wavelengths >= (line_wavelength - window/2)) &
            (wavelengths <= (line_wavelength + window/2)))
    x = wavelengths[mask]
    y = flux[mask]
    if len(x) < 5:
        return line_wavelength
    # invert flux => peak
    y_inv = 1.0 - y / np.max(y)
    p0 = [np.min(y_inv), line_wavelength, 1.0]
    try:
        popt, _ = curve_fit(gaussian, x, y_inv, p0=p0)
        return popt[1]
    except:
        return line_wavelength

def load_folder_data(folder_path, spectral_lines):
    """
    Return observations_data = {
      line_name: [ (wv_sub, fl_sub), (wv_sub, fl_sub), ... per file ],
      ...
    }
    """
    if WHAT_TO_READ == 'TXT':
        files = sorted(
            [f for f in os.listdir(folder_path) if f.startswith('obs_')],
            key=lambda x: int(x.split('_')[1].split('.')[0])
        )
    else:  # FITS
        files = sorted(
            [f for f in os.listdir(folder_path) if f.startswith('BLOeM')],
            key=lambda x: int(x.split('_')[2])  # adjust if needed
        )

    observations = { ln: [] for ln in spectral_lines }

    for fname in files:
        fullp = os.path.join(folder_path, fname)
        try:
            if WHAT_TO_READ == 'TXT':
                data = np.loadtxt(fullp, skiprows=1)
                wv_in, fl_in = data[:,0], data[:,1]
            else:
                with fits.open(fullp) as hdul:
                    d = hdul[1].data
                    wv_in = d['WAVELENGTH']
                    fl_in = d['SCI_NORM']
        except:
            continue

        for ln_name, ln_info in spectral_lines.items():
            rw  = ln_info['wavelength']
            win = ln_info['window']
            center = find_line_center(wv_in, fl_in, rw, win)
            delta  = center - rw
            wv_shift = wv_in - delta
            mask = ((wv_shift >= rw - win/2) & (wv_shift <= rw + win/2))
            observations[ln_name].append( (wv_shift[mask], fl_in[mask]) )

    return observations


###############################################################################
# Global variables for single figure approach
###############################################################################
spectral_lines = {
    'He4471': {'wavelength': 4471.5, 'window': 20.0},
    'He4026': {'wavelength': 4026.0, 'window': 20.0},
    'He4388': {'wavelength': 4388.0, 'window': 20.0},
    'H4340':  {'wavelength': 4340.468,'window': 20.0},
    'H4100':  {'wavelength': 4101.734,'window': 20.0},
}

subfolders = []
folder_index = 0

fig = None
axes = []
line_artists = []
title_artists = []
anim = None
observations_data = None
n_frames = 0


def init_animation():
    for ln in line_artists:
        ln.set_data([], [])
    return line_artists + title_artists

def update_animation(frame):
    ln_keys = list(spectral_lines.keys())
    for i, ln_name in enumerate(ln_keys):
        if frame < len(observations_data[ln_name]):
            wv, fl = observations_data[ln_name][frame]
            line_artists[i].set_data(wv, fl)
            title_artists[i].set_text(f"{ln_name} (Frame {frame+1})")
    return line_artists + title_artists


def load_and_make_animation():
    global anim, observations_data, n_frames
    folder_path = subfolders[folder_index]
    print(f"\nLoading folder [{folder_index+1}/{len(subfolders)}]: {folder_path}")

    # Load data
    observations_data = load_folder_data(folder_path, spectral_lines)
    first_line = list(spectral_lines.keys())[0]
    n_frames = len(observations_data[first_line])
    print(f"Number of frames in this folder: {n_frames}")

    # Stop old anim if needed
    if anim is not None:
        anim.event_source.stop()

    # Create new FuncAnimation
    anim_local = animation.FuncAnimation(
        fig, update_animation,
        init_func=init_animation,
        frames=n_frames,
        interval=600,
        blit=False
    )
    anim = anim_local

    # Update figure title
    base = os.path.basename(folder_path)
    fig.suptitle(f"Folder: {base}", fontsize=14)

    # Start the event source for new anim
    # (Sometimes needed to ensure it actually animates on Mac).
    anim.event_source.start()

    # Force a quick draw
    plt.draw()
    plt.pause(0.01)


def on_key(event):
    global folder_index
    print(f"Key pressed: {event.key}")
    if event.key in ('up','uparrow'):
        if folder_index < len(subfolders) - 1:
            folder_index += 1
            load_and_make_animation()
    elif event.key in ('down','downarrow'):
        if folder_index > 0:
            folder_index -= 1
            load_and_make_animation()
    else:
        print("Use up/down arrows to navigate folders.")


def main():
    global fig, axes, line_artists, title_artists, folder_index, subfolders

    # Use interactive mode
    plt.ion()

    root = Tk()
    root.withdraw()
    parent = filedialog.askdirectory(title="Select parent directory with F-NNN subfolders")
    root.destroy()

    if not parent:
        print("No folder => exit.")
        sys.exit()

    # find F-NNN subfolders
    pattern = re.compile(r'^\d-\d{3}$')
    found = []
    for entry in os.listdir(parent):
        fullp = os.path.join(parent, entry)
        if os.path.isdir(fullp) and pattern.match(entry):
            found.append(fullp)
    if not found:
        print("No subfolders => exit.")
        sys.exit()

    def parse_subfolder_name(p):
        base = os.path.basename(p)
        f_str, s_str = base.split('-')
        return (int(f_str), int(s_str))

    found.sort(key=parse_subfolder_name)
    subfolders = found

    print("Subfolders:")
    for i,fld in enumerate(subfolders):
        print(f"{i+1}) {fld}")

    folder_index = 0

    # Build single figure
    n_lines = len(spectral_lines)
    fig, axes = plt.subplots(1, n_lines, figsize=(4*n_lines, 5))
    if n_lines == 1:
        axes = [axes]

    # create line objects
    for ax, (ln_name, ln_info) in zip(axes, spectral_lines.items()):
        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("Flux")
        rw = ln_info['wavelength']
        win= ln_info['window']
        ax.set_xlim(rw - win/2, rw + win/2)
        ax.set_ylim(0.4, 1.1)
        t = ax.set_title(ln_name)
        l, = ax.plot([], [], lw=1.5)
        line_artists.append(l)
        title_artists.append(t)

    fig.canvas.mpl_connect('key_press_event', on_key)

    # load the first folder => create animation
    load_and_make_animation()

    print("Use Up/Down arrows to switch among folders. Press close to end.")
    # Show in non-blocking => event loop must keep running
    plt.show(block=False)

    # This while loop will keep the script alive until the figure is closed
    while plt.fignum_exists(fig.number):
        plt.pause(0.1)  # process GUI events

    print("Figure closed => script done.")


if __name__ == "__main__":
    main()
