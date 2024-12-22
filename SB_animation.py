import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Dictionary of lines with their rest wavelengths and windows
lines_info = {
    'He4471': {'wavelength': 4471.5, 'window': 20.0},
    'He4026': {'wavelength': 4026.0, 'window': 20.0},
    'He4388': {'wavelength': 4388.0, 'window': 20.0},
}


def load_observation_files(input_folder):
    """
    Load all observation files. We will extract line segments later.
    This function just loads the full wavelength and flux arrays for each epoch.
    """
    files = sorted(
        [f for f in os.listdir(input_folder) if f.startswith('obs_') and not f.endswith('.gif')],
        key=lambda x: int(x.split('_')[1])  # Assumes epoch number is the second field
    )
    observations = []

    for file in files:
        file_path = os.path.join(input_folder, file)

        # Try to load the file with flexibility for different formats
        try:
            # Attempt to read it as a two-column plain text file
            data = np.loadtxt(file_path, skiprows=1)  # Skip the header
            wavelengths, flux = data[:, 0], data[:, 1]
        except:
            # Fall back for files with metadata
            with open(file_path, 'r') as f:
                lines = f.readlines()

            data = []
            for line in lines:
                try:
                    split_line = line.split()
                    if len(split_line) >= 2:
                        w, fl = float(split_line[0]), float(split_line[1])
                        data.append((w, fl))
                except:
                    continue

            data = np.array(data)
            wavelengths, flux = data[:, 0], data[:, 1]

        observations.append((wavelengths, flux))

    return observations


def find_line_center(wavelengths, flux, rest_wavelength, search_window=5.0):
    """
    Find the observed line center by searching for the wavelength corresponding to
    the minimum flux (assuming absorption line) within a small window around the rest wavelength.

    If your lines are emission lines, you might want to find the maximum flux instead.
    """
    mask = (wavelengths >= rest_wavelength - search_window) & (wavelengths <= rest_wavelength + search_window)
    if not np.any(mask):
        # No data in this range, return the rest wavelength as fallback
        return rest_wavelength

    sub_wl = wavelengths[mask]
    sub_flux = flux[mask]

    # Find the index of minimum flux (for absorption line)
    min_idx = np.argmin(sub_flux)
    line_center = sub_wl[min_idx]
    return line_center


def get_line_data_for_observations(observations, lines_info):
    """
    For each line, extract the wavelength and flux arrays from each epoch,
    and shift them so that the line center matches the rest wavelength.
    Returns a dictionary keyed by line name with a list of (wavelength, flux) arrays for each epoch.
    """
    line_data = {line_name: [] for line_name in lines_info}

    for (wavelengths, flux) in observations:
        # For each epoch, find line segments for each line
        for line_name, line_params in lines_info.items():
            rest_wl = line_params['wavelength']
            window = line_params['window']
            # Extract the line region
            mask = (wavelengths >= rest_wl - window) & (wavelengths <= rest_wl + window)
            if not np.any(mask):
                # If no data in range, skip
                line_data[line_name].append((None, None))
                continue

            sub_wl = wavelengths[mask]
            sub_flux = flux[mask]

            # Find the observed line center
            observed_center = find_line_center(sub_wl, sub_flux, rest_wl, search_window=5.0)

            # Doppler shift: we want to shift the wavelength array so that the observed_center aligns with rest_wl
            # The shift needed:
            shift = rest_wl - observed_center
            shifted_wl = sub_wl + shift

            line_data[line_name].append((shifted_wl, sub_flux))

    return line_data


def animate_spectrum(input_folder, lines_info, output_gif=None):
    """
    Animate the spectra for the specified lines, doppler shifted to line center.
    Shows all three lines on separate subplots.
    """
    # Load all observations
    observations = load_observation_files(input_folder)
    if not observations:
        print("No observations found in the specified folder.")
        return

    # Extract and shift line data
    line_data = get_line_data_for_observations(observations, lines_info)

    # Check if we have data for each line
    for line_name in lines_info:
        if all(d[0] is None for d in line_data[line_name]):
            print(f"No data found for line {line_name}")
            return

    # Create figure with one subplot per line
    fig, axes = plt.subplots(nrows=1, ncols=len(lines_info), figsize=(15, 5))
    if len(lines_info) == 1:
        axes = [axes]  # Ensure axes is a list if only one line

    line_artists = {}
    titles = {}
    for ax, (line_name, line_params) in zip(axes, lines_info.items()):
        (wls, flx) = next((w, f) for (w, f) in line_data[line_name] if w is not None)
        ax.set_xlim(line_params['wavelength'] - line_params['window'],
                    line_params['wavelength'] + line_params['window'])
        # You might want to dynamically set y-limits based on data
        ax.set_ylim(np.min(flx) * 0.9, np.max(flx) * 1.1 if np.max(flx) * 1.1 > 0 else 1.1)

        ax.set_xlabel("Wavelength (Å)")
        ax.set_ylabel("Flux")
        ax.set_title(line_name)

        # Initialize the line artist
        line_artist, = ax.plot([], [], lw=2)
        line_artists[line_name] = line_artist
        titles[line_name] = ax.set_title(line_name, fontsize=10)

    def init():
        for line_name in line_artists:
            line_artists[line_name].set_data([], [])
        return list(line_artists.values())

    def update(frame):
        # Update each line
        for line_name in line_artists:
            w, f = line_data[line_name][frame]
            if w is not None and f is not None:
                line_artists[line_name].set_data(w, f)
        return list(line_artists.values())

    ani = animation.FuncAnimation(fig, update, frames=len(observations), init_func=init,
                                  blit=True, interval=500)  # Adjust interval for speed

    # Save the animation as a GIF if specified
    if output_gif:
        ani.save(output_gif, writer="pillow", fps=8)
        print(f"Animation saved to {output_gif}")
    else:
        plt.show()


# Input folder
input_folder = '/Users/tomsayada/PycharmProjects/BLOeM/source/obs/'  # Update with your folder path

# Run the animation
animate_spectrum(input_folder, lines_info, output_gif=None)
