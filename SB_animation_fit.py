import os
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg

def animate_fits(input_folder, output_gif=None, fps=8, rest_frame_center=4050, x_range=100):
    """
    Create an animated GIF from a series of epoch fit images, with consistent x-axis limits.

    Args:
        input_folder (str): Path to the folder containing epoch_*_fit.png files.
        output_gif (str): Path to save the animated GIF. If None, the animation is displayed instead.
        fps (int): Frames per second for the animation.
        rest_frame_center (float): Wavelength of the rest-frame line center (e.g., in Å).
        x_range (float): The range of wavelengths to display around the rest-frame center.
    """
    # Find all epoch_*_fit.png files and sort them by epoch number
    image_files = sorted(
        glob.glob(os.path.join(input_folder, 'epoch_*_fit.png')),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[1])
    )

    if not image_files:
        print("No epoch_X_fit.png files found in the specified folder.")
        return

    # Load the first image to initialize
    first_img = mpimg.imread(image_files[0])

    # Set consistent x-axis limits
    x_min = rest_frame_center - x_range / 2
    x_max = rest_frame_center + x_range / 2

    fig, ax = plt.subplots()
    im = ax.imshow(first_img, aspect='auto', extent=(x_min, x_max, 0, 1))  # Set consistent extent
    ax.set_xlim(x_min, x_max)  # Fix x-axis limits
    ax.axis('off')  # Hide axes

    def update(frame_index):
        # Load the next image
        img = mpimg.imread(image_files[frame_index])
        im.set_array(img)
        return [im]

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(image_files), interval=1000 // fps, blit=True
    )

    # Save the animation as a GIF if specified
    if output_gif:
        ani.save(output_gif, writer="pillow", fps=fps)
        print(f"Animation saved to {output_gif}")
    else:
        plt.show()

# Usage:
input_folder = '/Users/tomsayada/spectral_analysis_project/data/K_1_50.0_K_2_100.0_SNR_50.0_Q_0.5/simulation_1/standard_sym_weighted_nobaseline_fit_results/epoch_plots'
output_gif = '/Users/tomsayada/spectral_analysis_project/data/K_1_50.0_K_2_100.0_SNR_50.0_Q_0.5/simulation_1/standard_sym_weighted_nobaseline_fit_results/epoch_plots/fitted_animation.gif'

animate_fits(input_folder, output_gif=output_gif, fps=3, rest_frame_center=4050, x_range=100)
