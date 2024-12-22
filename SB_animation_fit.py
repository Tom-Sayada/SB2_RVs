import os
import glob
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.image as mpimg

def animate_fits(input_folder, output_gif=None, fps=8):
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

    fig, ax = plt.subplots()
    im = ax.imshow(first_img, aspect='auto')
    ax.axis('off')  # Hide axes if desired

    def update(frame_index):
        # Load the next image
        img = mpimg.imread(image_files[frame_index])
        im.set_array(img)
        return [im]

    # Create the animation
    ani = animation.FuncAnimation(
        fig, update, frames=len(image_files), interval=500, blit=True
    )

    # Save the animation as a GIF if specified
    if output_gif:
        ani.save(output_gif, writer="pillow", fps=fps)
        print(f"Animation saved to {output_gif}")
    else:
        plt.show()

# Usage:
input_folder = '/Users/tomsayada/spectral_analysis_project/output/standard_fit_results'
animate_fits(input_folder, output_gif='fitted_animation.gif', fps=3)
