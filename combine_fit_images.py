import os
from PIL import Image, ImageDraw, ImageFont

# Define the base directory
base_dir = "/Users/tomsayada/spectral_analysis_project/data/K_1_50.0_K_2_100.0_SNR_50.0_Q_0.5"
simulation_folders = [f"simulation_{i}" for i in range(1, 11)]

gaussian_folder = "standard_sym_gaussian_weighted_nobaseline_fit_results"
voigt_folder = "standard_sym_voigt_weighted_nobaseline_fit_results"

# Paths to images
gaussian_images = []
voigt_images = []

for sim_folder in simulation_folders:
    gaussian_path = os.path.join(base_dir, sim_folder, gaussian_folder, "real_calc_comparison.png")
    voigt_path = os.path.join(base_dir, sim_folder, voigt_folder, "real_calc_comparison.png")

    if os.path.exists(gaussian_path):
        gaussian_images.append(gaussian_path)
    if os.path.exists(voigt_path):
        voigt_images.append(voigt_path)

# Load images
loaded_gaussian_images = [Image.open(img_path) for img_path in gaussian_images]
loaded_voigt_images = [Image.open(img_path) for img_path in voigt_images]

# Verify both lists have the same number of images
if len(loaded_gaussian_images) != len(loaded_voigt_images):
    raise ValueError("The number of Gaussian and Voigt images do not match.")

# Define dimensions for the final combined image
image_width = max(img.width for img in loaded_gaussian_images + loaded_voigt_images)
image_height = max(img.height for img in loaded_gaussian_images + loaded_voigt_images)

columns = 2
rows = len(loaded_gaussian_images)
margin = 10
final_width = columns * image_width + margin * (columns + 1)
final_height = rows * image_height + margin * (rows + 1)

# Create a blank image for the combined result
combined_image = Image.new("RGB", (final_width, final_height), "white")

# Add titles
font_size = 20
try:
    font = ImageFont.truetype("arial.ttf", font_size)
except IOError:
    font = ImageFont.load_default()

draw = ImageDraw.Draw(combined_image)
draw.text((margin, margin // 2), "Gaussian Fits", fill="black", font=font)
draw.text((image_width + 2 * margin, margin // 2), "Voigt Fits", fill="black", font=font)

# Paste images into the final combined image
for i, (gaussian_img, voigt_img) in enumerate(zip(loaded_gaussian_images, loaded_voigt_images)):
    y_offset = margin + i * (image_height + margin)
    combined_image.paste(gaussian_img, (margin, y_offset))
    combined_image.paste(voigt_img, (image_width + 2 * margin, y_offset))

# Save the final image
output_path = os.path.join(base_dir, "combined_fit_results.png")
combined_image.save(output_path)

print(f"Combined image saved at: {output_path}")
