from astropy.io import fits
import pandas as pd

# File paths
fits_file_path = '/Users/tomsayada/spectral_analysis_project/data/obs/BLOeM_6-055_02_Combined.fits'
excel_file_path = '/Users/tomsayada/spectral_analysis_project/data/obs/metadata.xlsx'

# Open the FITS file
with fits.open(fits_file_path) as hdul:
    # Extract metadata from all headers
    metadata = []
    for idx, hdu in enumerate(hdul):
        header = hdu.header
        for key, value in header.items():
            # Append to metadata list
            metadata.append({'HDU': idx, 'Key': key, 'Value': value})

# Convert metadata to a pandas DataFrame
df = pd.DataFrame(metadata)

# Save to Excel
df.to_excel(excel_file_path, index=False)

print(f"Metadata saved to {excel_file_path}")
