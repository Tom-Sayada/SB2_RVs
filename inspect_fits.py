from astropy.io import fits
import numpy as np

# Base path for your FITS files and output
input_base_path = '/Users/tomsayada/spectral_analysis_project/src/'
output_base_path = '/Users/tomsayada/spectral_analysis_project/src/obs/'

# Epochs to process
epochs = [f"{i:02d}" for i in range(1, 10)]  # Epochs 01 through 09


def process_epoch_files():
    for epoch in epochs:
        # Construct the input and output filenames
        input_filename = f"{input_base_path}BLOeM_6-055_{epoch}_Combined.fits"
        output_filename = f"{output_base_path}obs_{epoch}.txt"

        # Open the FITS file
        with fits.open(input_filename) as hdul:
            # Find the binary table
            for hdu in hdul:
                if isinstance(hdu, fits.BinTableHDU):
                    # Extract wavelength and sci_NORM columns
                    table_data = hdu.data
                    if 'WAVELENGTH' in table_data.columns.names and 'SCI_NORM' in table_data.columns.names:
                        wavelength = table_data['WAVELENGTH']
                        sci_norm = table_data['SCI_NORM']

                        # Combine columns into a 2D array
                        data_to_save = np.column_stack((wavelength, sci_norm))

                        # Save to output file without headers
                        np.savetxt(output_filename, data_to_save, fmt='%.6f')
                        print(f"Created observation file: {output_filename}")
                    else:
                        print(f"Columns 'wavelength' and 'sci_NORM' not found in {input_filename}")
                    break
            else:
                print(f"No binary table found in {input_filename}")


# Run the processing
process_epoch_files()
