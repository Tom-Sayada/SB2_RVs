from astropy.io import fits
import numpy as np

# Base path for your FITS files and output
input_base_path = '/Users/tomsayada/spectral_analysis_project/src/SB2_data/6-055/'
output_base_path = '/Users/tomsayada/spectral_analysis_project/src/SB2_data/6-055/obs/'

# Epochs to process (01 through 09)
epochs = [f"{i:02d}" for i in range(1, 10)]

def process_epoch_files():
    for epoch in epochs:
        # Construct the input and output filenames
        input_filename = f"{input_base_path}BLOeM_6-055_{epoch}_Combined.fits"
        output_filename = f"{output_base_path}obs_{epoch}.txt"

        print(f"\n--- Inspecting file: {input_filename} ---")

        try:
            with fits.open(input_filename) as hdul:
                # Print info about the FITS file (list of HDUs, etc.)
                print("FITS file structure:")
                hdul.info()

                # Try to find a table extension containing desired columns
                found_table = False
                for idx, hdu in enumerate(hdul):
                    # Print a short overview of each HDU header
                    # so you can see what columns might be available
                    print(f"\nHDU #{idx}: {hdu.__class__.__name__}")
                    if hasattr(hdu, 'header'):
                        print("Header keywords:")
                        for key in hdu.header.keys():
                            print(f"  {key} = {hdu.header[key]}")

                    # If it’s a BinTableHDU or TableHDU, examine columns
                    if isinstance(hdu, (fits.BinTableHDU, fits.TableHDU)):
                        columns = hdu.columns.names
                        print("Available columns:", columns)

                        # Check if your desired columns are in this table
                        if 'WAVELENGTH' in columns and 'SCI_NORM' in columns:
                            print(f"Found desired columns in HDU #{idx}. Extracting data...")
                            table_data = hdu.data
                            wavelength = table_data['WAVELENGTH']
                            sci_norm = table_data['SCI_NORM']

                            # Combine columns into a 2D array
                            data_to_save = np.column_stack((wavelength, sci_norm))

                            # Save to output file without headers
                            np.savetxt(output_filename, data_to_save, fmt='%.6f')
                            print(f"Created observation file: {output_filename}")
                            found_table = True
                            break
                if not found_table:
                    print(f"No table with WAVELENGTH and SCI_NORM columns found in {input_filename}.")

        except FileNotFoundError:
            print(f"File not found: {input_filename}")
        except Exception as e:
            print(f"An error occurred while processing {input_filename}: {str(e)}")


# Run the processing
if __name__ == "__main__":
    process_epoch_files()
