import os
import re
import pandas as pd

# Define the directory
directory = '/Users/tomsayada/spectral_analysis_project/data/obs'


# Initialize an empty list to store the data
data = []

# Define the regex pattern to extract epoch, rv1, and rv2 from the filename
pattern = r'obs_(\d+)_V1_([+-]?\d+\.\d+)_V2_([+-]?\d+\.\d+)'

# Loop through each file in the directory
for filename in os.listdir(directory):
    print(f"Processing file: {filename}")  # Debug: Print each filename
    match = re.search(pattern, filename)
    if match:
        epoch = int(match.group(1))       # Extract epoch
        rv1 = float(match.group(2))       # Extract RV1
        rv2 = float(match.group(3))       # Extract RV2
        print(f"Match found - Epoch: {epoch}, RV1: {rv1}, RV2: {rv2}")  # Debug: Print extracted values
        data.append({'Epoch': epoch, 'RV1': rv1, 'RV2': rv2})
    else:
        print(f"No match found for file: {filename}")  # Debug: Inform if no match found

# Check if data was populated
if not data:
    print("No matching files were found. Please check the file names and pattern.")
else:
    # Create a DataFrame from the collected data
    df = pd.DataFrame(data)
    # Check if DataFrame is empty before sorting
    if df.empty:
        print("DataFrame is empty, no data to display.")
    else:
        # Sort by epoch for a more organized output
        df = df.sort_values(by='Epoch').reset_index(drop=True)
        # Print the table
        print(df)
    # Define the output file path
    output_file = '/Users/tomsayada/spectral_analysis_project/output/star_rvs_per_epoch.csv'

    # Save the DataFrame to a CSV file
    df.to_csv(output_file, index=False)
    print(f"Data saved to {output_file}")

    # # Load the data
    df = pd.read_csv('/Users/tomsayada/spectral_analysis_project/output/star_rvs_per_epoch.csv')
    print(df)
