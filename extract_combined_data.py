from tkinter import filedialog
import sys
import os
import shutil

BLOeM_id = '/Users/tomsayada/spectral_analysis_project/BLOeM_DR4.0_Combined/id,B dwarf.txt'
MainFolderPath = filedialog.askdirectory(title='Select main folder containing fields')

if not MainFolderPath:
    print("No folder selected, exiting.")
    sys.exit()

PathToOutput = filedialog.askdirectory(title='Select Output Folder')
if not PathToOutput:
    PathToOutput = r''
    print("No output folder selected, using default path.")

with open(BLOeM_id, 'r') as f:
    BLOeM_ids = [line.strip() for line in f]

for bloem_id in BLOeM_ids:
    sys_folder = os.path.join(PathToOutput, bloem_id)
    os.makedirs(sys_folder, exist_ok=True)
    for field in os.listdir(MainFolderPath):
        field_path = os.path.join(MainFolderPath, field, "FITS")
        if os.path.isdir(field_path):
            for file in os.listdir(field_path):
                if bloem_id in file and "Combined" in file:
                    source_file = os.path.join(field_path, file)
                    destination_file = os.path.join(sys_folder, file)
                    shutil.copy2(source_file, destination_file)


print('Done!')
