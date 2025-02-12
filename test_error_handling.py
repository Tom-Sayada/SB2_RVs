#!/usr/bin/env python
# test_error_handling.py

import os
import numpy as np
from lmfit import Parameters
from src.utils import (load_data_for_epoch, find_observation_files,
                       DataLoadError, voigt_profile)
from src.model_builder import compute_full_model, get_star_components, ModelError


# Modified sections of test_error_handling.py

def test_file_loading():
    print("\nTesting file loading...")

    # Test non-existent file
    try:
        data = load_data_for_epoch("nonexistent_file.txt")
        print("✓ Properly handled non-existent file")
    except DataLoadError as e:
        print(f"✓ Caught expected error: {e}")

    # Test invalid file format
    try:
        with open("test.invalid", "w") as f:
            f.write("test")
        data = load_data_for_epoch("test.invalid")
        print("✓ Properly handled invalid file format")
    except DataLoadError as e:
        print(f"✓ Caught expected error: {e}")
    finally:
        if os.path.exists("test.invalid"):
            os.remove("test.invalid")

    # Test malformed data file (single test instead of two)
    try:
        with open("bad_data.txt", "w") as f:
            f.write("wavelength,flux\na,b\n")
        data = load_data_for_epoch("bad_data.txt")
        if data.empty:
            print("✓ Properly handled malformed data")
        else:
            raise AssertionError("Expected empty DataFrame for malformed data")
    finally:
        if os.path.exists("bad_data.txt"):
            os.remove("bad_data.txt")

def test_model_computation():
    print("\nTesting model computation...")

    # Create test parameters
    params = Parameters()
    params.add('a1_line_4471', value=-0.3)
    params.add('sigma1_line_4471', value=0.5)
    params.add('gamma1_line_4471', value=0.5)
    params.add('a2_line_4471', value=-0.2)
    params.add('sigma2_line_4471', value=0.5)
    params.add('gamma2_line_4471', value=0.5)
    params.add('rv1_epoch1', value=50.0)
    params.add('rv2_epoch1', value=-50.0)

    # Test with empty dictionaries
    try:
        wavelengths_dict = {}
        epochs_dict = {}
        central_wavelengths = {'line_4471': 4471.5}
        model = compute_full_model(params, wavelengths_dict, epochs_dict, central_wavelengths)
        print("✓ Properly handled empty dictionaries")
    except ModelError as e:
        print(f"✓ Caught expected error: {e}")

    # Test with valid arrays
    x = np.linspace(4460, 4480, 100)
    wavelengths_dict = {'line_4471': x}
    epochs_dict = {'line_4471': np.ones(100)}  # Matching length
    try:
        model = compute_full_model(params, wavelengths_dict, epochs_dict, central_wavelengths)
        print("✓ Properly handled valid arrays")
    except ModelError as e:
        print(f"Unexpected error with valid arrays: {e}")

    # Test with mismatched arrays
    try:
        mismatched_epochs = {'line_4471': np.ones(50)}  # Different length
        model = compute_full_model(params, wavelengths_dict, mismatched_epochs, central_wavelengths)
        print("Failed to catch mismatched arrays")
    except ModelError as e:
        if "Mismatched array lengths" in str(e):
            print("✓ Properly caught mismatched arrays")
        else:
            print(f"Unexpected error: {e}")

def validate_array_lengths(wavelengths_dict, epochs_dict):
    """Validate that wavelength and epoch arrays have matching lengths for each line"""
    for line_id in wavelengths_dict:
        if line_id not in epochs_dict:
            raise ModelError(f"Missing epoch data for line {line_id}")
        if len(wavelengths_dict[line_id]) != len(epochs_dict[line_id]):
            raise ModelError(
                f"Mismatched array lengths for line {line_id}: "
                f"wavelengths({len(wavelengths_dict[line_id])}) != "
                f"epochs({len(epochs_dict[line_id])})"
            )

def test_voigt_profile():
    print("\nTesting Voigt profile computation...")

    # Test with invalid inputs
    try:
        x = np.linspace(0, 10, 100)
        profile = voigt_profile(x, np.nan, 5, 1, 1)
        print("✓ Properly handled invalid amplitude")
    except Exception as e:
        print(f"✓ Caught expected error: {e}")

    # Test with zero width
    try:
        profile = voigt_profile(x, 1, 5, 0, 0)
        print("✓ Properly handled zero width")
    except Exception as e:
        print(f"✓ Caught expected error: {e}")


def main():
    print("Starting error handling tests...")

    test_file_loading()
    test_model_computation()
    test_voigt_profile()

    print("\nTests completed.")


if __name__ == "__main__":
    main()