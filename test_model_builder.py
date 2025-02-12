#!/usr/bin/env python3
import numpy as np
import pytest
from lmfit import Parameters

from src.model_builder import (
    ModelError, ParameterError, ValidationError,
    validate_model_inputs, compute_model_line,
    compute_full_model, residuals, get_star_components
)


def create_test_data():
    """Create simple test data"""
    # Create wavelength array around He4471
    wavelength = np.linspace(4461, 4481, 100)

    # Create simple Gaussian profiles
    def gaussian(x, amp, center, sigma):
        return amp * np.exp(-(x - center) ** 2 / (2 * sigma ** 2))

    # Two components
    flux = (1 + gaussian(wavelength, -0.3, 4470, 0.5) +
            gaussian(wavelength, -0.2, 4473, 0.5))

    # Add some noise
    np.random.seed(42)
    noise = np.random.normal(0, 0.02, len(wavelength))
    flux += noise

    uncertainty = np.full_like(wavelength, 0.02)

    return wavelength, flux, uncertainty


def test_validate_model_inputs():
    """Test input validation"""
    # Create valid inputs
    wavelengths_dict = {'line_44715': np.linspace(4461, 4481, 100)}
    epochs_dict = {'line_44715': np.ones(100)}
    central_wavelengths = {'line_44715': 4471.5}

    params = Parameters()
    params.add('a1_line_44715', value=-0.3)
    params.add('sigma1_line_44715', value=0.6)
    params.add('gamma1_line_44715', value=0.6)
    params.add('a2_line_44715', value=-0.2)
    params.add('sigma2_line_44715', value=0.6)
    params.add('gamma2_line_44715', value=0.6)

    # Should not raise
    validate_model_inputs(params, wavelengths_dict, epochs_dict, central_wavelengths)

    # Test missing key in epochs_dict
    with pytest.raises(ValidationError):
        validate_model_inputs(params,
                              {'line_44715': np.array([])},
                              {},
                              central_wavelengths
                              )

    # Test length mismatch
    with pytest.raises(ValidationError):
        validate_model_inputs(params,
                              {'line_44715': np.array([1, 2, 3])},
                              {'line_44715': np.array([1, 2])},
                              central_wavelengths
                              )


def test_compute_model_line():
    """Test model line computation"""
    wavelength = np.linspace(4461, 4481, 100)

    params = Parameters()
    params.add('a1_line_44715', value=-0.3)
    params.add('sigma1_line_44715', value=0.6)
    params.add('gamma1_line_44715', value=0.6)
    params.add('a2_line_44715', value=-0.2)
    params.add('sigma2_line_44715', value=0.6)
    params.add('gamma2_line_44715', value=0.6)

    # Should compute without error
    combined, star1, star2 = compute_model_line(
        params, 'line_44715', wavelength,
        rv1=-50, rv2=50, rest_wavelength=4471.5
    )

    # Test shapes
    assert len(combined) == len(wavelength)
    assert len(star1) == len(wavelength)
    assert len(star2) == len(wavelength)

    # Test values are reasonable
    assert np.all(combined >= 0)  # Flux should be positive
    assert np.all(star1 >= 0)
    assert np.all(star2 >= 0)

    # Test error on invalid parameters
    params.add('sigma1_line_44715', value=-1.0)  # Invalid negative width
    with pytest.raises(ParameterError):
        compute_model_line(
            params, 'line_44715', wavelength,
            rv1=-50, rv2=50, rest_wavelength=4471.5
        )


def test_compute_full_model():
    """Test full model computation"""
    wavelength, flux, uncertainty = create_test_data()

    wavelengths_dict = {'line_44715': wavelength}
    epochs_dict = {'line_44715': np.ones_like(wavelength)}
    central_wavelengths = {'line_44715': 4471.5}

    params = Parameters()
    params.add('a1_line_44715', value=-0.3)
    params.add('sigma1_line_44715', value=0.6)
    params.add('gamma1_line_44715', value=0.6)
    params.add('a2_line_44715', value=-0.2)
    params.add('sigma2_line_44715', value=0.6)
    params.add('gamma2_line_44715', value=0.6)
    params.add('rv1_epoch1', value=-50)
    params.add('rv2_epoch1', value=50)

    # Should compute without error
    model_fluxes = compute_full_model(
        params, wavelengths_dict, epochs_dict, central_wavelengths
    )

    assert 'line_44715' in model_fluxes
    assert len(model_fluxes['line_44715']) == len(wavelength)


def test_residuals():
    """Test residuals computation with weighting"""
    wavelength, flux, uncertainty = create_test_data()

    wavelengths_dict = {'line_44715': wavelength}
    fluxes_dict = {'line_44715': flux}
    epochs_dict = {'line_44715': np.ones_like(wavelength)}
    uncertainties_dict = {'line_44715': uncertainty}
    central_wavelengths = {'line_44715': 4471.5}

    params = Parameters()
    params.add('a1_line_44715', value=-0.3)
    params.add('sigma1_line_44715', value=0.6)
    params.add('gamma1_line_44715', value=0.6)
    params.add('a2_line_44715', value=-0.2)
    params.add('sigma2_line_44715', value=0.6)
    params.add('gamma2_line_44715', value=0.6)
    params.add('rv1_epoch1', value=-50)
    params.add('rv2_epoch1', value=50)

    # Test unweighted residuals
    r1 = residuals(
        params, wavelengths_dict, fluxes_dict,
        epochs_dict, uncertainties_dict, central_wavelengths,
        weighted=False
    )

    # Test weighted residuals
    r2 = residuals(
        params, wavelengths_dict, fluxes_dict,
        epochs_dict, uncertainties_dict, central_wavelengths,
        weighted=True
    )

    # Weighted residuals should be different
    assert not np.allclose(r1, r2)

    # Test clipping works
    assert np.all(r1 >= -5) and np.all(r1 <= 5)
    assert np.all(r2 >= -5) and np.all(r2 <= 5)


def test_get_star_components():
    """Test star component extraction"""
    wavelength = np.linspace(4461, 4481, 100)

    params = Parameters()
    params.add('a1_line_44715', value=-0.3)
    params.add('sigma1_line_44715', value=0.6)
    params.add('gamma1_line_44715', value=0.6)
    params.add('a2_line_44715', value=-0.2)
    params.add('sigma2_line_44715', value=0.6)
    params.add('gamma2_line_44715', value=0.6)

    star1, star2 = get_star_components(
        params, 'line_44715', wavelength,
        rest_wv=4471.5, rv1=-50, rv2=50
    )

    assert len(star1) == len(wavelength)
    assert len(star2) == len(wavelength)

    # Components should be different
    assert not np.allclose(star1, star2)


if __name__ == "__main__":
    print("Running tests...")

    test_validate_model_inputs()
    print("✓ Input validation tests passed")

    test_compute_model_line()
    print("✓ Model line computation tests passed")

    test_compute_full_model()
    print("✓ Full model computation tests passed")

    test_residuals()
    print("✓ Residuals computation tests passed")

    test_get_star_components()
    print("✓ Star component extraction tests passed")

    print("\nAll tests passed successfully!")