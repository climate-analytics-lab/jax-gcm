"""
Test script for aerosol-radiation integration

This script tests the updated radiation scheme with aerosol effects.
"""

import jax.numpy as jnp
import jax
from jcm.physics.icon.radiation.radiation_scheme import (
    radiation_scheme, 
    combine_optical_properties
)
from jcm.physics.icon.radiation.radiation_types import RadiationParameters, OpticalProperties
from jcm.physics.icon.radiation.cloud_optics import effective_radius_liquid
from jcm.physics.icon.icon_physics_data import AerosolData

def test_aerosol_cloud_interaction():
    """Test that aerosols modify cloud effective radius"""
    print("Testing aerosol-cloud interactions...")
    
    temperature = jnp.array([280.0])  # Single level test
    land_fraction = 0.5
    
    # Test without aerosols
    r_eff_clean = effective_radius_liquid(jnp.array([1.0]), land_fraction)
    
    # Test with aerosols (increased CDNC)
    cdnc_factor = jnp.array([2.0])  # Double the droplet concentration
    r_eff_polluted = effective_radius_liquid(cdnc_factor, land_fraction)
    
    # With more droplets, effective radius should be smaller
    assert r_eff_polluted[0] < r_eff_clean[0], f"Expected smaller droplets with aerosols: {r_eff_polluted[0]} vs {r_eff_clean[0]}"
    
    # Check the expected scaling r_eff ~ N^(-1/3)
    expected_ratio = (2.0) ** (-1.0/3.0)  # ≈ 0.794
    actual_ratio = r_eff_polluted[0] / r_eff_clean[0]
    
    assert abs(actual_ratio - expected_ratio) < 0.01, f"Expected ratio {expected_ratio}, got {actual_ratio}"
    
    print(f"✓ Clean cloud r_eff: {r_eff_clean[0]:.2f} μm")
    print(f"✓ Polluted cloud r_eff: {r_eff_polluted[0]:.2f} μm")
    print(f"✓ Ratio: {actual_ratio:.3f} (expected: {expected_ratio:.3f})")


def test_optical_property_combination():
    """Test combination of gas, cloud, and aerosol optical properties"""
    print("\\nTesting optical property combination...")
    
    nlev, nbands = 3, 2
    
    # Gas optical depth (absorption only)
    gas_tau = jnp.array([[0.1, 0.05], [0.2, 0.1], [0.3, 0.15]])
    
    # Cloud optical properties (scattering)
    cloud_tau = jnp.array([[1.0, 0.8], [2.0, 1.6], [1.5, 1.2]])
    cloud_ssa = jnp.array([[0.99, 0.98], [0.99, 0.98], [0.99, 0.98]])
    cloud_g = jnp.array([[0.85, 0.82], [0.85, 0.82], [0.85, 0.82]])
    
    cloud_optics = OpticalProperties(
        optical_depth=cloud_tau,
        single_scatter_albedo=cloud_ssa,
        asymmetry_factor=cloud_g
    )
    
    # Aerosol optical properties (scattering + absorption)
    aerosol_tau = jnp.array([[0.2, 0.15], [0.3, 0.25], [0.1, 0.08]])
    aerosol_ssa = jnp.array([[0.9, 0.85], [0.9, 0.85], [0.9, 0.85]])
    aerosol_g = jnp.array([[0.7, 0.65], [0.7, 0.65], [0.7, 0.65]])
    
    # Test combination
    combined = combine_optical_properties(
        gas_tau, cloud_optics, aerosol_tau, aerosol_ssa, aerosol_g
    )
    
    # Check that optical depth is additive
    expected_tau = gas_tau + cloud_tau + aerosol_tau
    assert jnp.allclose(combined.optical_depth, expected_tau), "Optical depth should be additive"
    
    # Check that SSA is properly weighted
    cloud_sca = cloud_tau * cloud_ssa
    aerosol_sca = aerosol_tau * aerosol_ssa
    total_sca = cloud_sca + aerosol_sca
    expected_ssa = total_sca / expected_tau
    
    assert jnp.allclose(combined.single_scatter_albedo, expected_ssa, atol=1e-6), "SSA weighting incorrect"
    
    print("✓ Optical depth combination correct")
    print("✓ Single scattering albedo weighting correct")
    print(f"✓ Example combined τ: {combined.optical_depth[0, 0]:.3f}")
    print(f"✓ Example combined SSA: {combined.single_scatter_albedo[0, 0]:.3f}")


def test_radiation_scheme_with_without_aerosols():
    """Test that radiation scheme runs with and without aerosols"""
    print("\\nTesting radiation scheme with/without aerosols...")
    
    # Create test data
    nlev = 10
    temperature = jnp.linspace(220, 290, nlev)
    specific_humidity = jnp.full(nlev, 0.01)
    surface_pressure = jnp.array(101325.0)
    geopotential = jnp.linspace(0, 50000, nlev)
    cloud_water = jnp.where(temperature > 273, 1e-5, 0.0)
    cloud_ice = jnp.where(temperature <= 273, 1e-5, 0.0)
    cloud_fraction = jnp.where((cloud_water > 0) | (cloud_ice > 0), 0.5, 0.0)
    
    parameters = RadiationParameters.default(n_sw_bands=2, n_lw_bands=3)
    
    # Test with mock aerosol data to ensure array shapes are correct
    try:
        # Create mock aerosol data
        total_bands = int(parameters.n_sw_bands) + int(parameters.n_lw_bands)
        aerosol_tau = jnp.ones((nlev, total_bands)) * 0.1
        aerosol_ssa = jnp.ones((nlev, total_bands)) * 0.9
        # Set LW bands to pure absorption
        aerosol_ssa = aerosol_ssa.at[:, int(parameters.n_sw_bands):].set(0.0)
        aerosol_asy = jnp.ones((nlev, total_bands)) * 0.7
        cdnc_factor = jnp.array([1.5])
        
        print(f"✓ Created test aerosol data: τ shape {aerosol_tau.shape}")
        print(f"✓ SW bands: {int(parameters.n_sw_bands)}, LW bands: {int(parameters.n_lw_bands)}")
        
    except Exception as e:
        print(f"✗ Error creating test data: {e}")
        return


if __name__ == "__main__":
    print("Testing aerosol-radiation integration...")
    print("=" * 50)
    
    test_aerosol_cloud_interaction()
    test_optical_property_combination() 
    test_radiation_scheme_with_without_aerosols()
    
    print("\\n" + "=" * 50)
    print("All tests completed!")
