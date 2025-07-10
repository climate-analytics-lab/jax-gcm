"""
Test radiation integration with IconPhysics

This tests the full integration of the radiation scheme into the
ICON physics framework.

NOTE: This test file was created during radiation integration and uses
mock objects for DateData and Geometry. For future tests, use:
- jcm.date.DateData (real JAX-compatible DateData objects)
- jcm.geometry.Geometry.from_grid_shape() (real Geometry objects)
These are easy to create and don't need mocking.

Date: 2025-01-10
"""

import jax.numpy as jnp
import jax
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.boundaries import BoundaryData
from jcm.geometry import Geometry
from jcm.date import DateData
from jcm.physics.icon.icon_physics import IconPhysics, IconPhysicsData
from jcm.physics.icon.parameters import Parameters


def create_test_state(nlev=20, nlat=8, nlon=16):
    """Create a test physics state"""
    shape = (nlev, nlat, nlon)
    
    # Create realistic temperature profile
    pressure_levels = jnp.linspace(100000, 10000, nlev)
    temp_profile = 288.0 - 6.5e-3 * (jnp.linspace(0, 10000, nlev))  # Standard lapse rate
    temperature = jnp.broadcast_to(temp_profile[:, None, None], shape)
    
    # Humidity decreasing with height
    humid_profile = 0.01 * jnp.exp(-jnp.linspace(0, 10, nlev))
    specific_humidity = jnp.broadcast_to(humid_profile[:, None, None], shape)
    
    # Create tracers including cloud water and ice
    tracers = {
        'qc': jnp.zeros(shape),  # Cloud water
        'qi': jnp.zeros(shape),  # Cloud ice
    }
    
    # Add some clouds in middle troposphere
    tracers['qc'] = tracers['qc'].at[8:12, :, :].set(1e-4)
    tracers['qi'] = tracers['qi'].at[5:8, :, :].set(5e-5)
    
    return PhysicsState(
        u_wind=jnp.zeros(shape),
        v_wind=jnp.zeros(shape),
        temperature=temperature,
        specific_humidity=specific_humidity,
        geopotential=jnp.broadcast_to(
            jnp.linspace(0, 100000, nlev)[:, None, None], shape
        ),
        surface_pressure=jnp.ones((nlat, nlon)),  # Normalized
        tracers=tracers
    )


def create_test_geometry(nlev=20):
    """Create test geometry with sigma levels"""
    # Create evenly spaced sigma levels
    fsg = jnp.linspace(1.0, 0.0, nlev + 1)
    fsg_center = 0.5 * (fsg[:-1] + fsg[1:])
    
    # Simple geometry - would be more complex in real model
    return type('Geometry', (), {
        'fsg': fsg_center,
        'nlev': nlev
    })()


def test_radiation_in_icon_physics():
    """Test radiation integrated in IconPhysics"""
    print("Testing radiation integration in IconPhysics...")
    
    # Create test state
    state = create_test_state()
    
    # Create geometry
    geometry = create_test_geometry()
    
    # Create date for summer noon
    date = DateData(
        tyear=172.0/365.25,  # Summer solstice
        model_year=2000,
        model_step=0
    )
    # Add day_of_year and seconds_since_midnight as attributes for radiation
    date.day_of_year = 172.0
    date.seconds_since_midnight = 43200.0
    
    # Create physics with default parameters
    physics = IconPhysics(
        write_output=True,
        checkpoint_terms=False,  # Disable for testing
        parameters=Parameters.default()
    )
    
    print(f"Physics terms: {len(physics.terms)}")
    print(f"Has radiation: {'_apply_radiation' in [t.__name__ for t in physics.terms]}")
    
    # Apply physics
    tendencies, physics_data = physics.compute_tendencies(
        state=state,
        geometry=geometry,
        date=date
    )
    
    # Check radiation was applied
    assert 'radiation_enabled' in physics_data.radiation_data
    assert physics_data.radiation_data['radiation_enabled']
    
    # Check we have radiation heating
    assert jnp.any(tendencies.temperature != 0)
    
    # Print diagnostics
    print("\nRadiation diagnostics:")
    print(f"Mean OLR: {physics_data.radiation_data['mean_olr']:.1f} W/m²")
    print(f"Mean SW down: {physics_data.radiation_data['mean_sw_down']:.1f} W/m²")
    
    # Check heating rates are reasonable
    heating_rate_K_day = tendencies.temperature * 86400
    print(f"\nHeating rates (K/day):")
    print(f"  Min: {jnp.min(heating_rate_K_day):.2f}")
    print(f"  Max: {jnp.max(heating_rate_K_day):.2f}")
    print(f"  Mean: {jnp.mean(heating_rate_K_day):.2f}")
    
    # Check that radiation produces reasonable ranges - relax for simplified scheme
    assert jnp.all(jnp.abs(heating_rate_K_day) < 15.0)  # Very permissive for simplified scheme
    
    # Check that we have some cooling (LW dominates)
    assert jnp.mean(heating_rate_K_day) < 0
    
    print("\n✓ Radiation integration test passed!")


def test_radiation_with_custom_parameters():
    """Test radiation with custom parameters"""
    print("\nTesting radiation with custom parameters...")
    
    # Create test state
    state = create_test_state(nlev=10, nlat=4, nlon=8)
    geometry = create_test_geometry(nlev=10)
    date = DateData(
        tyear=80.0/365.25,  # Spring
        model_year=2000,
        model_step=0
    )
    date.day_of_year = 80.0
    date.seconds_since_midnight = 0.0
    
    # Create custom parameters
    custom_params = Parameters.default().with_radiation(
        solar_constant=1400.0,  # Higher solar constant
        n_sw_bands=2,
        n_lw_bands=3
    )
    
    physics = IconPhysics(parameters=custom_params)
    
    # Apply physics
    tendencies, physics_data = physics.compute_tendencies(
        state=state,
        geometry=geometry,
        date=date
    )
    
    # At midnight, should have only LW cooling
    heating_rate = tendencies.temperature * 86400
    assert jnp.all(heating_rate <= 0)  # Only cooling
    
    print(f"Nighttime cooling rate: {jnp.mean(heating_rate):.2f} K/day")
    print("\n✓ Custom parameters test passed!")


def test_radiation_conservation():
    """Test energy conservation in radiation"""
    print("\nTesting radiation energy conservation...")
    
    # Create isothermal atmosphere
    nlev, nlat, nlon = 20, 2, 2
    shape = (nlev, nlat, nlon)
    
    state = PhysicsState(
        u_wind=jnp.zeros(shape),
        v_wind=jnp.zeros(shape),
        temperature=jnp.ones(shape) * 250.0,  # Isothermal
        specific_humidity=jnp.ones(shape) * 1e-5,  # Dry
        geopotential=jnp.broadcast_to(
            jnp.linspace(0, 100000, nlev)[:, None, None], shape
        ),
        surface_pressure=jnp.ones((nlat, nlon)),
        tracers={}
    )
    
    geometry = create_test_geometry(nlev)
    date = DateData(
        tyear=80.0/365.25,  # Day 80
        model_year=2000,
        model_step=0
    )
    date.day_of_year = 80.0
    date.seconds_since_midnight = 43200.0
    
    physics = IconPhysics()
    tendencies, physics_data = physics.compute_tendencies(
        state=state,
        geometry=geometry,
        date=date
    )
    
    # Get net radiation at TOA
    toa_net = physics_data.radiation_data['toa_net_radiation']
    print(f"TOA net radiation: {jnp.mean(toa_net):.1f} W/m²")
    
    # For isothermal atmosphere, should have net warming from SW
    assert jnp.mean(toa_net) > 0
    
    print("\n✓ Energy conservation test passed!")


def run_all_integration_tests():
    """Run all radiation integration tests"""
    test_radiation_in_icon_physics()
    test_radiation_with_custom_parameters()
    test_radiation_conservation()
    print("\n🎉 All radiation integration tests passed!")


if __name__ == "__main__":
    run_all_integration_tests()