"""
Simple test to verify the unified Parameters object works correctly

Date: 2025-01-10
"""

import jax.numpy as jnp
from jcm.physics.icon.parameters import Parameters
from jcm.physics.icon.icon_physics import IconPhysics


def test_parameters_initialization():
    """Test that Parameters can be initialized with defaults"""
    params = Parameters()
    
    # Check that sub-parameters exist
    assert params.convection is not None
    assert params.clouds is not None
    assert params.microphysics is not None
    
    # Check some default values
    assert params.convection.entrpen == 1.0e-4
    assert params.clouds.crt == 0.9
    assert params.microphysics.ccraut == 5.0e-4
    
    print("✓ Default parameters initialized correctly")


def test_parameters_with_methods():
    """Test the with_* methods for updating parameters"""
    params = Parameters()
    
    # Test with_convection
    params2 = params.with_convection(entrpen=4.0e-4)
    assert params2.convection.entrpen == 4.0e-4
    assert params.convection.entrpen == 1.0e-4  # Original unchanged
    
    # Test with_clouds
    params3 = params.with_clouds(crt=0.85)
    assert params3.clouds.crt == 0.85
    assert params.clouds.crt == 0.9  # Original unchanged
    
    # Test with_microphysics
    params4 = params.with_microphysics(ccraut=0.5e-3)
    assert params4.microphysics.ccraut == 0.5e-3
    assert params.microphysics.ccraut == 5.0e-4  # Original unchanged
    
    print("✓ Parameter update methods work correctly")


def test_icon_physics_with_parameters():
    """Test that IconPhysics can be initialized with Parameters"""
    # Default parameters
    physics1 = IconPhysics()
    assert physics1.parameters is not None
    assert physics1.parameters.convection.entrpen == 1.0e-4
    
    # Custom parameters
    custom_params = Parameters().with_convection(entrpen=5.0e-4)
    physics2 = IconPhysics(parameters=custom_params)
    assert physics2.parameters.convection.entrpen == 5.0e-4
    
    print("✓ IconPhysics accepts Parameters object")


def test_physics_terms_use_parameters():
    """Test that physics terms can access parameters"""
    from jcm.physics_interface import PhysicsState
    from jcm.date import DateData
    
    # Create simple test state
    nlev, nlat, nlon = 8, 4, 4
    state = PhysicsState(
        u_wind=jnp.zeros((nlev, nlat, nlon)),
        v_wind=jnp.zeros((nlev, nlat, nlon)),
        temperature=jnp.ones((nlev, nlat, nlon)) * 280.0,
        specific_humidity=jnp.ones((nlev, nlat, nlon)) * 0.005,
        geopotential=jnp.ones((nlev, nlat, nlon)) * 1000.0,
        surface_pressure=jnp.ones((nlat, nlon)),
        tracers={
            'qc': jnp.zeros((nlev, nlat, nlon)),
            'qi': jnp.zeros((nlev, nlat, nlon))
        }
    )
    
    # Create physics with custom parameters
    custom_params = Parameters().with_clouds(crt=0.8)
    physics = IconPhysics(parameters=custom_params)
    
    # The physics should be able to compute tendencies
    # (This is a basic smoke test)
    import jcm.geometry as geo
    geometry = geo.Geometry.from_grid_shape((nlat, nlon), nlev)
    
    tendencies, physics_data = physics.compute_tendencies(
        state, 
        boundaries=None,
        geometry=geometry,
        date=DateData.zeros()
    )
    
    # Check that tendencies have the right shape
    assert tendencies.temperature.shape == (nlev, nlat, nlon)
    assert 'qc' in tendencies.tracers
    assert 'qi' in tendencies.tracers
    
    print("✓ Physics terms can use parameters correctly")


if __name__ == "__main__":
    test_parameters_initialization()
    test_parameters_with_methods()
    test_icon_physics_with_parameters()
    test_physics_terms_use_parameters()
    print("\nAll parameter tests passed! ✅")