"""Simple test to verify the unified Parameters object works correctly

Date: 2025-01-10
"""

import jax.numpy as jnp
from jcm.physics.icon.parameters import Parameters
from jcm.physics.icon.icon_terms import icon_physics


def test_parameters_initialization():
    """Test that Parameters can be initialized with defaults"""
    params = Parameters.default()
    
    # Check that sub-parameters exist
    assert params.convection is not None
    assert params.clouds is not None
    assert params.microphysics is not None
    
    # Check some default values.
    # ECHAM-matching convention: crt at surface (0.9), crs aloft (0.7);
    # ccraut = 15.0 (ECHAM default — Beheng-1994 coefficient, not the
    # KK2000 threshold the previous JAX port used).
    assert abs(float(params.convection.entrpen) - 1.0e-4) < 1e-7
    assert abs(float(params.clouds.crt) - 0.9) < 1e-7
    assert abs(float(params.clouds.crs) - 0.7) < 1e-7
    assert abs(float(params.microphysics.ccraut) - 15.0) < 1e-5
    
    print("✓ Default parameters initialized correctly")


def test_parameters_with_methods():
    """Test the with_* methods for updating parameters"""
    params = Parameters.default()
    
    # Test with_convection
    params2 = params.with_convection(entrpen=4.0e-4)
    assert abs(float(params2.convection.entrpen) - 4.0e-4) < 1e-7
    assert abs(float(params.convection.entrpen) - 1.0e-4) < 1e-7  # Original unchanged
    
    # Test with_clouds
    params3 = params.with_clouds(crt=0.85)
    assert abs(float(params3.clouds.crt) - 0.85) < 1e-7
    assert abs(float(params.clouds.crt) - 0.9) < 1e-7  # Original unchanged
    
    # Test with_microphysics
    params4 = params.with_microphysics(ccraut=0.5e-3)
    assert abs(float(params4.microphysics.ccraut) - 0.5e-3) < 1e-7
    assert abs(float(params.microphysics.ccraut) - 15.0) < 1e-5  # Original unchanged (Beheng default)
    
    print("✓ Parameter update methods work correctly")


def test_icon_physics_with_parameters():
    """Test that icon_physics() can be initialized with Parameters"""
    # Default parameters
    physics1 = icon_physics()
    assert physics1.parameters is not None
    assert abs(float(physics1.parameters.convection.entrpen) - 1.0e-4) < 1e-7
    
    # Custom parameters
    custom_params = Parameters.default().with_convection(entrpen=5.0e-4)
    physics2 = icon_physics(parameters=custom_params)
    assert abs(float(physics2.parameters.convection.entrpen) - 5.0e-4) < 1e-7
    
    print("✓ icon_physics() accepts Parameters object")


def test_physics_terms_use_parameters():
    """Test that physics terms can access parameters"""
    from jcm.physics_interface import PhysicsState
    from jcm.date import DateData
    from jcm.forcing import ForcingData
    import jax_datetime as jdt
    from datetime import datetime
    
    # Create simple test state
    nlev, nlat, nlon = 8, 64, 32
    state = PhysicsState(
        u_wind=jnp.zeros((nlev, nlat, nlon)),
        v_wind=jnp.zeros((nlev, nlat, nlon)),
        temperature=jnp.ones((nlev, nlat, nlon)) * 280.0,
        specific_humidity=jnp.ones((nlev, nlat, nlon)) * 0.005,
        geopotential=jnp.ones((nlev, nlat, nlon)) * 1000.0,
        normalized_surface_pressure=jnp.ones((nlat, nlon)),
        tracers={
            'qc': jnp.zeros((nlev, nlat, nlon)),
            'qi': jnp.zeros((nlev, nlat, nlon))
        }
    )
    
    # Create physics with custom parameters
    custom_params = Parameters.default().with_clouds(crt=0.8)
    physics = icon_physics(parameters=custom_params)
    
    # The physics should be able to compute tendencies
    # (This is a basic smoke test)
    import numpy as np
    from jcm.utils import get_coords
    from jcm.terrain import TerrainData
    sigma_boundaries = np.linspace(0, 1, nlev + 1)
    coords = get_coords(sigma_boundaries, nodal_shape=(nlat, nlon))
    terrain = TerrainData.aquaplanet(coords)
    physics.cache_coords(coords)
    # Physics terms now consume forcing that has already been time-sliced by
    # `Model._get_step_fn_factory` → `ForcingData.select(date)`. Build a
    # 2-D forcing here (no time axis) so the smoke test mirrors the per-step
    # contract; for time-varying boundary conditions the Model handles the
    # slicing.
    forcing = ForcingData.zeros((nlat, nlon),
                                    sea_surface_temperature=jnp.ones((nlat, nlon)) * 288.0,
                                    sice_am=jnp.zeros((nlat, nlon)))
    date = DateData.set_date(jdt.Datetime.from_pydatetime(datetime(2020, 6, 21)))
    forcing = forcing.select(date)

    tendencies, physics_data = physics.compute_tendencies(
        state,
        forcing=forcing,
        terrain=terrain,
        date=date,
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