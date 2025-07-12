"""
Simple unit tests for ICON physics that avoid complex schemes

This provides basic tests for the ICON physics infrastructure without
running the full radiation and aerosol schemes that have JAX compatibility issues.

Date: 2025-01-11
"""

import jax.numpy as jnp
import pytest
from jcm.physics.icon.icon_physics import IconPhysics, _prepare_common_physics_state
from jcm.physics.icon.icon_physics_data import PhysicsData
from jcm.physics.icon.parameters import Parameters
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.date import DateData
from jcm.boundaries import BoundaryData
from jcm.geometry import Geometry
from typing import Tuple


def apply_simple_test_physics(
    state: PhysicsState,
    physics_data: PhysicsData,
    parameters: Parameters,
    boundaries: BoundaryData,
    geometry: Geometry
) -> Tuple[PhysicsTendency, PhysicsData]:
    """Simple test physics that just returns small tendencies"""
    nlev, ncols = state.temperature.shape
    
    # Create simple tendencies
    temperature_tendency = jnp.ones((nlev, ncols)) * 1e-5  # Small warming
    humidity_tendency = jnp.ones((nlev, ncols)) * -1e-7   # Small drying
    u_tendency = jnp.zeros((nlev, ncols))
    v_tendency = jnp.zeros((nlev, ncols))
    
    # Create tracer tendencies
    tracer_tends = {}
    for name, tracer in state.tracers.items():
        tracer_tends[name] = jnp.zeros_like(tracer)
    
    tendencies = PhysicsTendency(
        temperature=temperature_tendency,
        specific_humidity=humidity_tendency,
        u_wind=u_tendency,
        v_wind=v_tendency,
        tracers=tracer_tends
    )
    
    return tendencies, physics_data


def test_prepare_common_physics_state():
    """Test common physics state preparation"""
    # Setup
    nlev, nlat, nlon = 8, 4, 8
    ncols = nlat * nlon
    
    # Create state
    tracers = {
        'qc': jnp.zeros((nlev, ncols)),
        'qi': jnp.zeros((nlev, ncols))
    }
    state = PhysicsState(
        temperature=jnp.ones((nlev, ncols)) * 280.0,
        specific_humidity=jnp.ones((nlev, ncols)) * 0.01,
        u_wind=jnp.zeros((nlev, ncols)),
        v_wind=jnp.zeros((nlev, ncols)),
        geopotential=jnp.zeros((nlev, ncols)),
        surface_pressure=jnp.ones(ncols),
        tracers=tracers
    )
    
    # Create other inputs
    date = DateData.zeros()
    physics_data = PhysicsData.zeros((nlat, nlon), nlev, date=date)
    parameters = Parameters.default()
    boundaries = BoundaryData.zeros((nlat, nlon))
    geometry = Geometry.from_grid_shape((nlat, nlon), nlev)
    
    # Run preparation
    tendencies, updated_physics_data = _prepare_common_physics_state(
        state, physics_data, parameters, boundaries, geometry
    )
    
    # Check outputs
    assert jnp.all(tendencies.temperature == 0)
    assert jnp.all(tendencies.specific_humidity == 0)
    assert hasattr(updated_physics_data, 'diagnostics')
    assert hasattr(updated_physics_data.diagnostics, 'pressure_full')
    assert updated_physics_data.diagnostics.pressure_full.shape == (nlev, ncols)


def test_simple_physics_integration():
    """Test simple physics integration without complex schemes"""
    # Setup
    nlev, nlat, nlon = 8, 4, 8
    ncols = nlat * nlon
    
    # Create test physics with only simple terms
    class SimpleIconPhysics(IconPhysics):
        def __init__(self):
            super().__init__()
            # Override terms with simple test physics
            self.terms = [
                _prepare_common_physics_state,
                apply_simple_test_physics
            ]
    
    physics = SimpleIconPhysics()
    
    # Create state with reshape to 3D
    shape_3d = (nlev, nlat, nlon)
    tracers = {
        'qc': jnp.zeros(shape_3d),
        'qi': jnp.zeros(shape_3d)
    }
    state = PhysicsState(
        temperature=jnp.ones(shape_3d) * 280.0,
        specific_humidity=jnp.ones(shape_3d) * 0.01,
        u_wind=jnp.zeros(shape_3d),
        v_wind=jnp.zeros(shape_3d),
        geopotential=jnp.zeros(shape_3d),
        surface_pressure=jnp.ones((nlat, nlon)),
        tracers=tracers
    )
    
    # Create other inputs
    date = DateData.zeros()
    boundaries = BoundaryData.zeros((nlat, nlon))
    geometry = Geometry.from_grid_shape((nlat, nlon), nlev)
    
    # Run physics
    tendencies, physics_data = physics.compute_tendencies(
        state, boundaries, geometry, date
    )
    
    # Check outputs
    assert tendencies.temperature.shape == shape_3d
    assert tendencies.specific_humidity.shape == shape_3d
    assert jnp.any(tendencies.temperature != 0)  # Should have non-zero from test physics
    assert jnp.any(tendencies.specific_humidity != 0)
    
    # Check tracer tendencies
    assert 'qc' in tendencies.tracers
    assert 'qi' in tendencies.tracers
    assert tendencies.tracers['qc'].shape == shape_3d


def test_physics_vectorization():
    """Test that physics properly vectorizes over columns"""
    # Setup
    nlev, nlat, nlon = 8, 4, 8
    
    physics = IconPhysics()
    physics.terms = [_prepare_common_physics_state, apply_simple_test_physics]
    
    # Create state - make temperature vary by column
    shape_3d = (nlev, nlat, nlon)
    temp_base = 280.0
    temperature = jnp.ones(shape_3d) * temp_base
    # Add variation by latitude
    for i in range(nlat):
        temperature = temperature.at[:, i, :].set(temp_base + i * 2.0)
    
    tracers = {'qc': jnp.zeros(shape_3d)}
    state = PhysicsState(
        temperature=temperature,
        specific_humidity=jnp.ones(shape_3d) * 0.01,
        u_wind=jnp.zeros(shape_3d),
        v_wind=jnp.zeros(shape_3d),
        geopotential=jnp.zeros(shape_3d),
        surface_pressure=jnp.ones((nlat, nlon)),
        tracers=tracers
    )
    
    # Create other inputs
    date = DateData.zeros()
    boundaries = BoundaryData.zeros((nlat, nlon))
    geometry = Geometry.from_grid_shape((nlat, nlon), nlev)
    
    # Run physics
    tendencies, physics_data = physics.compute_tendencies(
        state, boundaries, geometry, date
    )
    
    # Check that all columns got processed
    assert jnp.all(tendencies.temperature > 0)  # All should have heating
    assert tendencies.temperature.shape == shape_3d
    
    # Check physics data diagnostics exist
    assert hasattr(physics_data, 'diagnostics')
    assert hasattr(physics_data.diagnostics, 'pressure_full')


if __name__ == "__main__":
    test_prepare_common_physics_state()
    test_simple_physics_integration() 
    test_physics_vectorization()
    print("All simple physics tests passed!")