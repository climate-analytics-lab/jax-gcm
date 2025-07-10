"""
Tests for ICON physics implementation

Date: 2025-01-09
"""

import jax.numpy as jnp
import pytest
from jcm.physics.icon.icon_physics import IconPhysics, IconPhysicsData, set_physics_flags
from jcm.physics.icon.constants import physical_constants
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.date import DateData

# Simple mock geometry class for testing
class MockGeometry:
    def __init__(self, nlev):
        self.fsg = jnp.linspace(0.05, 1.0, nlev)

def test_physical_constants():
    """Test that physical constants are properly defined"""
    # Test basic constants
    assert physical_constants.grav == 9.81
    assert physical_constants.rearth == 6.371e6
    assert physical_constants.cp == 1004.0
    assert physical_constants.rgas == 287.0
    
    # Test derived constants
    assert physical_constants.eps == 0.622
    assert physical_constants.tmelt == 273.15

def test_icon_physics_data():
    """Test IconPhysicsData container"""
    date = DateData(tyear=0.0, model_year=2000, model_step=0)
    
    # Test creation
    physics_data = IconPhysicsData.zeros(date=date)
    assert physics_data.date == date
    assert physics_data.radiation_data == {}
    assert physics_data.convection_data == {}
    assert physics_data.cloud_data == {}
    assert physics_data.surface_data == {}
    
    # Test copy method with updated data
    new_convection_data = {'test_value': 42}
    new_data = physics_data.copy(convection_data=new_convection_data)
    assert new_data.date == date
    assert new_data.convection_data == new_convection_data
    assert new_data.radiation_data == {}  # Unchanged

def test_icon_physics_initialization():
    """Test IconPhysics initialization"""
    # Test default initialization
    physics = IconPhysics()
    assert physics.write_output == True
    assert physics.checkpoint_terms == True
    
    # Test custom initialization
    physics = IconPhysics(
        write_output=False,
        checkpoint_terms=False
    )
    assert physics.write_output == False
    assert physics.checkpoint_terms == False

def test_set_physics_flags():
    """Test physics flag setting"""
    # Create test data
    date = DateData(tyear=0.0, model_year=2000, model_step=0)
    physics_data = IconPhysicsData.zeros(date=date)
    
    # Create dummy physics state with tracers
    # compute_tendencies expects shape (nlev, nlat, nlon)
    dummy_shape = (20, 32, 64)  # Example shape (lev, lat, lon)
    tracers = {
        'qc': jnp.zeros(dummy_shape),  # Cloud water
        'qi': jnp.zeros(dummy_shape),  # Cloud ice
        'chem': jnp.zeros(dummy_shape)  # Chemical tracer
    }
    state = PhysicsState(
        temperature=jnp.zeros(dummy_shape),
        specific_humidity=jnp.zeros(dummy_shape),
        u_wind=jnp.zeros(dummy_shape),
        v_wind=jnp.zeros(dummy_shape),
        geopotential=jnp.zeros(dummy_shape),
        surface_pressure=jnp.zeros(dummy_shape[1:]),  # (lat, lon)
        tracers=tracers
    )
    
    # Test flag setting
    tendencies, updated_data = set_physics_flags(state, physics_data)
    
    # Check that tendencies are initialized to zero
    assert tendencies.temperature.shape == dummy_shape
    assert jnp.all(tendencies.temperature == 0)
    assert jnp.all(tendencies.specific_humidity == 0)

def test_physics_state_with_tracers():
    """Test PhysicsState with tracers field"""
    # Create test shape
    dummy_shape = (20, 32, 64)  # (nlev, nlat, nlon)
    
    # Create tracers
    tracers = {
        'qc': jnp.zeros(dummy_shape),
        'qi': jnp.zeros(dummy_shape),
        'chem': jnp.ones(dummy_shape) * 0.5
    }
    
    # Create physics state
    state = PhysicsState(
        temperature=jnp.ones(dummy_shape) * 280.0,
        specific_humidity=jnp.ones(dummy_shape) * 0.01,
        u_wind=jnp.zeros(dummy_shape),
        v_wind=jnp.zeros(dummy_shape),
        geopotential=jnp.zeros(dummy_shape),
        surface_pressure=jnp.ones(dummy_shape[1:]),
        tracers=tracers
    )
    
    # Check that state has tracers
    assert hasattr(state, 'tracers')
    assert 'qc' in state.tracers
    assert 'qi' in state.tracers
    assert 'chem' in state.tracers
    assert jnp.allclose(state.tracers['chem'], 0.5)

def test_physics_tendency_with_tracers():
    """Test PhysicsTendency with tracer tendencies"""
    # Create test shape
    dummy_shape = (20, 32, 64)  # (nlev, nlat, nlon)
    
    # Create tracer tendencies
    tracer_tends = {
        'qc': jnp.ones(dummy_shape) * 1e-5,
        'qi': jnp.ones(dummy_shape) * 1e-6,
        'chem': jnp.zeros(dummy_shape)
    }
    
    # Create physics tendency
    tendency = PhysicsTendency(
        temperature=jnp.zeros(dummy_shape),
        specific_humidity=jnp.zeros(dummy_shape),
        u_wind=jnp.zeros(dummy_shape),
        v_wind=jnp.zeros(dummy_shape),
        tracers=tracer_tends
    )
    
    # Check that tendency has tracers
    assert hasattr(tendency, 'tracers')
    assert 'qc' in tendency.tracers
    assert 'qi' in tendency.tracers
    assert 'chem' in tendency.tracers
    assert jnp.allclose(tendency.tracers['qc'], 1e-5)

if __name__ == "__main__":
    test_physical_constants()
    test_icon_physics_data()
    test_icon_physics_initialization()
    test_set_physics_flags()
    test_physics_state_with_tracers()
    test_physics_tendency_with_tracers()
    print("All tests passed!")