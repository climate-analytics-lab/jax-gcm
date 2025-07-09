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
    physics_data = IconPhysicsData(date=date)
    assert physics_data.date == date
    assert physics_data.radiation_data == {}
    
    # Test copy
    new_data = physics_data.copy(test_field=123)
    assert new_data.date == date
    assert new_data['test_field'] == 123

def test_icon_physics_initialization():
    """Test IconPhysics initialization"""
    # Test default initialization
    physics = IconPhysics()
    assert physics.write_output == True
    assert physics.enable_radiation == True
    assert physics.enable_convection == True
    
    # Test custom initialization
    physics = IconPhysics(
        enable_radiation=False,
        enable_chemistry=True
    )
    assert physics.enable_radiation == False
    assert physics.enable_chemistry == True

def test_set_physics_flags():
    """Test physics flag setting"""
    # Create test data
    date = DateData(tyear=0.0, model_year=2000, model_step=0)
    physics_data = IconPhysicsData(date=date)
    
    # Create dummy physics state
    dummy_shape = (32, 64, 20)  # Example shape (lat, lon, lev)
    state = PhysicsState(
        temperature=jnp.zeros(dummy_shape),
        specific_humidity=jnp.zeros(dummy_shape),
        u_wind=jnp.zeros(dummy_shape),
        v_wind=jnp.zeros(dummy_shape),
        geopotential=jnp.zeros(dummy_shape),
        surface_pressure=jnp.zeros(dummy_shape[:2])
    )
    
    # Test flag setting
    tendencies, updated_data = set_physics_flags(state, physics_data)
    
    # Check that tendencies are initialized to zero
    assert tendencies.temperature.shape == dummy_shape
    assert jnp.all(tendencies.temperature == 0)
    assert jnp.all(tendencies.specific_humidity == 0)

def test_icon_physics_call():
    """Test IconPhysics call method"""
    # Create test objects
    date = DateData(tyear=0.0, model_year=2000, model_step=0)
    physics_data = IconPhysicsData(date=date)
    physics = IconPhysics()
    
    # Create dummy physics state
    dummy_shape = (32, 64, 20)
    state = PhysicsState(
        temperature=jnp.zeros(dummy_shape),
        specific_humidity=jnp.zeros(dummy_shape),
        u_wind=jnp.zeros(dummy_shape),
        v_wind=jnp.zeros(dummy_shape),
        geopotential=jnp.zeros(dummy_shape),
        surface_pressure=jnp.zeros(dummy_shape[:2])
    )
    
    # Test physics call
    tendencies, updated_data = physics(state, physics_data)
    
    # Check outputs
    assert tendencies.temperature.shape == dummy_shape
    assert isinstance(updated_data, IconPhysicsData)
    assert updated_data.date == date

if __name__ == "__main__":
    test_physical_constants()
    test_icon_physics_data()
    test_icon_physics_initialization()
    test_set_physics_flags()
    test_icon_physics_call()
    print("All tests passed!")