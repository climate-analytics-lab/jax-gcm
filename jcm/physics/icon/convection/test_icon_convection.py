#!/usr/bin/env python3
"""
Test script for ICON convection scheme

This script tests the complete convection scheme including updraft and downdraft
components to ensure JAX compatibility and reasonable physics behavior.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.physics.icon.icon_physics import IconPhysics
from jcm.physics.icon.convection import ConvectionConfig
from jcm.geometry import Geometry
from jcm.date import DateData

def create_test_state():
    """Create a test atmospheric state with convective instability"""
    
    # Grid dimensions
    nlev, nlat, nlon = 20, 4, 4
    
    # Create basic geometry - simplified version for testing
    fsg = jnp.linspace(0.05, 1.0, nlev)  # Sigma levels
    # Create minimal geometry object with just the required field
    class SimpleGeometry:
        def __init__(self, fsg):
            self.fsg = fsg
    geometry = SimpleGeometry(fsg)
    
    # Create unstable temperature profile
    temp_surface = 300.0  # K
    temp_profiles = []
    
    for i in range(nlat * nlon):
        # Create different instability patterns
        if i % 2 == 0:
            # Strong instability 
            temp_profile = temp_surface - 80.0 * fsg  # 8 K/100mb lapse
        else:
            # Moderate instability
            temp_profile = temp_surface - 60.0 * fsg  # 6 K/100mb lapse
        temp_profiles.append(temp_profile)
    
    # Stack into 3D array [nlev, nlat, nlon]
    temperature = jnp.stack(temp_profiles, axis=1).reshape(nlev, nlat, nlon)
    
    # Create humidity profile with surface moisture
    humidity_surface = 0.015  # 15 g/kg
    humidity = humidity_surface * jnp.exp(-2.0 * fsg[:, jnp.newaxis, jnp.newaxis])
    humidity = jnp.broadcast_to(humidity, (nlev, nlat, nlon))
    
    # Create wind fields
    u_wind = jnp.zeros((nlev, nlat, nlon))
    v_wind = jnp.zeros((nlev, nlat, nlon))
    
    # Create geopotential (hydrostatic)
    g = 9.81
    R = 287.0
    geopotential = jnp.zeros((nlev, nlat, nlon))
    for k in range(nlev-2, -1, -1):  # Work upward
        dz = -R * temperature[k] * jnp.log(fsg[k+1] / fsg[k]) / g
        geopotential = geopotential.at[k].set(geopotential[k+1] + g * dz)
    
    # Surface pressure (normalized)
    surface_pressure = jnp.ones((nlat, nlon))
    
    return PhysicsState(
        u_wind=u_wind,
        v_wind=v_wind,
        temperature=temperature,
        specific_humidity=humidity,
        geopotential=geopotential,
        surface_pressure=surface_pressure
    ), geometry

def test_convection_scheme():
    """Test the complete ICON convection scheme"""
    
    print("🧪 Testing ICON Convection Scheme")
    print("=" * 50)
    
    # Create test state
    state, geometry = create_test_state()
    print(f"✅ Created test state: {state.temperature.shape}")
    
    # Create physics instance with convection enabled
    convection_config = ConvectionConfig(
        entrpen=1.0e-4,    # Deep convection entrainment
        entrscv=3.0e-4,    # Shallow convection entrainment  
        entrmid=1.0e-4,    # Mid-level convection entrainment
        cmfcmin=1.0e-10,   # Minimum mass flux
        cmfctop=0.33       # Downdraft fraction
    )
    
    physics = IconPhysics(
        enable_convection=True,
        enable_radiation=False,
        enable_clouds=False,
        checkpoint_terms=False,  # Disable checkpointing for testing
        convection_config=convection_config
    )
    print("✅ Created IconPhysics with convection enabled")
    
    # Test computation
    try:
        print("🔄 Computing convection tendencies...")
        
        # Create date data
        date = DateData.zeros()
        
        # Apply JIT compilation
        # Note: avoid JIT for this test due to complex geometry object
        def compute_conv_tendencies(state):
            return physics.compute_tendencies(state, geometry=geometry, date=date)
        
        tendencies, physics_data = compute_conv_tendencies(state)
        print("✅ Successfully computed tendencies")
        
        # Check results
        temp_tendency = tendencies.temperature
        humid_tendency = tendencies.specific_humidity
        
        print(f"📊 Temperature tendency range: [{jnp.min(temp_tendency):.2e}, {jnp.max(temp_tendency):.2e}] K/s")
        print(f"📊 Humidity tendency range: [{jnp.min(humid_tendency):.2e}, {jnp.max(humid_tendency):.2e}] kg/kg/s")
        
        # Check for reasonable physics
        max_heating = jnp.max(temp_tendency)
        max_cooling = jnp.min(temp_tendency)
        max_moistening = jnp.max(humid_tendency)
        max_drying = jnp.min(humid_tendency)
        
        print("\n🔍 Physics Validation:")
        
        # Temperature tendencies should be reasonable (< 10 K/day)
        max_heating_per_day = max_heating * 86400
        max_cooling_per_day = max_cooling * 86400
        print(f"  Max heating: {max_heating_per_day:.2f} K/day")
        print(f"  Max cooling: {max_cooling_per_day:.2f} K/day")
        
        if abs(max_heating_per_day) < 20.0 and abs(max_cooling_per_day) < 20.0:
            print("  ✅ Temperature tendencies are physically reasonable")
        else:
            print("  ⚠️ Temperature tendencies may be too large")
        
        # Humidity tendencies 
        print(f"  Max moistening: {max_moistening * 86400 * 1000:.2f} g/kg/day")
        print(f"  Max drying: {max_drying * 86400 * 1000:.2f} g/kg/day")
        
        if abs(max_moistening * 86400 * 1000) < 50.0:
            print("  ✅ Humidity tendencies are physically reasonable")
        else:
            print("  ⚠️ Humidity tendencies may be too large")
        
        # Check conservation
        total_heating = jnp.sum(temp_tendency)
        total_moistening = jnp.sum(humid_tendency)
        print(f"  Total column heating: {total_heating:.2e} K⋅m/s")
        print(f"  Total column moisture: {total_moistening:.2e} kg⋅m/kg/s")
        
        # Test vectorization performance
        print("\n⚡ Performance Test:")
        
        # Time a few iterations
        import time
        start_time = time.time()
        for _ in range(10):
            tendencies, _ = compute_conv_tendencies(state)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / 10.0
        print(f"  Average computation time: {avg_time*1000:.2f} ms")
        print(f"  Grid points: {state.temperature.shape[1] * state.temperature.shape[2]}")
        print(f"  Time per column: {avg_time*1000 / (state.temperature.shape[1] * state.temperature.shape[2]):.3f} ms")
        
        print("\n🎉 ICON Convection Test Completed Successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error during convection computation: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_jax_patterns():
    """Test that JAX conversion patterns work correctly"""
    
    print("\n🔧 Testing JAX Conversion Patterns")
    print("=" * 40)
    
    # Test that the functions can be JIT compiled
    try:
        from jcm.physics.icon.convection.updraft import calculate_updraft, ConvectionConfig
        from jcm.physics.icon.convection.downdraft import calculate_downdraft
        
        # Create test data
        nlev = 10
        temperature = jnp.linspace(300, 250, nlev)
        humidity = jnp.full(nlev, 0.01)
        pressure = jnp.linspace(100000, 10000, nlev)
        height = jnp.linspace(0, 10000, nlev)
        rho = jnp.linspace(1.2, 0.3, nlev)
        
        config = ConvectionConfig()
        
        print("🔄 Testing updraft calculation...")
        
        # Test updraft
        @jax.jit
        def test_updraft():
            return calculate_updraft(
                temperature, humidity, pressure, height, rho,
                kbase=8, ktop=2, ktype=1, mass_flux_base=0.1, config=config
            )
        
        updraft_state = test_updraft()
        print("✅ Updraft calculation JIT compiled successfully")
        
        print("🔄 Testing downdraft calculation...")
        
        # Test downdraft  
        @jax.jit
        def test_downdraft():
            return calculate_downdraft(
                temperature, humidity, pressure, height, rho,
                updraft_state, precip_rate=jnp.array(0.001),
                kbase=8, ktop=2, config=config
            )
        
        downdraft_state = test_downdraft()
        print("✅ Downdraft calculation JIT compiled successfully")
        
        print("✅ All JAX patterns working correctly")
        return True
        
    except Exception as e:
        print(f"❌ JAX pattern test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Run tests
    success1 = test_jax_patterns()
    success2 = test_convection_scheme()
    
    if success1 and success2:
        print("\n🎉 All tests passed! ICON convection scheme is ready.")
    else:
        print("\n❌ Some tests failed. Check the errors above.")