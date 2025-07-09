#!/usr/bin/env python3
"""
Test script for full Tiedtke-Nordeng integration in IconPhysics

Tests the complete integration of the Tiedtke-Nordeng convection scheme
with updraft, downdraft, and tracer transport.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.physics.icon.icon_physics import IconPhysics
from jcm.physics.icon.convection import ConvectionConfig
from jcm.date import DateData

def create_test_state():
    """Create a test atmospheric state"""
    
    # Grid dimensions
    nlev, nlat, nlon = 20, 2, 2
    
    # Create geometry
    fsg = jnp.linspace(0.05, 1.0, nlev)
    class SimpleGeometry:
        def __init__(self, fsg):
            self.fsg = fsg
    geometry = SimpleGeometry(fsg)
    
    # Create unstable temperature profile
    # fsg goes from 0.05 (top) to 1.0 (surface), so we need to make surface warm
    temp_surface = 300.0
    
    # Create height-like coordinate from sigma levels (higher fsg = lower altitude)
    height_coord = (1.0 - fsg) * 10000  # 0-10km altitude
    
    # Strong lapse rate for testing
    temperature = temp_surface - 0.012 * height_coord  # 12 K/km lapse rate
    temperature = temperature[:, jnp.newaxis, jnp.newaxis]
    temperature = jnp.broadcast_to(temperature, (nlev, nlat, nlon))
    
    print(f"  Temperature profile: surface={temperature[-1,0,0]:.1f}K, top={temperature[0,0,0]:.1f}K")
    print(f"  Lapse rate: {(temperature[-1,0,0] - temperature[0,0,0])/10:.1f} K/km")
    
    # Create humidity profile (higher at surface, decreasing with height)
    humidity_surface = 0.015  # 15 g/kg at surface
    # Use (1-fsg) so that surface (fsg=1) gives maximum humidity
    humidity = humidity_surface * jnp.exp(-2.0 * (1.0 - fsg[:, jnp.newaxis, jnp.newaxis]))
    humidity = jnp.broadcast_to(humidity, (nlev, nlat, nlon))
    
    print(f"  Humidity profile: surface={humidity[-1,0,0]*1000:.1f}g/kg, top={humidity[0,0,0]*1000:.1f}g/kg")
    
    # Create wind fields
    u_wind = jnp.zeros((nlev, nlat, nlon))
    v_wind = jnp.zeros((nlev, nlat, nlon))
    
    # Create geopotential
    g = 9.81
    R = 287.0
    geopotential = jnp.zeros((nlev, nlat, nlon))
    for k in range(nlev-2, -1, -1):
        dz = -R * temperature[k] * jnp.log(fsg[k+1] / fsg[k]) / g
        geopotential = geopotential.at[k].set(geopotential[k+1] + g * dz)
    
    # Surface pressure
    surface_pressure = jnp.ones((nlat, nlon))
    
    state = PhysicsState(
        u_wind=u_wind,
        v_wind=v_wind,
        temperature=temperature,
        specific_humidity=humidity,
        geopotential=geopotential,
        surface_pressure=surface_pressure
    )
    
    return state, geometry

def test_tiedtke_integration():
    """Test the full Tiedtke-Nordeng integration"""
    
    print("🧪 Testing Full Tiedtke-Nordeng Integration")
    print("=" * 50)
    
    # Create test state
    state, geometry = create_test_state()
    print(f"✅ Created test state: {state.temperature.shape}")
    
    # Create physics with Tiedtke-Nordeng
    convection_config = ConvectionConfig(
        entrpen=1.0e-4,
        entrscv=3.0e-4,
        entrmid=1.0e-4,
        cmfcmin=1.0e-10,
        cmfctop=0.33,
        tau=7200.0,
        cprcon=0.0014
    )
    
    physics = IconPhysics(
        enable_convection=True,
        enable_radiation=False,
        enable_clouds=False,
        checkpoint_terms=False,
        convection_config=convection_config
    )
    print("✅ Created IconPhysics with Tiedtke-Nordeng")
    
    # Test computation
    try:
        print("🔄 Computing Tiedtke-Nordeng tendencies...")
        
        date = DateData.zeros()
        
        tendencies, physics_data = physics.compute_tendencies(
            state, geometry=geometry, date=date
        )
        print("✅ Successfully computed Tiedtke-Nordeng tendencies")
        
        # Debug one column profile
        print("\n🔍 Debug atmospheric column profile:")
        col_temp = state.temperature[:, 0, 0]
        col_humid = state.specific_humidity[:, 0, 0] 
        print(f"  Temperature: {col_temp}")
        print(f"  Humidity: {col_humid*1000}")
        print(f"  Sigma levels: {geometry.fsg}")
        
        # Test convection on this specific column
        print("\n🔍 Test convection on single column:")
        from jcm.physics.icon.convection.tiedtke_nordeng import tiedtke_nordeng_convection
        from jcm.physics.speedy.physical_constants import p0
        
        # Create single column data
        nlev = state.temperature.shape[0]
        pressure_col = geometry.fsg * state.surface_pressure[0, 0] * p0
        height_col = jnp.linspace(0, 10000, nlev)  # Simple height profile
        
        col_tend, col_state = tiedtke_nordeng_convection(
            col_temp, col_humid, pressure_col, height_col,
            state.u_wind[:, 0, 0], state.v_wind[:, 0, 0], 1800.0
        )
        
        print(f"  Single column conv type: {col_state.ktype}")
        print(f"  Single column temp tend: [{jnp.min(col_tend.dtedt):.2e}, {jnp.max(col_tend.dtedt):.2e}]")
        
        # Debug CAPE calculation
        from jcm.physics.icon.convection.tiedtke_nordeng import calculate_cape_cin
        cape, cin = calculate_cape_cin(col_temp, col_humid, pressure_col, height_col, col_state.kbase, convection_config)
        print(f"  Single column CAPE: {cape:.1f} J/kg, CIN: {cin:.1f} J/kg, cloud base: {col_state.kbase}")
        print(f"  Pressure range: {pressure_col[0]:.0f} to {pressure_col[-1]:.0f} Pa")
        print(f"  Surface detection: argmax(pressure) = {jnp.argmax(pressure_col)}")
        
        # Check convection data
        conv_data = physics_data.convection_data
        print(f"📊 Convection active: {conv_data.get('tiedtke_nordeng_active', False)}")
        
        # Check tendencies
        temp_tend = tendencies.temperature
        humid_tend = tendencies.specific_humidity
        
        print(f"📊 Temperature tendency range: [{jnp.min(temp_tend):.2e}, {jnp.max(temp_tend):.2e}] K/s")
        print(f"📊 Humidity tendency range: [{jnp.min(humid_tend):.2e}, {jnp.max(humid_tend):.2e}] kg/kg/s")
        
        # Check convective products
        if 'convective_cloud_water' in conv_data:
            qc_conv = conv_data['convective_cloud_water']
            qi_conv = conv_data['convective_cloud_ice']
            precip = conv_data['convective_precipitation']
            
            print(f"☁️ Convective cloud water range: [{jnp.min(qc_conv):.2e}, {jnp.max(qc_conv):.2e}] kg/kg")
            print(f"❄️ Convective cloud ice range: [{jnp.min(qi_conv):.2e}, {jnp.max(qi_conv):.2e}] kg/kg")
            print(f"🌧️ Convective precipitation range: [{jnp.min(precip):.2e}, {jnp.max(precip):.2e}] kg/m²/s")
        
        # Physics validation
        max_heating = jnp.max(temp_tend) * 86400  # K/day
        max_cooling = jnp.min(temp_tend) * 86400
        max_moistening = jnp.max(humid_tend) * 86400 * 1000  # g/kg/day
        max_drying = jnp.min(humid_tend) * 86400 * 1000
        
        print("\n🔍 Physics Validation:")
        print(f"  Max heating: {max_heating:.2f} K/day")
        print(f"  Max cooling: {max_cooling:.2f} K/day")
        print(f"  Max moistening: {max_moistening:.2f} g/kg/day")
        print(f"  Max drying: {max_drying:.2f} g/kg/day")
        
        # Check if tendencies are reasonable
        if abs(max_heating) < 50.0 and abs(max_cooling) < 50.0:
            print("  ✅ Temperature tendencies are physically reasonable")
        else:
            print("  ⚠️ Temperature tendencies may be too large")
        
        print("\n🎉 Tiedtke-Nordeng Integration Test Completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during Tiedtke-Nordeng computation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tiedtke_integration()
    
    if success:
        print("\n🎉 Full Tiedtke-Nordeng convection scheme is integrated and working!")
    else:
        print("\n❌ Integration test failed. Check the errors above.")