"""
Test ICON physics integration with shallow cloud scheme

Date: 2025-01-10
"""

import jax.numpy as jnp
import jax
from jcm.physics.icon.icon_physics import IconPhysics
from jcm.physics_interface import PhysicsState
from jcm.boundaries import BoundaryData
from jcm.geometry import Geometry
from jcm.date import DateData
from jcm.physics.icon.clouds import saturation_specific_humidity
import tree_math


@tree_math.struct
class TestGeometry:
    """Test geometry for physics integration"""
    fsg: jnp.ndarray
    lat: jnp.ndarray
    lon: jnp.ndarray
    nodal_shape: tuple


def create_test_geometry(nlev, nlat, nlon):
    """Create a simple test geometry"""
    # Create sigma levels (vertical coordinate)
    fsg = jnp.linspace(1.0, 0.1, nlev)  # From surface to top
    
    # Create lat/lon grid
    lat = jnp.linspace(-90, 90, nlat)
    lon = jnp.linspace(0, 360, nlon)
    
    return TestGeometry(
        fsg=fsg,
        lat=lat, 
        lon=lon,
        nodal_shape=(nlev, nlat, nlon)
    )


def test_icon_physics_with_clouds():
    """Test ICON physics with shallow cloud scheme"""
    
    # Setup dimensions
    nlev, nlat, nlon = 20, 32, 64
    
    # Create test state with realistic profile
    pressure_1d = jnp.linspace(100000, 20000, nlev)
    temperature_1d = jnp.linspace(288, 220, nlev)
    
    # Broadcast to 3D
    temperature = jnp.broadcast_to(temperature_1d[:, None, None], (nlev, nlat, nlon))
    pressure = jnp.broadcast_to(pressure_1d[:, None, None], (nlev, nlat, nlon))
    
    # Create humid layer in mid-troposphere
    qs = jax.vmap(jax.vmap(jax.vmap(saturation_specific_humidity)))(pressure, temperature)
    specific_humidity = 0.5 * qs  # 50% RH base
    # Add moist layer
    specific_humidity = specific_humidity.at[8:12, :, :].set(0.9 * qs[8:12, :, :])
    
    # Initialize cloud water and ice tracers
    cloud_water = jnp.zeros((nlev, nlat, nlon))
    cloud_water = cloud_water.at[8:12, :, :].set(0.0005)  # Some existing cloud water
    cloud_ice = jnp.zeros((nlev, nlat, nlon))
    
    # Create physics state
    state = PhysicsState(
        u_wind=jnp.zeros((nlev, nlat, nlon)),
        v_wind=jnp.zeros((nlev, nlat, nlon)),
        temperature=temperature,
        specific_humidity=specific_humidity,
        geopotential=jnp.zeros((nlev, nlat, nlon)),
        surface_pressure=jnp.ones((nlat, nlon)),  # Normalized
        tracers={'qc': cloud_water, 'qi': cloud_ice}
    )
    
    # Create other required objects
    boundaries = BoundaryData.zeros((nlat, nlon))
    geometry = create_test_geometry(nlev, nlat, nlon)
    date = DateData.zeros()
    
    # Initialize ICON physics
    physics = IconPhysics(write_output=True)
    
    # Compute tendencies
    tendencies, physics_data = physics.compute_tendencies(
        state, boundaries, geometry, date
    )
    
    # Check results
    print("Test Results:")
    print(f"Temperature tendency shape: {tendencies.temperature.shape}")
    print(f"Temperature tendency range: [{jnp.min(tendencies.temperature):.6f}, {jnp.max(tendencies.temperature):.6f}] K/s")
    print(f"Humidity tendency range: [{jnp.min(tendencies.specific_humidity):.6e}, {jnp.max(tendencies.specific_humidity):.6e}] kg/kg/s")
    
    # Check cloud diagnostics
    if 'cloud_data' in physics_data.__dict__ and physics_data.cloud_data:
        cloud_data = physics_data.cloud_data
        print(f"\nCloud diagnostics:")
        print(f"Cloud fraction shape: {cloud_data['cloud_fraction'].shape}")
        print(f"Max cloud fraction: {jnp.max(cloud_data['cloud_fraction']):.3f}")
        print(f"Total precipitation flux: {jnp.sum(cloud_data['total_precipitation']):.6f} kg/m²/s")
    
    # Check tracer tendencies
    if 'qc' in tendencies.tracers:
        print(f"\nCloud water tendency range: [{jnp.min(tendencies.tracers['qc']):.6e}, {jnp.max(tendencies.tracers['qc']):.6e}] kg/kg/s")
    if 'qi' in tendencies.tracers:
        print(f"Cloud ice tendency range: [{jnp.min(tendencies.tracers['qi']):.6e}, {jnp.max(tendencies.tracers['qi']):.6e}] kg/kg/s")
    
    # Verify no NaNs
    assert not jnp.any(jnp.isnan(tendencies.temperature)), "NaN in temperature tendencies"
    assert not jnp.any(jnp.isnan(tendencies.specific_humidity)), "NaN in humidity tendencies"
    
    print("\nTest passed!")


if __name__ == "__main__":
    test_icon_physics_with_clouds()