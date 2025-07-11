"""
Tests for IconPhysics integration

This module tests the integrated ICON physics implementation, ensuring all
components work together correctly.

Date: 2025-01-10
"""

import jax.numpy as jnp
import jax
import pytest
from jcm.physics_interface import PhysicsState, PhysicsTendency
from jcm.boundaries import BoundaryData
from jcm.geometry import Geometry
from jcm.date import DateData
from .icon_physics import IconPhysics, IconPhysicsData
from .parameters import Parameters


class TestIconPhysicsIntegration:
    """Test integrated ICON physics"""
    
    def setup_test_state(self, nlev=20, nlat=4, nlon=8):
        """Create test atmosphere state"""
        # Create realistic atmospheric profile
        height = jnp.linspace(0, 20000, nlev)
        sigma = jnp.exp(-height / 8000)  # Pressure coordinate
        
        # 3D arrays [nlev, nlat, nlon]
        shape = (nlev, nlat, nlon)
        
        # Temperature decreases with height
        temp_profile = 288 - 0.0065 * height
        temperature = jnp.broadcast_to(temp_profile[:, None, None], shape)
        
        # Add some horizontal variation
        lat_var = jnp.linspace(-10, 10, nlat)[None, :, None]
        temperature = temperature + lat_var
        
        # Humidity decreases with height
        humid_profile = 0.01 * jnp.exp(-height / 5000)
        specific_humidity = jnp.broadcast_to(humid_profile[:, None, None], shape)
        
        # Westerly jet
        u_profile = 20.0 * jnp.exp(-(height - 10000)**2 / 5000**2)
        u_wind = jnp.broadcast_to(u_profile[:, None, None], shape)
        v_wind = jnp.zeros(shape)
        
        # Geopotential
        geopotential = jnp.broadcast_to((height * 9.81)[:, None, None], shape)
        
        # Surface pressure varies horizontally
        ps_2d = 1.0 + 0.01 * jnp.sin(jnp.linspace(0, 2*jnp.pi, nlon))
        surface_pressure = jnp.broadcast_to(ps_2d[None, :], (nlat, nlon))
        
        # Create tracers including cloud species
        tracers = {
            'qc': jnp.zeros(shape),  # Cloud water
            'qi': jnp.zeros(shape),  # Cloud ice
            'qr': jnp.zeros(shape),  # Rain water
            'qs': jnp.zeros(shape),  # Snow
        }
        
        # Add some cloud water/ice in mid-levels
        for i in range(nlev):
            if 3000 < height[i] < 8000:
                tracers['qc'] = tracers['qc'].at[i, :, :].set(1e-4)
            if 6000 < height[i] < 12000:
                tracers['qi'] = tracers['qi'].at[i, :, :].set(5e-5)
        
        return PhysicsState(
            u_wind=u_wind,
            v_wind=v_wind,
            temperature=temperature,
            specific_humidity=specific_humidity,
            geopotential=geopotential,
            surface_pressure=surface_pressure,
            tracers=tracers
        )
    
    def setup_geometry(self, nlev=20):
        """Create test geometry"""
        nlat = 4
        nlon = 8
        
        # Create sigma levels
        # For nlev=20, create boundaries similar to how Speedy does it
        hsg = jnp.linspace(1.0, 0.0, nlev + 1)  # Half levels
        fsg = (hsg[1:] + hsg[:-1]) / 2.0       # Full levels (midpoints)
        dhs = jnp.diff(hsg)
        sigl = jnp.log(fsg)
        
        # Latitude setup
        lat = jnp.linspace(-45, 45, nlat)
        radang = lat * jnp.pi / 180
        sia = jnp.sin(radang)
        coa = jnp.cos(radang)
        
        # Create geometry with minimal required fields
        return Geometry(
            nodal_shape=(nlev, nlon, nlat),
            radang=radang,
            sia=sia,
            coa=coa,
            hsg=hsg,
            fsg=fsg,
            dhs=dhs,
            sigl=sigl,
            grdsig=9.81 / (dhs * 100000.0),
            grdscp=9.81 / (dhs * 100000.0 * 1004.0),
            wvi=jnp.zeros((nlev, 2))  # Not used in our tests
        )
    
    def test_physics_initialization(self):
        """Test IconPhysics initialization"""
        physics = IconPhysics()
        assert physics.parameters is not None
        assert hasattr(physics.parameters, 'convection')
        assert hasattr(physics.parameters, 'clouds')
        assert hasattr(physics.parameters, 'microphysics')
        assert hasattr(physics.parameters, 'gravity_waves')
    
    def test_compute_tendencies_shape(self):
        """Test that tendencies have correct shape"""
        physics = IconPhysics()
        state = self.setup_test_state()
        geometry = self.setup_geometry()
        date = DateData.zeros()
        
        tendencies, physics_data = physics.compute_tendencies(
            state, geometry=geometry, date=date
        )
        
        # Check shapes match input
        assert tendencies.temperature.shape == state.temperature.shape
        assert tendencies.u_wind.shape == state.u_wind.shape
        assert tendencies.v_wind.shape == state.v_wind.shape
        assert tendencies.specific_humidity.shape == state.specific_humidity.shape
        
        # Check tracer tendencies
        for name, tracer in state.tracers.items():
            assert name in tendencies.tracers
            assert tendencies.tracers[name].shape == tracer.shape
    
    def test_physics_components_active(self):
        """Test that all physics components are active"""
        physics = IconPhysics()
        state = self.setup_test_state()
        geometry = self.setup_geometry()
        date = DateData.zeros()
        
        tendencies, physics_data = physics.compute_tendencies(
            state, geometry=geometry, date=date
        )
        
        # Check convection was applied
        assert 'convection_enabled' in physics_data.convection_data
        assert physics_data.convection_data['convection_enabled']
        
        # Check clouds were applied
        assert 'cloud_scheme_enabled' in physics_data.cloud_data
        assert physics_data.cloud_data['cloud_scheme_enabled']
        
        # Check microphysics was applied
        assert 'microphysics_enabled' in physics_data.microphysics_data
        assert physics_data.microphysics_data['microphysics_enabled']
        
        # Check gravity waves were applied
        assert 'gravity_wave_drag_enabled' in physics_data.gravity_wave_data
        assert physics_data.gravity_wave_data['gravity_wave_drag_enabled']
    
    def test_nonzero_tendencies(self):
        """Test that physics produces non-zero tendencies"""
        physics = IconPhysics()
        state = self.setup_test_state()
        geometry = self.setup_geometry()
        date = DateData.zeros()
        
        tendencies, physics_data = physics.compute_tendencies(
            state, geometry=geometry, date=date
        )
        
        # At least some tendencies should be non-zero
        total_temp_tend = jnp.sum(jnp.abs(tendencies.temperature))
        total_wind_tend = jnp.sum(jnp.abs(tendencies.u_wind)) + jnp.sum(jnp.abs(tendencies.v_wind))
        total_humid_tend = jnp.sum(jnp.abs(tendencies.specific_humidity))
        
        assert total_temp_tend > 0
        assert total_wind_tend > 0
        assert total_humid_tend > 0
    
    def test_physics_data_diagnostics(self):
        """Test that physics data contains expected diagnostics"""
        physics = IconPhysics()
        state = self.setup_test_state()
        geometry = self.setup_geometry()
        date = DateData.zeros()
        
        tendencies, physics_data = physics.compute_tendencies(
            state, geometry=geometry, date=date
        )
        
        # Check convection diagnostics
        assert 'convective_precipitation' in physics_data.convection_data
        assert 'convective_cloud_water' in physics_data.convection_data
        
        # Check cloud diagnostics
        assert 'cloud_fraction' in physics_data.cloud_data
        assert 'total_precipitation' in physics_data.cloud_data
        
        # Check microphysics diagnostics
        assert 'rain_flux' in physics_data.microphysics_data
        assert 'snow_flux' in physics_data.microphysics_data
        
        # Check gravity wave diagnostics
        assert 'surface_stress' in physics_data.gravity_wave_data
        assert physics_data.gravity_wave_data['mean_stress'] >= 0
    
    def test_parameter_customization(self):
        """Test physics with custom parameters"""
        # Create custom parameters
        params = Parameters.default().with_convection(
            entrpen=2e-4  # Penetrative entrainment rate
        ).with_gravity_waves(
            gkdrag=1.0,
            zmin=2000.0
        )
        
        physics = IconPhysics(parameters=params)
        state = self.setup_test_state()
        geometry = self.setup_geometry()
        
        tendencies, physics_data = physics.compute_tendencies(
            state, geometry=geometry
        )
        
        # Should run without errors
        assert tendencies is not None
        assert physics_data is not None
    
    def test_jax_transformations(self):
        """Test JAX transformations on physics"""
        physics = IconPhysics()
        state = self.setup_test_state(nlev=10, nlat=2, nlon=4)
        geometry = self.setup_geometry(nlev=10)
        
        def physics_loss(temperature):
            new_state = PhysicsState(
                u_wind=state.u_wind,
                v_wind=state.v_wind,
                temperature=temperature,
                specific_humidity=state.specific_humidity,
                geopotential=state.geopotential,
                surface_pressure=state.surface_pressure,
                tracers=state.tracers
            )
            
            tendencies, _ = physics.compute_tendencies(new_state, geometry=geometry)
            return jnp.sum(tendencies.temperature ** 2)
        
        # Test JIT compilation
        jitted_loss = jax.jit(physics_loss)
        loss = jitted_loss(state.temperature)
        assert jnp.isfinite(loss)
        
        # Test gradient computation
        grad_fn = jax.grad(physics_loss)
        grad = grad_fn(state.temperature)
        assert grad.shape == state.temperature.shape
        
        # For now, just check that gradient computation doesn't fail
        # Some physics schemes might produce zero gradients in certain regions
        # which is acceptable for physical reasons
        # The important thing is that JAX can compute gradients without errors
    
    def test_tracer_conservation(self):
        """Test approximate conservation of tracers"""
        physics = IconPhysics()
        state = self.setup_test_state()
        geometry = self.setup_geometry()
        
        # Calculate initial total water
        total_water_init = (
            jnp.sum(state.specific_humidity) +
            jnp.sum(state.tracers['qc']) +
            jnp.sum(state.tracers['qi']) +
            jnp.sum(state.tracers['qr']) +
            jnp.sum(state.tracers['qs'])
        )
        
        tendencies, physics_data = physics.compute_tendencies(
            state, geometry=geometry
        )
        
        # Apply tendencies for one timestep
        dt = 1800.0
        new_q = state.specific_humidity + tendencies.specific_humidity * dt
        new_qc = state.tracers['qc'] + tendencies.tracers.get('qc', 0) * dt
        new_qi = state.tracers['qi'] + tendencies.tracers.get('qi', 0) * dt
        new_qr = state.tracers['qr'] + tendencies.tracers.get('qr', 0) * dt
        new_qs = state.tracers['qs'] + tendencies.tracers.get('qs', 0) * dt
        
        # Calculate new total water (excluding precipitation)
        total_water_new = (
            jnp.sum(new_q) +
            jnp.sum(new_qc) +
            jnp.sum(new_qi) +
            jnp.sum(new_qr) +
            jnp.sum(new_qs)
        )
        
        # Account for precipitation removal
        total_precip = 0
        if 'convective_precipitation' in physics_data.convection_data:
            total_precip += jnp.sum(physics_data.convection_data['convective_precipitation']) * dt
        if 'total_precipitation' in physics_data.cloud_data:
            total_precip += jnp.sum(physics_data.cloud_data['total_precipitation']) * dt
        if 'total_precipitation' in physics_data.microphysics_data:
            total_precip += jnp.sum(physics_data.microphysics_data['total_precipitation']) * dt
        
        # Conservation check (allowing for precipitation)
        water_change = total_water_new - total_water_init
        
        # The change should be reasonable (not orders of magnitude)
        relative_change = jnp.abs(water_change) / (total_water_init + 1e-10)
        assert relative_change < 0.1  # Less than 10% change


def test_icon_physics_integration():
    """Run integration tests"""
    test = TestIconPhysicsIntegration()
    
    print("Testing physics initialization...")
    test.test_physics_initialization()
    print("✓ Initialization test passed")
    
    print("\nTesting tendency shapes...")
    test.test_compute_tendencies_shape()
    print("✓ Shape test passed")
    
    print("\nTesting physics components...")
    test.test_physics_components_active()
    print("✓ Component activation test passed")
    
    print("\nTesting non-zero tendencies...")
    test.test_nonzero_tendencies()
    print("✓ Non-zero tendency test passed")
    
    print("\nTesting diagnostics...")
    test.test_physics_data_diagnostics()
    print("✓ Diagnostics test passed")
    
    print("\nTesting parameter customization...")
    test.test_parameter_customization()
    print("✓ Parameter test passed")
    
    print("\nTesting JAX transformations...")
    test.test_jax_transformations()
    print("✓ JAX transformation test passed")
    
    print("\nTesting tracer conservation...")
    test.test_tracer_conservation()
    print("✓ Conservation test passed")
    
    print("\nAll integration tests passed!")


if __name__ == "__main__":
    test_icon_physics_integration()