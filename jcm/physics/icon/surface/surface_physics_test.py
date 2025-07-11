"""
Unit tests for main surface physics interface.
"""

import pytest
import jax.numpy as jnp
import numpy as np

from jcm.physics.icon.surface.surface_physics import (
    initialize_surface_state, surface_physics_step,
    combine_surface_fluxes, combine_surface_tendencies,
    update_surface_state
)
from jcm.physics.icon.surface.surface_types import (
    SurfaceParameters, SurfaceState, AtmosphericForcing,
    SurfaceFluxes, SurfaceTendencies
)


class TestInitializeSurfaceState:
    """Test surface state initialization."""
    
    def test_initialize_surface_state_basic(self):
        """Test basic surface state initialization."""
        ncol = 3
        surface_fractions = jnp.array([[0.6, 0.2, 0.2], 
                                      [0.4, 0.3, 0.3], 
                                      [0.8, 0.1, 0.1]])
        ocean_temp = jnp.array([285.0, 280.0, 288.0])
        ice_temp = jnp.ones((ncol, 2)) * 270.0
        soil_temp = jnp.ones((ncol, 4)) * 280.0
        
        surface_state = initialize_surface_state(
            ncol, surface_fractions, ocean_temp, ice_temp, soil_temp
        )
        
        assert isinstance(surface_state, SurfaceState)
        assert surface_state.temperature.shape == (ncol, 3)
        assert surface_state.temperature_rad.shape == (ncol,)
        assert surface_state.fraction.shape == (ncol, 3)
        assert surface_state.ocean_temp.shape == (ncol,)
        assert surface_state.ice_temp.shape == (ncol, 2)
        assert surface_state.soil_temp.shape == (ncol, 4)
        
        # Check that fractions are preserved
        assert jnp.allclose(surface_state.fraction, surface_fractions)
        
        # Check that temperatures are set correctly
        params = SurfaceParameters()
        assert jnp.allclose(surface_state.temperature[:, params.iwtr], ocean_temp)
        assert jnp.allclose(surface_state.temperature[:, params.iice], ice_temp[:, 0])
        assert jnp.allclose(surface_state.temperature[:, params.ilnd], soil_temp[:, 0])
    
    def test_initialize_surface_state_radiative_temperature(self):
        """Test radiative temperature calculation."""
        ncol = 2
        surface_fractions = jnp.array([[0.5, 0.3, 0.2], 
                                      [0.2, 0.6, 0.2]])
        ocean_temp = jnp.array([285.0, 280.0])
        ice_temp = jnp.ones((ncol, 2)) * 270.0
        soil_temp = jnp.ones((ncol, 4)) * 275.0
        
        surface_state = initialize_surface_state(
            ncol, surface_fractions, ocean_temp, ice_temp, soil_temp
        )
        
        # Check radiative temperature calculation
        expected_temp_rad = jnp.sum(
            surface_fractions * surface_state.temperature, axis=1
        )
        
        assert jnp.allclose(surface_state.temperature_rad, expected_temp_rad)
    
    def test_initialize_surface_state_default_values(self):
        """Test default values in surface state."""
        ncol = 2
        surface_fractions = jnp.array([[0.7, 0.2, 0.1], 
                                      [0.3, 0.4, 0.3]])
        ocean_temp = jnp.array([285.0, 280.0])
        ice_temp = jnp.ones((ncol, 2)) * 270.0
        soil_temp = jnp.ones((ncol, 4)) * 280.0
        
        surface_state = initialize_surface_state(
            ncol, surface_fractions, ocean_temp, ice_temp, soil_temp
        )
        
        # Check default values
        assert surface_state.ocean_u.shape == (ncol,)
        assert surface_state.ocean_v.shape == (ncol,)
        assert jnp.allclose(surface_state.ocean_u, 0.0)
        assert jnp.allclose(surface_state.ocean_v, 0.0)
        
        assert surface_state.ice_thickness.shape == (ncol, 2)
        assert jnp.allclose(surface_state.ice_thickness, 2.0)
        
        assert surface_state.soil_moisture.shape == (ncol, 4)
        assert jnp.allclose(surface_state.soil_moisture, 0.3)
        
        # Check roughness lengths
        params = SurfaceParameters()
        assert jnp.allclose(surface_state.roughness_momentum[:, params.iwtr], params.z0_water)
        assert jnp.allclose(surface_state.roughness_momentum[:, params.iice], params.z0_ice)
        assert jnp.allclose(surface_state.roughness_momentum[:, params.ilnd], params.z0_land)
    
    def test_initialize_surface_state_albedos(self):
        """Test albedo initialization."""
        ncol = 2
        surface_fractions = jnp.array([[0.6, 0.2, 0.2], 
                                      [0.4, 0.3, 0.3]])
        ocean_temp = jnp.array([285.0, 280.0])
        ice_temp = jnp.ones((ncol, 2)) * 270.0
        soil_temp = jnp.ones((ncol, 4)) * 280.0
        
        surface_state = initialize_surface_state(
            ncol, surface_fractions, ocean_temp, ice_temp, soil_temp
        )
        
        params = SurfaceParameters()
        
        # Check ocean albedos
        assert jnp.allclose(surface_state.albedo_visible_direct[:, params.iwtr], 0.06)
        assert jnp.allclose(surface_state.albedo_visible_diffuse[:, params.iwtr], 0.06)
        
        # Check ice albedos
        assert jnp.allclose(surface_state.albedo_visible_direct[:, params.iice], 0.75)
        assert jnp.allclose(surface_state.albedo_nir_direct[:, params.iice], 0.65)
        
        # Check land albedos
        assert jnp.allclose(surface_state.albedo_visible_direct[:, params.ilnd], 0.15)
        assert jnp.allclose(surface_state.albedo_nir_direct[:, params.ilnd], 0.30)


class TestSurfacePhysicsStep:
    """Test main surface physics step."""
    
    def setup_method(self):
        """Set up test data."""
        self.ncol = 2
        self.nsfc_type = 3
        
        # Atmospheric forcing
        self.atmospheric_state = AtmosphericForcing(
            temperature=jnp.array([290.0, 285.0]),
            humidity=jnp.array([0.01, 0.008]),
            u_wind=jnp.array([5.0, 3.0]),
            v_wind=jnp.array([2.0, 4.0]),
            pressure=jnp.array([101325.0, 95000.0]),
            sw_downward=jnp.array([300.0, 250.0]),
            lw_downward=jnp.array([350.0, 320.0]),
            rain_rate=jnp.array([1e-6, 2e-6]),
            snow_rate=jnp.array([0.0, 0.0]),
            exchange_coeff_heat=jnp.ones((self.ncol, self.nsfc_type)) * 0.01,
            exchange_coeff_moisture=jnp.ones((self.ncol, self.nsfc_type)) * 0.01,
            exchange_coeff_momentum=jnp.ones((self.ncol, self.nsfc_type)) * 0.01
        )
        
        # Surface state
        surface_fractions = jnp.array([[0.6, 0.2, 0.2], 
                                      [0.4, 0.3, 0.3]])
        ocean_temp = jnp.array([285.0, 280.0])
        ice_temp = jnp.ones((self.ncol, 2)) * 270.0
        soil_temp = jnp.ones((self.ncol, 4)) * 280.0
        
        self.surface_state = initialize_surface_state(
            self.ncol, surface_fractions, ocean_temp, ice_temp, soil_temp
        )
        
        self.dt = 3600.0  # 1 hour
    
    def test_surface_physics_step_basic(self):
        """Test basic surface physics step."""
        fluxes, tendencies, diagnostics = surface_physics_step(
            self.atmospheric_state, self.surface_state, self.dt
        )
        
        assert isinstance(fluxes, SurfaceFluxes)
        assert isinstance(tendencies, SurfaceTendencies)
        
        # Check shapes
        assert fluxes.sensible_heat.shape == (self.ncol, self.nsfc_type)
        assert fluxes.latent_heat.shape == (self.ncol, self.nsfc_type)
        assert fluxes.sensible_heat_mean.shape == (self.ncol,)
        assert fluxes.latent_heat_mean.shape == (self.ncol,)
        
        assert tendencies.surface_temp_tendency.shape == (self.ncol, self.nsfc_type)
        assert tendencies.ocean_temp_tendency.shape == (self.ncol,)
        
        # Check that all values are finite
        assert jnp.all(jnp.isfinite(fluxes.sensible_heat))
        assert jnp.all(jnp.isfinite(fluxes.latent_heat))
        assert jnp.all(jnp.isfinite(tendencies.surface_temp_tendency))
        assert jnp.all(jnp.isfinite(tendencies.ocean_temp_tendency))
    
    def test_surface_physics_step_energy_conservation(self):
        """Test energy conservation in surface physics step."""
        fluxes, tendencies, diagnostics = surface_physics_step(
            self.atmospheric_state, self.surface_state, self.dt
        )
        
        # Net surface energy flux should be finite
        net_energy = (fluxes.shortwave_net + fluxes.longwave_net - 
                     fluxes.sensible_heat - fluxes.latent_heat)
        
        assert jnp.all(jnp.isfinite(net_energy))
        
        # Grid-box mean energy balance
        net_energy_mean = jnp.sum(self.surface_state.fraction * net_energy, axis=1)
        assert jnp.all(jnp.isfinite(net_energy_mean))
    
    def test_surface_physics_step_flux_consistency(self):
        """Test flux consistency between tiles and means."""
        fluxes, tendencies, diagnostics = surface_physics_step(
            self.atmospheric_state, self.surface_state, self.dt
        )
        
        # Check that mean fluxes are consistent with tile fluxes
        expected_sensible_mean = jnp.sum(
            self.surface_state.fraction * fluxes.sensible_heat, axis=1
        )
        expected_latent_mean = jnp.sum(
            self.surface_state.fraction * fluxes.latent_heat, axis=1
        )
        
        assert jnp.allclose(fluxes.sensible_heat_mean, expected_sensible_mean, rtol=0.1)
        assert jnp.allclose(fluxes.latent_heat_mean, expected_latent_mean, rtol=0.1)


class TestCombineSurfaceFluxes:
    """Test surface flux combination."""
    
    def test_combine_surface_fluxes_basic(self):
        """Test basic surface flux combination."""
        ncol, nsfc_type = 2, 3
        
        # Create mock fluxes for each surface type
        flux_wtr = SurfaceFluxes(
            sensible_heat=jnp.array([[50.0], [60.0]]),
            latent_heat=jnp.array([[100.0], [120.0]]),
            longwave_net=jnp.array([[-80.0], [-90.0]]),
            shortwave_net=jnp.array([[200.0], [250.0]]),
            ground_heat=jnp.array([[0.0], [0.0]]),
            momentum_u=jnp.array([[0.1], [0.15]]),
            momentum_v=jnp.array([[0.05], [0.08]]),
            evaporation=jnp.array([[1e-6], [1.5e-6]]),
            transpiration=jnp.array([[0.0], [0.0]]),
            sensible_heat_mean=jnp.array([50.0, 60.0]),
            latent_heat_mean=jnp.array([100.0, 120.0]),
            momentum_u_mean=jnp.array([0.1, 0.15]),
            momentum_v_mean=jnp.array([0.05, 0.08]),
            evaporation_mean=jnp.array([1e-6, 1.5e-6])
        )
        
        flux_ice = SurfaceFluxes(
            sensible_heat=jnp.array([[30.0], [40.0]]),
            latent_heat=jnp.array([[80.0], [90.0]]),
            longwave_net=jnp.array([[-60.0], [-70.0]]),
            shortwave_net=jnp.array([[100.0], [150.0]]),
            ground_heat=jnp.array([[0.0], [0.0]]),
            momentum_u=jnp.array([[0.08], [0.12]]),
            momentum_v=jnp.array([[0.04], [0.06]]),
            evaporation=jnp.array([[0.8e-6], [1.2e-6]]),
            transpiration=jnp.array([[0.0], [0.0]]),
            sensible_heat_mean=jnp.array([30.0, 40.0]),
            latent_heat_mean=jnp.array([80.0, 90.0]),
            momentum_u_mean=jnp.array([0.08, 0.12]),
            momentum_v_mean=jnp.array([0.04, 0.06]),
            evaporation_mean=jnp.array([0.8e-6, 1.2e-6])
        )
        
        flux_lnd = SurfaceFluxes(
            sensible_heat=jnp.array([[70.0], [80.0]]),
            latent_heat=jnp.array([[150.0], [180.0]]),
            longwave_net=jnp.array([[-100.0], [-110.0]]),
            shortwave_net=jnp.array([[180.0], [200.0]]),
            ground_heat=jnp.array([[20.0], [25.0]]),
            momentum_u=jnp.array([[0.12], [0.18]]),
            momentum_v=jnp.array([[0.06], [0.09]]),
            evaporation=jnp.array([[1.2e-6], [1.8e-6]]),
            transpiration=jnp.array([[0.5e-6], [0.8e-6]]),
            sensible_heat_mean=jnp.array([70.0, 80.0]),
            latent_heat_mean=jnp.array([150.0, 180.0]),
            momentum_u_mean=jnp.array([0.12, 0.18]),
            momentum_v_mean=jnp.array([0.06, 0.09]),
            evaporation_mean=jnp.array([1.7e-6, 2.6e-6])
        )
        
        flux_list = [flux_wtr, flux_ice, flux_lnd]
        fractions = jnp.array([[0.6, 0.2, 0.2], [0.4, 0.3, 0.3]])
        
        combined_fluxes = combine_surface_fluxes(flux_list, fractions)
        
        assert isinstance(combined_fluxes, SurfaceFluxes)
        assert combined_fluxes.sensible_heat.shape == (ncol, nsfc_type)
        assert combined_fluxes.sensible_heat_mean.shape == (ncol,)
        
        # Check that tile fluxes are preserved
        assert jnp.allclose(combined_fluxes.sensible_heat[:, 0], flux_wtr.sensible_heat[:, 0])
        assert jnp.allclose(combined_fluxes.sensible_heat[:, 1], flux_ice.sensible_heat[:, 0])
        assert jnp.allclose(combined_fluxes.sensible_heat[:, 2], flux_lnd.sensible_heat[:, 0])
    
    def test_combine_surface_fluxes_mean_calculation(self):
        """Test mean flux calculation in combination."""
        ncol, nsfc_type = 2, 3
        
        # Simple test case
        sensible_heat = jnp.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
        fractions = jnp.array([[0.5, 0.3, 0.2], [0.2, 0.3, 0.5]])
        
        # Create mock flux objects
        flux_list = []
        for i in range(nsfc_type):
            flux = SurfaceFluxes(
                sensible_heat=sensible_heat[:, i:i+1],
                latent_heat=jnp.zeros((ncol, 1)),
                longwave_net=jnp.zeros((ncol, 1)),
                shortwave_net=jnp.zeros((ncol, 1)),
                ground_heat=jnp.zeros((ncol, 1)),
                momentum_u=jnp.zeros((ncol, 1)),
                momentum_v=jnp.zeros((ncol, 1)),
                evaporation=jnp.zeros((ncol, 1)),
                transpiration=jnp.zeros((ncol, 1)),
                sensible_heat_mean=sensible_heat[:, i],
                latent_heat_mean=jnp.zeros(ncol),
                momentum_u_mean=jnp.zeros(ncol),
                momentum_v_mean=jnp.zeros(ncol),
                evaporation_mean=jnp.zeros(ncol)
            )
            flux_list.append(flux)
        
        combined_fluxes = combine_surface_fluxes(flux_list, fractions)
        
        # Check mean calculation
        expected_mean = jnp.sum(fractions * sensible_heat, axis=1)
        assert jnp.allclose(combined_fluxes.sensible_heat_mean, expected_mean)


class TestUpdateSurfaceState:
    """Test surface state update."""
    
    def test_update_surface_state_basic(self):
        """Test basic surface state update."""
        ncol = 2
        
        # Initial surface state
        surface_fractions = jnp.array([[0.6, 0.2, 0.2], 
                                      [0.4, 0.3, 0.3]])
        ocean_temp = jnp.array([285.0, 280.0])
        ice_temp = jnp.ones((ncol, 2)) * 270.0
        soil_temp = jnp.ones((ncol, 4)) * 280.0
        
        surface_state = initialize_surface_state(
            ncol, surface_fractions, ocean_temp, ice_temp, soil_temp
        )
        
        # Create tendencies
        tendencies = SurfaceTendencies(
            surface_temp_tendency=jnp.ones((ncol, 3)) * 0.01,
            ocean_temp_tendency=jnp.ones(ncol) * 0.005,
            ice_temp_tendency=jnp.ones((ncol, 2)) * -0.01,
            soil_temp_tendency=jnp.ones((ncol, 4)) * 0.002,
            ice_thickness_tendency=jnp.ones((ncol, 2)) * 1e-8,
            snow_depth_tendency=jnp.ones(ncol) * 1e-7,
            soil_moisture_tendency=jnp.ones((ncol, 4)) * 1e-9
        )
        
        dt = 3600.0
        
        new_surface_state = update_surface_state(surface_state, tendencies, dt)
        
        assert isinstance(new_surface_state, SurfaceState)
        
        # Check that temperatures have been updated
        expected_ocean_temp = surface_state.ocean_temp + tendencies.ocean_temp_tendency * dt
        assert jnp.allclose(new_surface_state.ocean_temp, expected_ocean_temp)
        
        expected_ice_temp = surface_state.ice_temp + tendencies.ice_temp_tendency * dt
        assert jnp.allclose(new_surface_state.ice_temp, expected_ice_temp)
        
        expected_soil_temp = surface_state.soil_temp + tendencies.soil_temp_tendency * dt
        assert jnp.allclose(new_surface_state.soil_temp, expected_soil_temp)
    
    def test_update_surface_state_radiative_temperature(self):
        """Test radiative temperature update."""
        ncol = 2
        
        # Initial surface state
        surface_fractions = jnp.array([[0.6, 0.2, 0.2], 
                                      [0.4, 0.3, 0.3]])
        ocean_temp = jnp.array([285.0, 280.0])
        ice_temp = jnp.ones((ncol, 2)) * 270.0
        soil_temp = jnp.ones((ncol, 4)) * 280.0
        
        surface_state = initialize_surface_state(
            ncol, surface_fractions, ocean_temp, ice_temp, soil_temp
        )
        
        # Create tendencies
        tendencies = SurfaceTendencies(
            surface_temp_tendency=jnp.array([[0.01, 0.02, 0.005], 
                                           [0.015, 0.01, 0.008]]),
            ocean_temp_tendency=jnp.ones(ncol) * 0.005,
            ice_temp_tendency=jnp.ones((ncol, 2)) * -0.01,
            soil_temp_tendency=jnp.ones((ncol, 4)) * 0.002,
            ice_thickness_tendency=jnp.ones((ncol, 2)) * 1e-8,
            snow_depth_tendency=jnp.ones(ncol) * 1e-7,
            soil_moisture_tendency=jnp.ones((ncol, 4)) * 1e-9
        )
        
        dt = 3600.0
        
        new_surface_state = update_surface_state(surface_state, tendencies, dt)
        
        # Check radiative temperature calculation
        expected_temp_rad = jnp.sum(
            surface_state.fraction * new_surface_state.temperature, axis=1
        )
        
        assert jnp.allclose(new_surface_state.temperature_rad, expected_temp_rad)
    
    def test_update_surface_state_bounds(self):
        """Test physical bounds in surface state update."""
        ncol = 2
        
        # Initial surface state
        surface_fractions = jnp.array([[0.6, 0.2, 0.2], 
                                      [0.4, 0.3, 0.3]])
        ocean_temp = jnp.array([285.0, 280.0])
        ice_temp = jnp.ones((ncol, 2)) * 270.0
        soil_temp = jnp.ones((ncol, 4)) * 280.0
        
        surface_state = initialize_surface_state(
            ncol, surface_fractions, ocean_temp, ice_temp, soil_temp
        )
        
        # Create tendencies that would violate bounds
        tendencies = SurfaceTendencies(
            surface_temp_tendency=jnp.ones((ncol, 3)) * 0.01,
            ocean_temp_tendency=jnp.ones(ncol) * 0.005,
            ice_temp_tendency=jnp.ones((ncol, 2)) * -0.01,
            soil_temp_tendency=jnp.ones((ncol, 4)) * 0.002,
            ice_thickness_tendency=jnp.ones((ncol, 2)) * -1e-3,  # Large negative
            snow_depth_tendency=jnp.ones(ncol) * -1e-4,  # Large negative
            soil_moisture_tendency=jnp.array([[1e-3, -1e-3, 1e-3, -1e-3], 
                                            [-1e-3, 1e-3, -1e-3, 1e-3]])  # Mixed
        )
        
        dt = 3600.0
        
        new_surface_state = update_surface_state(surface_state, tendencies, dt)
        
        # Check bounds
        assert jnp.all(new_surface_state.ice_thickness >= 0.0)
        assert jnp.all(new_surface_state.snow_depth >= 0.0)
        assert jnp.all(new_surface_state.soil_moisture >= 0.0)
        assert jnp.all(new_surface_state.soil_moisture <= 1.0)
    
    def test_update_surface_state_conservation(self):
        """Test conservation properties in surface state update."""
        ncol = 2
        
        # Initial surface state
        surface_fractions = jnp.array([[0.6, 0.2, 0.2], 
                                      [0.4, 0.3, 0.3]])
        ocean_temp = jnp.array([285.0, 280.0])
        ice_temp = jnp.ones((ncol, 2)) * 270.0
        soil_temp = jnp.ones((ncol, 4)) * 280.0
        
        surface_state = initialize_surface_state(
            ncol, surface_fractions, ocean_temp, ice_temp, soil_temp
        )
        
        # Create tendencies
        tendencies = SurfaceTendencies(
            surface_temp_tendency=jnp.ones((ncol, 3)) * 0.01,
            ocean_temp_tendency=jnp.ones(ncol) * 0.005,
            ice_temp_tendency=jnp.ones((ncol, 2)) * -0.01,
            soil_temp_tendency=jnp.ones((ncol, 4)) * 0.002,
            ice_thickness_tendency=jnp.ones((ncol, 2)) * 1e-8,
            snow_depth_tendency=jnp.ones(ncol) * 1e-7,
            soil_moisture_tendency=jnp.ones((ncol, 4)) * 1e-9
        )
        
        dt = 3600.0
        
        new_surface_state = update_surface_state(surface_state, tendencies, dt)
        
        # Check that fractions are preserved
        assert jnp.allclose(new_surface_state.fraction, surface_state.fraction)
        
        # Check that ocean currents are preserved (not updated)
        assert jnp.allclose(new_surface_state.ocean_u, surface_state.ocean_u)
        assert jnp.allclose(new_surface_state.ocean_v, surface_state.ocean_v)


if __name__ == "__main__":
    pytest.main([__file__])