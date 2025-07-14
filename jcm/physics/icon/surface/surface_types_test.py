"""
Unit tests for surface types and data structures.
"""

import pytest
import jax.numpy as jnp
import numpy as np

from jcm.physics.icon.surface.surface_types import (
    SurfaceParameters, SurfaceState, AtmosphericForcing, 
    SurfaceFluxes, SurfaceTendencies, SurfaceDiagnostics, SurfaceResistances
)


class TestSurfaceParameters:
    """Test SurfaceParameters data structure."""
    
    def test_default_parameters(self):
        """Test default parameter values."""
        params = SurfaceParameters.default()
        
        # Surface types
        assert params.nsfc_type == 3
        assert params.iwtr == 0
        assert params.iice == 1
        assert params.ilnd == 2
        
        # Physical constants
        assert params.ml_depth == 50.0
        assert params.rho_water == 1025.0
        assert params.cp_water == 3994.0
        assert params.von_karman == 0.4
        assert params.min_wind_speed == 1.0
        
        # Roughness lengths
        assert params.z0_water == 1e-4
        assert params.z0_ice == 1e-3
        assert params.z0_land == 0.1
    
    def test_custom_parameters(self):
        """Test custom parameter values."""
        params = SurfaceParameters.default(
            nsfc_type=2,
            ml_depth=100.0,
            von_karman=0.35
        )
        
        assert params.nsfc_type == 2
        assert params.ml_depth == 100.0
        assert params.von_karman == 0.35
        # Other values should remain default
        assert params.rho_water == 1025.0


class TestSurfaceState:
    """Test SurfaceState data structure."""
    
    def test_surface_state_creation(self):
        """Test creation of surface state."""
        ncol, nsfc_type = 5, 3
        nice_layers, nsoil_layers = 2, 4
        
        state = SurfaceState(
            temperature=jnp.ones((ncol, nsfc_type)) * 280.0,
            temperature_rad=jnp.ones(ncol) * 280.0,
            fraction=jnp.ones((ncol, nsfc_type)) / nsfc_type,
            ocean_temp=jnp.ones(ncol) * 285.0,
            ocean_u=jnp.zeros(ncol),
            ocean_v=jnp.zeros(ncol),
            ice_thickness=jnp.ones((ncol, nice_layers)) * 2.0,
            ice_temp=jnp.ones((ncol, nice_layers)) * 270.0,
            snow_depth=jnp.ones(ncol) * 0.1,
            soil_temp=jnp.ones((ncol, nsoil_layers)) * 280.0,
            soil_moisture=jnp.ones((ncol, nsoil_layers)) * 0.3,
            vegetation_temp=jnp.ones(ncol) * 280.0,
            roughness_momentum=jnp.ones((ncol, nsfc_type)) * 0.01,
            roughness_heat=jnp.ones((ncol, nsfc_type)) * 0.001,
            albedo_visible_direct=jnp.ones((ncol, nsfc_type)) * 0.1,
            albedo_visible_diffuse=jnp.ones((ncol, nsfc_type)) * 0.1,
            albedo_nir_direct=jnp.ones((ncol, nsfc_type)) * 0.2,
            albedo_nir_diffuse=jnp.ones((ncol, nsfc_type)) * 0.2
        )
        
        assert state.temperature.shape == (ncol, nsfc_type)
        assert state.temperature_rad.shape == (ncol,)
        assert state.fraction.shape == (ncol, nsfc_type)
        assert state.ocean_temp.shape == (ncol,)
        assert state.ice_thickness.shape == (ncol, nice_layers)
        assert state.soil_temp.shape == (ncol, nsoil_layers)
        
        # Check values
        assert jnp.allclose(state.temperature, 280.0)
        assert jnp.allclose(state.ocean_temp, 285.0)
        assert jnp.allclose(state.ice_temp, 270.0)
    
    def test_surface_state_modification(self):
        """Test modification of surface state."""
        ncol, nsfc_type = 3, 3
        nice_layers, nsoil_layers = 2, 4
        
        state = SurfaceState(
            temperature=jnp.ones((ncol, nsfc_type)) * 280.0,
            temperature_rad=jnp.ones(ncol) * 280.0,
            fraction=jnp.ones((ncol, nsfc_type)) / nsfc_type,
            ocean_temp=jnp.ones(ncol) * 285.0,
            ocean_u=jnp.zeros(ncol),
            ocean_v=jnp.zeros(ncol),
            ice_thickness=jnp.ones((ncol, nice_layers)) * 2.0,
            ice_temp=jnp.ones((ncol, nice_layers)) * 270.0,
            snow_depth=jnp.ones(ncol) * 0.1,
            soil_temp=jnp.ones((ncol, nsoil_layers)) * 280.0,
            soil_moisture=jnp.ones((ncol, nsoil_layers)) * 0.3,
            vegetation_temp=jnp.ones(ncol) * 280.0,
            roughness_momentum=jnp.ones((ncol, nsfc_type)) * 0.01,
            roughness_heat=jnp.ones((ncol, nsfc_type)) * 0.001,
            albedo_visible_direct=jnp.ones((ncol, nsfc_type)) * 0.1,
            albedo_visible_diffuse=jnp.ones((ncol, nsfc_type)) * 0.1,
            albedo_nir_direct=jnp.ones((ncol, nsfc_type)) * 0.2,
            albedo_nir_diffuse=jnp.ones((ncol, nsfc_type)) * 0.2
        )
        
        # Modify ocean temperature
        new_ocean_temp = state.ocean_temp + 5.0
        new_state = state._replace(ocean_temp=new_ocean_temp)
        
        assert jnp.allclose(new_state.ocean_temp, 290.0)
        assert jnp.allclose(new_state.temperature, 280.0)  # Other fields unchanged


class TestAtmosphericForcing:
    """Test AtmosphericForcing data structure."""
    
    def test_atmospheric_forcing_creation(self):
        """Test creation of atmospheric forcing."""
        ncol, nsfc_type = 4, 3
        
        forcing = AtmosphericForcing(
            temperature=jnp.ones(ncol) * 290.0,
            humidity=jnp.ones(ncol) * 0.01,
            u_wind=jnp.ones(ncol) * 5.0,
            v_wind=jnp.ones(ncol) * 3.0,
            pressure=jnp.ones(ncol) * 101325.0,
            sw_downward=jnp.ones(ncol) * 300.0,
            lw_downward=jnp.ones(ncol) * 350.0,
            rain_rate=jnp.ones(ncol) * 1e-6,
            snow_rate=jnp.ones(ncol) * 1e-7,
            exchange_coeff_heat=jnp.ones((ncol, nsfc_type)) * 0.01,
            exchange_coeff_moisture=jnp.ones((ncol, nsfc_type)) * 0.01,
            exchange_coeff_momentum=jnp.ones((ncol, nsfc_type)) * 0.01
        )
        
        assert forcing.temperature.shape == (ncol,)
        assert forcing.humidity.shape == (ncol,)
        assert forcing.u_wind.shape == (ncol,)
        assert forcing.pressure.shape == (ncol,)
        assert forcing.exchange_coeff_heat.shape == (ncol, nsfc_type)
        
        # Check values
        assert jnp.allclose(forcing.temperature, 290.0)
        assert jnp.allclose(forcing.pressure, 101325.0)
        assert jnp.allclose(forcing.sw_downward, 300.0)


class TestSurfaceFluxes:
    """Test SurfaceFluxes data structure."""
    
    def test_surface_fluxes_creation(self):
        """Test creation of surface fluxes."""
        ncol, nsfc_type = 3, 3
        
        fluxes = SurfaceFluxes(
            sensible_heat=jnp.ones((ncol, nsfc_type)) * 50.0,
            latent_heat=jnp.ones((ncol, nsfc_type)) * 100.0,
            longwave_net=jnp.ones((ncol, nsfc_type)) * -80.0,
            shortwave_net=jnp.ones((ncol, nsfc_type)) * 200.0,
            ground_heat=jnp.ones((ncol, nsfc_type)) * 10.0,
            momentum_u=jnp.ones((ncol, nsfc_type)) * 0.1,
            momentum_v=jnp.ones((ncol, nsfc_type)) * 0.05,
            evaporation=jnp.ones((ncol, nsfc_type)) * 1e-6,
            transpiration=jnp.ones((ncol, nsfc_type)) * 2e-6,
            sensible_heat_mean=jnp.ones(ncol) * 50.0,
            latent_heat_mean=jnp.ones(ncol) * 100.0,
            momentum_u_mean=jnp.ones(ncol) * 0.1,
            momentum_v_mean=jnp.ones(ncol) * 0.05,
            evaporation_mean=jnp.ones(ncol) * 3e-6
        )
        
        assert fluxes.sensible_heat.shape == (ncol, nsfc_type)
        assert fluxes.latent_heat.shape == (ncol, nsfc_type)
        assert fluxes.sensible_heat_mean.shape == (ncol,)
        assert fluxes.evaporation_mean.shape == (ncol,)
        
        # Check values
        assert jnp.allclose(fluxes.sensible_heat, 50.0)
        assert jnp.allclose(fluxes.latent_heat, 100.0)
        assert jnp.allclose(fluxes.longwave_net, -80.0)
    
    def test_flux_consistency(self):
        """Test consistency between tile and mean fluxes."""
        ncol, nsfc_type = 2, 3
        
        # Create tile fluxes
        sensible_tile = jnp.array([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]])
        fractions = jnp.array([[0.3, 0.4, 0.3], [0.2, 0.5, 0.3]])
        
        # Expected mean
        expected_mean = jnp.sum(sensible_tile * fractions, axis=1)
        
        fluxes = SurfaceFluxes(
            sensible_heat=sensible_tile,
            latent_heat=jnp.zeros((ncol, nsfc_type)),
            longwave_net=jnp.zeros((ncol, nsfc_type)),
            shortwave_net=jnp.zeros((ncol, nsfc_type)),
            ground_heat=jnp.zeros((ncol, nsfc_type)),
            momentum_u=jnp.zeros((ncol, nsfc_type)),
            momentum_v=jnp.zeros((ncol, nsfc_type)),
            evaporation=jnp.zeros((ncol, nsfc_type)),
            transpiration=jnp.zeros((ncol, nsfc_type)),
            sensible_heat_mean=expected_mean,
            latent_heat_mean=jnp.zeros(ncol),
            momentum_u_mean=jnp.zeros(ncol),
            momentum_v_mean=jnp.zeros(ncol),
            evaporation_mean=jnp.zeros(ncol)
        )
        
        # Verify consistency
        computed_mean = jnp.sum(fluxes.sensible_heat * fractions, axis=1)
        assert jnp.allclose(computed_mean, fluxes.sensible_heat_mean)


class TestSurfaceTendencies:
    """Test SurfaceTendencies data structure."""
    
    def test_surface_tendencies_creation(self):
        """Test creation of surface tendencies."""
        ncol, nsfc_type = 3, 3
        nice_layers, nsoil_layers = 2, 4
        
        tendencies = SurfaceTendencies(
            surface_temp_tendency=jnp.ones((ncol, nsfc_type)) * 0.01,
            ocean_temp_tendency=jnp.ones(ncol) * 0.005,
            ice_temp_tendency=jnp.ones((ncol, nice_layers)) * -0.01,
            soil_temp_tendency=jnp.ones((ncol, nsoil_layers)) * 0.002,
            ice_thickness_tendency=jnp.ones((ncol, nice_layers)) * 1e-8,
            snow_depth_tendency=jnp.ones(ncol) * 1e-7,
            soil_moisture_tendency=jnp.ones((ncol, nsoil_layers)) * 1e-9
        )
        
        assert tendencies.surface_temp_tendency.shape == (ncol, nsfc_type)
        assert tendencies.ocean_temp_tendency.shape == (ncol,)
        assert tendencies.ice_temp_tendency.shape == (ncol, nice_layers)
        assert tendencies.soil_temp_tendency.shape == (ncol, nsoil_layers)
        
        # Check values
        assert jnp.allclose(tendencies.surface_temp_tendency, 0.01)
        assert jnp.allclose(tendencies.ocean_temp_tendency, 0.005)
        assert jnp.allclose(tendencies.ice_temp_tendency, -0.01)


class TestSurfaceDiagnostics:
    """Test SurfaceDiagnostics data structure."""
    
    def test_surface_diagnostics_creation(self):
        """Test creation of surface diagnostics."""
        ncol, nsfc_type = 4, 3
        
        diagnostics = SurfaceDiagnostics(
            temperature_2m=jnp.ones(ncol) * 285.0,
            humidity_2m=jnp.ones(ncol) * 0.008,
            dewpoint_2m=jnp.ones(ncol) * 280.0,
            wind_speed_10m=jnp.ones(ncol) * 6.0,
            u_wind_10m=jnp.ones(ncol) * 4.0,
            v_wind_10m=jnp.ones(ncol) * 3.0,
            friction_velocity=jnp.ones(ncol) * 0.3,
            richardson_number=jnp.ones(ncol) * 0.1,
            surface_layer_height=jnp.ones(ncol) * 100.0,
            net_radiation=jnp.ones(ncol) * 150.0,
            radiation_balance=jnp.ones(ncol) * 150.0,
            energy_balance_residual=jnp.ones(ncol) * 5.0,
            temperature_2m_tile=jnp.ones((ncol, nsfc_type)) * 285.0,
            humidity_2m_tile=jnp.ones((ncol, nsfc_type)) * 0.008,
            wind_speed_10m_tile=jnp.ones((ncol, nsfc_type)) * 6.0
        )
        
        assert diagnostics.temperature_2m.shape == (ncol,)
        assert diagnostics.humidity_2m.shape == (ncol,)
        assert diagnostics.temperature_2m_tile.shape == (ncol, nsfc_type)
        assert diagnostics.wind_speed_10m_tile.shape == (ncol, nsfc_type)
        
        # Check values
        assert jnp.allclose(diagnostics.temperature_2m, 285.0)
        assert jnp.allclose(diagnostics.wind_speed_10m, 6.0)
        assert jnp.allclose(diagnostics.net_radiation, 150.0)


class TestSurfaceResistances:
    """Test SurfaceResistances data structure."""
    
    def test_surface_resistances_creation(self):
        """Test creation of surface resistances."""
        ncol, nsfc_type = 3, 3
        
        resistances = SurfaceResistances(
            aerodynamic_heat=jnp.ones((ncol, nsfc_type)) * 50.0,
            aerodynamic_moisture=jnp.ones((ncol, nsfc_type)) * 60.0,
            aerodynamic_momentum=jnp.ones((ncol, nsfc_type)) * 40.0,
            surface_moisture=jnp.ones((ncol, nsfc_type)) * 100.0,
            canopy_resistance=jnp.ones(ncol) * 150.0,
            soil_resistance=jnp.ones(ncol) * 200.0,
            stability_heat=jnp.ones((ncol, nsfc_type)) * 1.2,
            stability_momentum=jnp.ones((ncol, nsfc_type)) * 1.1
        )
        
        assert resistances.aerodynamic_heat.shape == (ncol, nsfc_type)
        assert resistances.aerodynamic_moisture.shape == (ncol, nsfc_type)
        assert resistances.canopy_resistance.shape == (ncol,)
        assert resistances.soil_resistance.shape == (ncol,)
        assert resistances.stability_heat.shape == (ncol, nsfc_type)
        
        # Check values
        assert jnp.allclose(resistances.aerodynamic_heat, 50.0)
        assert jnp.allclose(resistances.canopy_resistance, 150.0)
        assert jnp.allclose(resistances.stability_heat, 1.2)


if __name__ == "__main__":
    pytest.main([__file__])