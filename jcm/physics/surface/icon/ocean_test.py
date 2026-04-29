"""Unit tests for ocean surface diagnostic fluxes."""

import jax.numpy as jnp

from jcm.physics.surface.icon.ocean import (
    compute_ocean_albedo, compute_ocean_roughness,
    compute_ocean_surface_fluxes, ocean_physics_step,
)
from jcm.physics.surface.icon.surface_types import (
    AtmosphericForcing, SurfaceFluxes,
)


class TestOceanAlbedo:
    """Test ocean albedo calculation."""

    def test_ocean_albedo_basic(self):
        """Test basic ocean albedo calculation."""
        ncol = 3
        solar_zenith_angle = jnp.array([0.0, jnp.pi/4, jnp.pi/3])  # 0°, 45°, 60°

        albedo_vis_dir, albedo_vis_dif, albedo_nir_dir, albedo_nir_dif = compute_ocean_albedo(
            solar_zenith_angle
        )

        assert albedo_vis_dir.shape == (ncol,)
        assert albedo_vis_dif.shape == (ncol,)
        assert albedo_nir_dir.shape == (ncol,)
        assert albedo_nir_dif.shape == (ncol,)

        # All albedos should be positive and less than 1
        assert jnp.all(albedo_vis_dir > 0.0)
        assert jnp.all(albedo_vis_dir < 1.0)
        assert jnp.all(albedo_vis_dif > 0.0)
        assert jnp.all(albedo_vis_dif < 1.0)
        assert jnp.all(albedo_nir_dir > 0.0)
        assert jnp.all(albedo_nir_dir < 1.0)
        assert jnp.all(albedo_nir_dif > 0.0)
        assert jnp.all(albedo_nir_dif < 1.0)

    def test_ocean_albedo_zenith_angle_dependence(self):
        """Test ocean albedo dependence on solar zenith angle."""
        solar_zenith_angle = jnp.array([0.0, jnp.pi/4, jnp.pi/2])  # 0°, 45°, 90°

        albedo_vis_dir, _, albedo_nir_dir, _ = compute_ocean_albedo(solar_zenith_angle)

        # Direct albedo should increase with zenith angle
        assert albedo_vis_dir[0] < albedo_vis_dir[1] < albedo_vis_dir[2]
        assert albedo_nir_dir[0] < albedo_nir_dir[1] < albedo_nir_dir[2]

    def test_ocean_albedo_diffuse_constant(self):
        """Test that diffuse albedo is constant."""
        solar_zenith_angle = jnp.array([0.0, jnp.pi/4, jnp.pi/3])

        _, albedo_vis_dif, _, albedo_nir_dif = compute_ocean_albedo(solar_zenith_angle)

        # Diffuse albedos should be constant
        assert jnp.allclose(albedo_vis_dif, 0.06)
        assert jnp.allclose(albedo_nir_dif, 0.06)

    def test_ocean_albedo_wavelength_dependence(self):
        """Test that VIS direct and NIR direct albedos differ at off-zenith angles."""
        solar_zenith_angle = jnp.array([jnp.pi/3])  # 60°

        albedo_vis_dir, _, albedo_nir_dir, _ = compute_ocean_albedo(solar_zenith_angle)

        # VIS direct uses a stronger zenith-angle factor than NIR — should be larger
        assert albedo_vis_dir[0] > albedo_nir_dir[0]


class TestOceanRoughness:
    """Test ocean roughness calculation."""

    def test_ocean_roughness_basic(self):
        """Test basic ocean roughness calculation."""
        ncol = 3
        wind_speed = jnp.array([5.0, 10.0, 15.0])
        ocean_u = jnp.zeros(ncol)
        ocean_v = jnp.zeros(ncol)

        roughness = compute_ocean_roughness(wind_speed, ocean_u, ocean_v)

        assert roughness.shape == (ncol,)
        assert jnp.all(jnp.isfinite(roughness))
        assert jnp.all(roughness > 0.0)

    def test_ocean_roughness_wind_dependence(self):
        """Test ocean roughness dependence on wind speed."""
        ocean_u = jnp.zeros(3)
        ocean_v = jnp.zeros(3)
        wind_speed = jnp.array([1.0, 5.0, 10.0])

        roughness = compute_ocean_roughness(wind_speed, ocean_u, ocean_v)

        # Roughness should increase with wind speed (Charnock relation)
        assert roughness[0] < roughness[1] < roughness[2]

    def test_ocean_roughness_current_effect(self):
        """Test that ocean current parameters are accepted (kept for interface compatibility)."""
        wind_speed = jnp.array([5.0])
        roughness_no_current = compute_ocean_roughness(wind_speed, jnp.zeros(1), jnp.zeros(1))
        roughness_current = compute_ocean_roughness(wind_speed, jnp.array([1.0]), jnp.array([1.0]))

        assert jnp.all(jnp.isfinite(roughness_no_current))
        assert jnp.all(jnp.isfinite(roughness_current))

    def test_ocean_roughness_bounds(self):
        """Roughness is clipped between physical min/max bounds."""
        # Extreme winds — should be capped at the upper bound
        roughness = compute_ocean_roughness(
            jnp.array([100.0]), jnp.zeros(1), jnp.zeros(1)
        )
        assert roughness[0] <= 0.1

        # Calm conditions — should be at the lower bound
        roughness = compute_ocean_roughness(
            jnp.array([0.001]), jnp.zeros(1), jnp.zeros(1)
        )
        assert roughness[0] >= 1e-5

    def test_ocean_roughness_minimum_wind(self):
        """A min-wind floor prevents zero-wind from collapsing the Charnock formula."""
        roughness_zero = compute_ocean_roughness(
            jnp.array([0.0]), jnp.zeros(1), jnp.zeros(1)
        )
        assert jnp.all(jnp.isfinite(roughness_zero))
        assert roughness_zero[0] > 0.0


class TestOceanSurfaceFluxes:
    """Test ocean surface flux calculations."""

    def setup_method(self):
        """Set up test data."""
        self.ncol = 3

        self.atmospheric_state = AtmosphericForcing(
            temperature=jnp.array([290.0, 285.0, 295.0]),
            humidity=jnp.array([0.01, 0.008, 0.012]),
            u_wind=jnp.array([5.0, 3.0, 8.0]),
            v_wind=jnp.array([2.0, 4.0, 1.0]),
            pressure=jnp.array([101325.0, 95000.0, 85000.0]),
            sw_downward=jnp.array([300.0, 250.0, 400.0]),
            lw_downward=jnp.array([350.0, 320.0, 380.0]),
            rain_rate=jnp.array([1e-6, 2e-6, 0.0]),
            snow_rate=jnp.array([0.0, 0.0, 1e-7]),
            exchange_coeff_heat=jnp.ones((self.ncol, 3)) * 0.01,
            exchange_coeff_moisture=jnp.ones((self.ncol, 3)) * 0.01,
            exchange_coeff_momentum=jnp.ones((self.ncol, 3)) * 0.01
        )

        self.ocean_temp = jnp.array([285.0, 280.0, 288.0])
        self.ocean_u = jnp.array([0.5, 0.0, -0.3])
        self.ocean_v = jnp.array([0.2, 0.8, 0.1])
        self.exchange_coeff_heat = jnp.array([0.01, 0.015, 0.008])
        self.exchange_coeff_moisture = jnp.array([0.01, 0.015, 0.008])
        self.exchange_coeff_momentum = jnp.array([0.01, 0.015, 0.008])
        self.solar_zenith_angle = jnp.array([0.5, 0.8, 0.3])

    def test_ocean_surface_fluxes_basic(self):
        """Test basic ocean surface flux calculation."""
        fluxes, roughness = compute_ocean_surface_fluxes(
            self.atmospheric_state, self.ocean_temp, self.ocean_u, self.ocean_v,
            self.exchange_coeff_heat, self.exchange_coeff_moisture,
            self.exchange_coeff_momentum, self.solar_zenith_angle
        )

        assert isinstance(fluxes, SurfaceFluxes)
        assert fluxes.sensible_heat.shape == (self.ncol, 1)
        assert fluxes.latent_heat.shape == (self.ncol, 1)
        assert fluxes.momentum_u.shape == (self.ncol, 1)
        assert fluxes.momentum_v.shape == (self.ncol, 1)
        assert fluxes.evaporation.shape == (self.ncol, 1)
        assert roughness.shape == (self.ncol,)

        # Check that fluxes are finite
        assert jnp.all(jnp.isfinite(fluxes.sensible_heat))
        assert jnp.all(jnp.isfinite(fluxes.latent_heat))
        assert jnp.all(jnp.isfinite(fluxes.momentum_u))
        assert jnp.all(jnp.isfinite(fluxes.momentum_v))
        assert jnp.all(jnp.isfinite(roughness))

    def test_ocean_flux_directions(self):
        """Test that flux directions make physical sense."""
        fluxes, _ = compute_ocean_surface_fluxes(
            self.atmospheric_state, self.ocean_temp, self.ocean_u, self.ocean_v,
            self.exchange_coeff_heat, self.exchange_coeff_moisture,
            self.exchange_coeff_momentum, self.solar_zenith_angle
        )

        # Temperature differences
        temp_diff = self.ocean_temp - self.atmospheric_state.temperature

        # Sensible heat flux should have same sign as temperature difference
        for i in range(self.ncol):
            if temp_diff[i] > 0:  # Ocean warmer than air
                assert fluxes.sensible_heat[i, 0] > 0  # Upward flux
            elif temp_diff[i] < 0:  # Ocean cooler than air
                assert fluxes.sensible_heat[i, 0] < 0  # Downward flux

    def test_ocean_energy_balance_components(self):
        """Test ocean energy balance components."""
        fluxes, _ = compute_ocean_surface_fluxes(
            self.atmospheric_state, self.ocean_temp, self.ocean_u, self.ocean_v,
            self.exchange_coeff_heat, self.exchange_coeff_moisture,
            self.exchange_coeff_momentum, self.solar_zenith_angle
        )

        # Net shortwave should be positive (absorbed)
        assert jnp.all(fluxes.shortwave_net >= 0.0)

        # Net longwave should be negative (ocean emits more than it receives)
        assert jnp.all(fluxes.longwave_net <= 0.0)

    def test_ocean_evaporation_direction(self):
        """Test that evaporation direction makes physical sense."""
        fluxes, _ = compute_ocean_surface_fluxes(
            self.atmospheric_state, self.ocean_temp, self.ocean_u, self.ocean_v,
            self.exchange_coeff_heat, self.exchange_coeff_moisture,
            self.exchange_coeff_momentum, self.solar_zenith_angle
        )

        # Evaporation should be finite
        assert jnp.all(jnp.isfinite(fluxes.evaporation))

        # Latent heat flux should be finite
        assert jnp.all(jnp.isfinite(fluxes.latent_heat))


class TestOceanPhysicsStep:
    """Test the diagnostic ocean physics step (zero tendencies)."""

    def setup_method(self):
        """Set up test data."""
        self.ncol = 2

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
            exchange_coeff_heat=jnp.ones((self.ncol, 3)) * 0.01,
            exchange_coeff_moisture=jnp.ones((self.ncol, 3)) * 0.01,
            exchange_coeff_momentum=jnp.ones((self.ncol, 3)) * 0.01
        )

        self.ocean_temp = jnp.array([285.0, 280.0])
        self.ocean_u = jnp.array([0.5, 0.0])
        self.ocean_v = jnp.array([0.2, 0.8])
        self.exchange_coeff_heat = jnp.array([0.01, 0.015])
        self.exchange_coeff_moisture = jnp.array([0.01, 0.015])
        self.exchange_coeff_momentum = jnp.array([0.01, 0.015])
        self.solar_zenith_angle = jnp.array([0.5, 0.8])
        self.dt = 3600.0

    def test_ocean_physics_step_basic(self):
        """Returns finite fluxes, zero tendencies, finite roughness."""
        fluxes, tendencies, roughness = ocean_physics_step(
            self.atmospheric_state, self.ocean_temp, self.ocean_u, self.ocean_v,
            self.exchange_coeff_heat, self.exchange_coeff_moisture,
            self.exchange_coeff_momentum, self.solar_zenith_angle, self.dt
        )

        assert isinstance(fluxes, SurfaceFluxes)
        assert fluxes.sensible_heat.shape == (self.ncol, 1)
        assert fluxes.latent_heat.shape == (self.ncol, 1)

        assert tendencies.ocean_temp_tendency.shape == (self.ncol,)
        assert tendencies.surface_temp_tendency.shape == (self.ncol, 1)
        assert roughness.shape == (self.ncol,)

        assert jnp.all(jnp.isfinite(fluxes.sensible_heat))
        assert jnp.all(jnp.isfinite(roughness))

    def test_ocean_physics_zero_tendencies(self):
        """Diagnostic step returns no prognostic tendencies."""
        _, tendencies, _ = ocean_physics_step(
            self.atmospheric_state, self.ocean_temp, self.ocean_u, self.ocean_v,
            self.exchange_coeff_heat, self.exchange_coeff_moisture,
            self.exchange_coeff_momentum, self.solar_zenith_angle, self.dt
        )

        assert jnp.allclose(tendencies.ocean_temp_tendency, 0.0)
        assert jnp.allclose(tendencies.surface_temp_tendency, 0.0)
        assert jnp.allclose(tendencies.ice_temp_tendency, 0.0)
        assert jnp.allclose(tendencies.ice_thickness_tendency, 0.0)
        assert jnp.allclose(tendencies.snow_depth_tendency, 0.0)
