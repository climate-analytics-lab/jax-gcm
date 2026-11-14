"""Unit tests for ocean surface diagnostic fluxes."""

import jax.numpy as jnp
import numpy as np

import jcm.constants as c
from jcm.physics.surface.echam.ocean import (
    CHARNOCK_ALPHA_DEFAULT, Z0_OCEAN_MIN, compute_ocean_albedo,
    compute_ocean_roughness, compute_ocean_surface_fluxes, ocean_physics_step,
)
from jcm.physics.surface.echam.surface_types import (
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
    """Test ocean roughness calculation.

    ``compute_ocean_roughness`` takes the surface-layer FRICTION VELOCITY
    directly (not a raw wind speed) — see its docstring for the derivation
    from ECHAM's ``update_z0_ocean`` (``mo_surface_ocean.f90``). It is the
    shared formula for both the live water-tile roughness
    (``TteTkeVerticalDiffusion``, ``VDiffParameters.cchar``) and this
    module's own bulk-flux reference path (jax-gcm#1118).
    """

    def test_ocean_roughness_basic(self):
        """Test basic ocean roughness calculation."""
        ncol = 3
        friction_velocity = jnp.array([0.2, 0.4, 0.6])

        roughness = compute_ocean_roughness(friction_velocity)

        assert roughness.shape == (ncol,)
        assert jnp.all(jnp.isfinite(roughness))
        assert jnp.all(roughness > 0.0)

    def test_ocean_roughness_wind_dependence(self):
        """Test ocean roughness dependence on friction velocity."""
        friction_velocity = jnp.array([0.05, 0.2, 0.4])

        roughness = compute_ocean_roughness(friction_velocity)

        # Roughness should increase with u* (Charnock relation)
        assert roughness[0] < roughness[1] < roughness[2]

    def test_ocean_roughness_no_upper_cap(self):
        """ECHAM's ``update_z0_ocean`` has only a floor, no ceiling: a very
        high u* (an extreme-wind event) must keep growing, not saturate at
        the old, since-removed 10 cm cap.
        """
        moderate = compute_ocean_roughness(jnp.array([0.3]))
        extreme = compute_ocean_roughness(jnp.array([3.0]))
        # u* is 10x; z0 ∝ u*² so it should grow ~100x, not clip to a shared
        # ceiling.
        np.testing.assert_allclose(
            np.asarray(extreme) / np.asarray(moderate), 100.0, rtol=1e-6)

    def test_ocean_roughness_floor(self):
        """Calm conditions floor at ECHAM's ``zepzzo`` (1.5e-5 m)."""
        roughness = compute_ocean_roughness(jnp.array([0.001]))
        assert roughness[0] == Z0_OCEAN_MIN

    def test_ocean_roughness_zero_friction_velocity(self):
        """Zero u* floors at ``Z0_OCEAN_MIN`` rather than collapsing to zero."""
        roughness_zero = compute_ocean_roughness(jnp.array([0.0]))
        assert jnp.all(jnp.isfinite(roughness_zero))
        assert roughness_zero[0] == Z0_OCEAN_MIN

    def test_ocean_roughness_cchar_override(self):
        """A non-default ``cchar`` (e.g. ``VDiffParameters.cchar``) scales linearly."""
        u_star = jnp.array([0.5])
        base = compute_ocean_roughness(u_star)
        doubled = compute_ocean_roughness(u_star, cchar=2.0 * CHARNOCK_ALPHA_DEFAULT)
        np.testing.assert_allclose(np.asarray(doubled), 2.0 * np.asarray(base), rtol=1e-6)

    def test_gradients_through_cchar_and_friction_velocity(self):
        """``cchar`` (reintroduced as ``VDiffParameters.cchar``, jax-gcm#1118)
        is a genuine differentiable leaf of this formula, not an
        observationally dead tuning literal (the #999/#1123 concern this
        issue resolves): ``d(z0)/d(cchar) = u*²/g`` and ``d(z0)/d(u*) =
        2·cchar·u*/g``, both away from the floor.
        """
        from jcm.testing import check_gradients

        def f(cchar, friction_velocity):
            return compute_ocean_roughness(friction_velocity, cchar=cchar)

        check_gradients(
            f, (jnp.asarray(CHARNOCK_ALPHA_DEFAULT), jnp.array([0.3, 0.5])),
            rtol=1e-3,
        )

    def test_charnock_alpha_default_matches_the_formula(self):
        """``CHARNOCK_ALPHA_DEFAULT`` (jax-gcm#999/#1118) is the one this uses.

        Unfloored u* (the floor would hide a wrong coefficient), so the
        formula itself is pinned, not the bound.
        """
        friction_velocity = jnp.array([0.3, 0.5])
        roughness = compute_ocean_roughness(friction_velocity)
        expected = CHARNOCK_ALPHA_DEFAULT * friction_velocity**2 / float(c.grav)
        np.testing.assert_allclose(np.asarray(roughness), np.asarray(expected),
                                   rtol=1e-6)
        assert CHARNOCK_ALPHA_DEFAULT == 0.018


class TestOceanSurfaceFluxes:
    """Test ocean surface flux calculations."""

    def setup_method(self):
        """Set up test data."""
        self.ncol = 3

        self.atmospheric_state = AtmosphericForcing(
            reference_height=jnp.full(self.ncol, 60.0),
            heat_2m_reduction=jnp.full((self.ncol, 3), 0.7),
            temperature=jnp.array([290.0, 285.0, 295.0]),
            humidity=jnp.array([0.01, 0.008, 0.012]),
            u_wind=jnp.array([5.0, 3.0, 8.0]),
            v_wind=jnp.array([2.0, 4.0, 1.0]),
            pressure=jnp.array([101325.0, 95000.0, 85000.0]),
            surface_pressure=jnp.array([101325.0, 95000.0, 85000.0]),
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

    def test_saturation_is_the_ua_table_at_the_surface_pressure(self):
        """``precalc_ocean``: ``ua`` at the SST over ``paphm1``.

        The evaporation is ``ρ·C_q·(qs − q)`` with ECHAM's ``qs`` of the
        ``ua`` table (ice at and below tmelt, so a 271 K sea reads the ice
        fit) at the SURFACE pressure, not the lowest level's.
        """
        import jax
        import jcm.constants as c
        from jcm.physics import thermodynamics

        with jax.enable_x64(True):
            sst = jnp.array([285.0, 271.0, c.tmelt])
            state = self.atmospheric_state._replace(
                pressure=jnp.array([99000.0, 94000.0, 84000.0]),
                surface_pressure=jnp.array([101325.0, 95000.0, 85000.0]))
            fluxes, _ = compute_ocean_surface_fluxes(
                state, sst, self.ocean_u, self.ocean_v,
                self.exchange_coeff_heat, self.exchange_coeff_moisture,
                self.exchange_coeff_momentum, self.solar_zenith_angle)
            es = jnp.where(sst <= c.tmelt, thermodynamics.es_ice(sst),
                           thermodynamics.es_water(sst))
            x = es * c.rd / c.rv / state.surface_pressure
            qs = x / (1.0 - c.vtmpc1 * x)
            rho = state.pressure / (c.rd * state.temperature)
            want = rho * self.exchange_coeff_moisture * (qs - state.humidity)
            assert jnp.allclose(fluxes.evaporation_mean, want, rtol=1e-12,
                                atol=0.0)

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
            reference_height=jnp.full(self.ncol, 60.0),
            heat_2m_reduction=jnp.full((self.ncol, 3), 0.7),
            temperature=jnp.array([290.0, 285.0]),
            humidity=jnp.array([0.01, 0.008]),
            u_wind=jnp.array([5.0, 3.0]),
            v_wind=jnp.array([2.0, 4.0]),
            pressure=jnp.array([101325.0, 95000.0]),
            surface_pressure=jnp.array([101325.0, 95000.0]),
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
