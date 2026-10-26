"""Unit tests for main surface physics interface."""

import jax
import pytest
import jax.numpy as jnp

from jcm.physics.surface.echam.surface_physics import (
    initialize_surface_state, surface_physics_step,
    combine_surface_fluxes
)
from jcm.physics.surface.echam.surface_types import (
    SurfaceParameters, SurfaceState, AtmosphericForcing,
    SurfaceFluxes, SurfaceTendencies
)
from jcm.testing import check_gradients


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
        params = SurfaceParameters.default()
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
        params = SurfaceParameters.default()
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
        
        params = SurfaceParameters.default()
        
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
            reference_height=jnp.full(self.ncol, 60.0),
            heat_2m_reduction=jnp.full((self.ncol, self.nsfc_type), 0.7),
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
            self.atmospheric_state, self.surface_state, self.dt,
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
            self.atmospheric_state, self.surface_state, self.dt,
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
            self.atmospheric_state, self.surface_state, self.dt,
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

    def test_surface_physics_step_uses_passed_exchange_coeffs(self):
        """The flux is built from the passed-in exchange coefficients.

        Regression for the surface-wiring fix: ``surface_physics_step`` must
        consume the exchange velocities (CH·|U|, CE·|U|, CM·|U|) computed by the
        surface-layer scheme upstream (threaded in via ``AtmosphericForcing``),
        not recompute its own bulk-Richardson scheme. The momentum stress is
        ``rho * CM*|U| * u`` per tile and has no thermodynamic feedback, so
        doubling the passed momentum coefficient doubles the grid-box-mean
        momentum stress exactly.
        """
        base, _, _ = surface_physics_step(
            self.atmospheric_state, self.surface_state, self.dt,
        )
        doubled = self.atmospheric_state._replace(
            exchange_coeff_momentum=self.atmospheric_state.exchange_coeff_momentum * 2.0
        )
        scaled, _, _ = surface_physics_step(doubled, self.surface_state, self.dt)

        assert jnp.allclose(scaled.momentum_u_mean, 2.0 * base.momentum_u_mean, rtol=1e-5)
        assert jnp.allclose(scaled.momentum_v_mean, 2.0 * base.momentum_v_mean, rtol=1e-5)

    def test_surface_physics_step_honours_passed_parameters(self):
        """Parameters passed positionally after ``dt`` reach the tiles.

        ``EchamSurface`` calls ``surface_physics_step(atm, surface, dt,
        params)``, so the fourth positional argument is the parameter set.
        Every tile's upward longwave is ``emissivity * sigma * T**4``, so a
        lower emissivity must change the net longwave; a step that fell back
        to the default parameters would return the default flux.
        """
        default, _, _ = surface_physics_step(
            self.atmospheric_state, self.surface_state, self.dt,
        )
        grey_surface = SurfaceParameters.default(emissivity=0.5)
        positional, _, _ = surface_physics_step(
            self.atmospheric_state, self.surface_state, self.dt, grey_surface,
        )
        keyword, _, _ = surface_physics_step(
            self.atmospheric_state, self.surface_state, self.dt,
            params=grey_surface,
        )

        assert jnp.allclose(positional.longwave_net, keyword.longwave_net)
        assert jnp.all(
            jnp.abs(positional.longwave_net - default.longwave_net) > 1.0)


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


class TestSurfacePhysicsStepGradients:
    """Whole-scheme gradients through ``surface_physics_step`` (#820).

    ``EchamSurface`` is in the default ECHAM stack, so this is the surface
    path a run differentiates through: ocean, sea-ice and land tiles, their
    combination into grid-box means, and the diagnostics.

    Green, against a central difference, for a single column and for a
    four-column block. The two finiteness tests below carry the substance:
    the module's wind norms are cone tips at calm wind, where no two-sided
    derivative exists, and a run reaches them (an initialised-at-rest spin-up starts
    there).
    """

    @staticmethod
    def _forcing(ncol, u_wind, v_wind):
        """Lowest-level atmospheric forcing for ``ncol`` columns."""
        nsfc_type = 3
        return AtmosphericForcing(
            reference_height=jnp.full(ncol, 60.0),
            heat_2m_reduction=jnp.full((ncol, 3), 0.7),
            temperature=jnp.linspace(290.0, 284.0, ncol),
            humidity=jnp.linspace(0.010, 0.007, ncol),
            u_wind=u_wind,
            v_wind=v_wind,
            pressure=jnp.linspace(101325.0, 97000.0, ncol),
            surface_pressure=jnp.linspace(101325.0, 97000.0, ncol),
            sw_downward=jnp.linspace(300.0, 240.0, ncol),
            lw_downward=jnp.linspace(350.0, 315.0, ncol),
            rain_rate=jnp.full(ncol, 1.0e-6),
            snow_rate=jnp.zeros(ncol),
            exchange_coeff_heat=jnp.full((ncol, nsfc_type), 0.011),
            exchange_coeff_moisture=jnp.full((ncol, nsfc_type), 0.010),
            exchange_coeff_momentum=jnp.full((ncol, nsfc_type), 0.013),
        )

    @staticmethod
    def _state(ncol, ocean_temp=None, land_temp=283.0):
        """Build a mixed water/ice/land surface for ``ncol`` columns."""
        fraction = jnp.stack([jnp.full(ncol, 0.5), jnp.full(ncol, 0.2),
                              jnp.full(ncol, 0.3)], axis=1)
        if ocean_temp is None:
            ocean_temp = jnp.linspace(291.5, 285.5, ncol)
        return initialize_surface_state(
            ncol, fraction, ocean_temp, jnp.full((ncol, 2), 268.0),
            jnp.full((ncol, 4), land_temp))

    def _step(self, ncol, atmospheric_state, surface_state):
        """Return ``f(u, v, T, q, CH, CM, CE, T_ocean)``."""
        def f(u_wind, v_wind, temperature, humidity, exchange_heat,
              exchange_momentum, exchange_moisture, ocean_temp):
            state = atmospheric_state._replace(
                u_wind=u_wind, v_wind=v_wind, temperature=temperature,
                humidity=humidity, exchange_coeff_heat=exchange_heat,
                exchange_coeff_momentum=exchange_momentum,
                exchange_coeff_moisture=exchange_moisture)
            # The water tile's temperature *is* the ocean temperature, so
            # both have to move together or the check would compare a
            # derivative against a difference taken along a different
            # direction.
            surface = surface_state._replace(
                ocean_temp=ocean_temp,
                temperature=surface_state.temperature.at[:, 0].set(ocean_temp))
            return surface_physics_step(state, surface, 3600.0)

        return f

    @pytest.mark.parametrize("ncol", [1, 4], ids=["column", "block"])
    def test_step_gradients_match_a_central_difference(self, ncol):
        """A single column and a four-column block, both off the switches.

        The winds are O(5 m/s) — above ``min_wind_speed = 1``, so the
        ``jnp.maximum`` floors in ``compute_surface_resistances`` and the
        Charnock roughness are inactive — and the tile temperatures differ
        from the air temperature by a few kelvin, so no flux sits on a sign
        change.
        """
        # Float64 resolves the finite-difference subtraction across the full
        # mixed-unit output bundle; retain the strict AD/reference tolerance.
        # Float32 calm/degenerate-state gradient checks remain below.
        with jax.enable_x64(True):
            u_wind = jnp.linspace(6.0, 3.0, ncol)
            v_wind = jnp.linspace(2.5, 4.5, ncol)
            atmospheric_state = self._forcing(ncol, u_wind, v_wind)
            surface_state = self._state(ncol)
            args = (u_wind, v_wind, atmospheric_state.temperature,
                    atmospheric_state.humidity,
                    atmospheric_state.exchange_coeff_heat,
                    atmospheric_state.exchange_coeff_momentum,
                    atmospheric_state.exchange_coeff_moisture,
                    surface_state.ocean_temp)
            check_gradients(self._step(ncol, atmospheric_state, surface_state),
                            args, rtol=1e-3)

    def test_calm_wind_gradients_are_finite(self):
        """``u = v = 0``: every wind norm in the step is at its cone tip.

        ``surface_physics.py`` and ``turbulent_fluxes.py`` form
        ``sqrt(maximum(u^2 + v^2, 1e-30))`` for the wind speed, the momentum
        flux magnitude and the friction velocity. The gradient is finite because the floor is 1e-30 rather than 0: below it
        ``jnp.maximum`` routes the whole derivative to the constant branch,
        so ``sqrt'`` at the floor is multiplied by an exact zero instead of
        meeting a 0.5/0.5 tie and forming ``0 * inf``.

        Momentum is exactly zero here too, so ``:318``/``:322`` sit at their
        own tips at the same time.
        """
        ncol = 2
        calm = jnp.zeros(ncol)
        atmospheric_state = self._forcing(ncol, calm, calm)
        surface_state = self._state(ncol)
        step = self._step(ncol, atmospheric_state, surface_state)
        args = (calm, calm, atmospheric_state.temperature,
                atmospheric_state.humidity,
                atmospheric_state.exchange_coeff_heat,
                atmospheric_state.exchange_coeff_momentum,
                atmospheric_state.exchange_coeff_moisture,
                surface_state.ocean_temp)

        def total(*a):
            return sum(jnp.sum(leaf ** 2)
                       for leaf in jax.tree.leaves(step(*a)))

        gradients = jax.grad(total, argnums=tuple(range(len(args))))(*args)
        names = ("u_wind", "v_wind", "temperature", "humidity",
                 "exchange_coeff_heat", "exchange_coeff_momentum",
                 "exchange_coeff_moisture", "ocean_temp")
        for name, gradient in zip(names, gradients):
            assert jnp.all(jnp.isfinite(gradient)), (
                f"d/d{name} is not finite at calm wind: {gradient}")

    def test_zero_temperature_difference_gradients_are_finite(self):
        """Every tile at the air temperature: all three fluxes vanish at once.

        The sensible-heat, ground-heat and longwave terms are each linear in
        a temperature difference that is exactly zero here, and the land
        tile's ``jnp.maximum(net_radiation * 0.5, 0.0)`` and
        ``jnp.maximum(evaporation, 0.0)`` hinges (``land.py:241/402``) are
        the nearest switches. Finiteness is the assertion: these are kinks,
        so a central difference would report the mean of two one-sided
        slopes rather than anything AD computes.
        """
        ncol = 2
        u_wind = jnp.linspace(6.0, 3.0, ncol)
        v_wind = jnp.linspace(2.5, 4.5, ncol)
        atmospheric_state = self._forcing(ncol, u_wind, v_wind)
        air_temperature = atmospheric_state.temperature
        surface_state = self._state(ncol, ocean_temp=air_temperature)
        surface_state = surface_state._replace(
            temperature=jnp.broadcast_to(air_temperature[:, None], (ncol, 3)),
            soil_temp=jnp.broadcast_to(air_temperature[:, None], (ncol, 4)),
            ice_temp=jnp.broadcast_to(air_temperature[:, None], (ncol, 2)),
            vegetation_temp=air_temperature)
        step = self._step(ncol, atmospheric_state, surface_state)
        args = (u_wind, v_wind, air_temperature, atmospheric_state.humidity,
                atmospheric_state.exchange_coeff_heat,
                atmospheric_state.exchange_coeff_momentum,
                atmospheric_state.exchange_coeff_moisture,
                surface_state.ocean_temp)

        def total(*a):
            return sum(jnp.sum(leaf ** 2)
                       for leaf in jax.tree.leaves(step(*a)))

        gradients = jax.grad(total, argnums=tuple(range(len(args))))(*args)
        names = ("u_wind", "v_wind", "temperature", "humidity",
                 "exchange_coeff_heat", "exchange_coeff_momentum",
                 "exchange_coeff_moisture", "ocean_temp")
        for name, gradient in zip(names, gradients):
            assert jnp.all(jnp.isfinite(gradient)), (
                f"d/d{name} is not finite at zero temperature difference: "
                f"{gradient}")


class TestEchamSurfaceTerm:
    """Term-level contract of ``EchamSurface`` after the vdiff coupling.

    The surface exchange is now the bottom boundary row of the vdiff
    implicit solve (`TteTkeVerticalDiffusion`), so the term must (a) return
    ZERO u/v/T/qv tendencies — the old imp_* single-layer delivery block is
    gone — and (b) republish the vdiff-delivered fluxes as the public
    ``"surface"`` fields, with ``evaporation == effective_evaporation``.
    """

    def test_zero_tendencies_and_republished_vdiff_fluxes(self):
        from types import SimpleNamespace

        import numpy as np

        from jcm.forcing import ForcingData
        from jcm.physics.surface.echam.surface_physics import EchamSurface
        from jcm.physics.surface.echam.surface_types import SurfaceData
        from jcm.physics.vertical_diffusion.tte_tke.vertical_diffusion_types import (
            VerticalDiffusionData,
        )
        from jcm.physics_interface import PhysicsState
        from jcm.terrain import TerrainData

        nlev, ncols = 6, 1
        z_half = jnp.linspace(6000.0, 0.0, nlev + 1)[:, None]
        z_full = 0.5 * (z_half[:-1] + z_half[1:])
        p_half = jnp.linspace(5.0e4, 1.0e5, nlev + 1)[:, None]
        p_full = 0.5 * (p_half[:-1] + p_half[1:])

        state = PhysicsState(
            u_wind=jnp.full((nlev, ncols), 8.0),
            v_wind=jnp.zeros((nlev, ncols)),
            temperature=jnp.full((nlev, ncols), 290.0),
            specific_humidity=jnp.full((nlev, ncols), 0.008),
            geopotential=9.81 * z_full,
            normalized_surface_pressure=jnp.ones((ncols,)),
            tracers={},
        )

        # Seed the vdiff diagnostics with distinct delivered-flux values so
        # republication is unambiguous.
        vdiff = VerticalDiffusionData.zeros((ncols,), nlev).copy(
            surface_evaporation=jnp.array([3.0e-5]),
            surface_sensible_heat=jnp.array([12.0]),
            surface_latent_heat=jnp.array([75.0]),
            surface_stress_u=jnp.array([0.08]),
            surface_stress_v=jnp.array([-0.02]),
            surface_exchange_heat=jnp.full((ncols, 3), 0.01),
            surface_exchange_moisture=jnp.full((ncols, 3), 0.01),
            surface_exchange_momentum=jnp.full((ncols, 3), 0.012),
        )
        diagnostics = {
            "_dt_seconds": 900.0,
            "pressure_full": p_full,
            "pressure_half": p_half,
            "height_full": z_full,
            "height_half": z_half,
            "surface": SurfaceData.zeros((ncols,), nlev).copy(
                surface_temperature=jnp.array([292.0]),
                roughness_length=jnp.array([1e-4]),
            ),
            "vertical_diffusion": vdiff,
            "radiation": SimpleNamespace(
                surface_sw_down=jnp.zeros(ncols),
                surface_lw_down=jnp.full(ncols, 350.0),
            ),
        }
        terrain = TerrainData.single_column(fmask=0.0)
        forcing = ForcingData.zeros((1, 1)).copy(
            sea_surface_temperature=jnp.full((1, 1), 292.0),
        )

        term = EchamSurface()
        tend, out = term(state, diagnostics, forcing, terrain)

        # (a) Zero prognostic tendencies — delivery lives in the vdiff solve.
        assert float(jnp.max(jnp.abs(tend.u_wind))) == 0.0
        assert float(jnp.max(jnp.abs(tend.v_wind))) == 0.0
        assert float(jnp.max(jnp.abs(tend.temperature))) == 0.0
        assert float(jnp.max(jnp.abs(tend.specific_humidity))) == 0.0

        # (b) Published fluxes are the vdiff-delivered values.
        surface = out["surface"]
        np.testing.assert_allclose(np.asarray(surface.evaporation), 3.0e-5)
        np.testing.assert_allclose(
            np.asarray(surface.effective_evaporation),
            np.asarray(surface.evaporation),
        )
        np.testing.assert_allclose(np.asarray(surface.sensible_heat_flux), 12.0)
        np.testing.assert_allclose(np.asarray(surface.latent_heat_flux), 75.0)
        np.testing.assert_allclose(np.asarray(surface.momentum_flux_u), 0.08)
        np.testing.assert_allclose(np.asarray(surface.momentum_flux_v), -0.02)

    def test_bulk_reference_fluxes_are_published_not_discarded(self):
        """``surface_physics_step``'s explicit bulk flux is now published
        (jax-gcm#1118), not computed-and-thrown-away every step.

        It is a SEPARATE, explicit one-shot bulk-aerodynamic estimate from
        the vdiff-delivered flux this term republishes above, so it need not
        equal it exactly — but with a nonzero air/surface temperature and
        humidity difference and a nonzero exchange coefficient it must be
        finite and nonzero, which a value that was never computed (left at
        its ``SurfaceData.zeros()`` default) would not be.
        """
        from types import SimpleNamespace


        from jcm.forcing import ForcingData
        from jcm.physics.surface.echam.surface_physics import EchamSurface
        from jcm.physics.surface.echam.surface_types import SurfaceData
        from jcm.physics.vertical_diffusion.tte_tke.vertical_diffusion_types import (
            VerticalDiffusionData,
        )
        from jcm.physics_interface import PhysicsState
        from jcm.terrain import TerrainData

        nlev, ncols = 6, 1
        z_half = jnp.linspace(6000.0, 0.0, nlev + 1)[:, None]
        z_full = 0.5 * (z_half[:-1] + z_half[1:])
        p_half = jnp.linspace(5.0e4, 1.0e5, nlev + 1)[:, None]
        p_full = 0.5 * (p_half[:-1] + p_half[1:])

        state = PhysicsState(
            u_wind=jnp.full((nlev, ncols), 8.0),
            v_wind=jnp.full((nlev, ncols), 2.0),
            temperature=jnp.full((nlev, ncols), 290.0),
            specific_humidity=jnp.full((nlev, ncols), 0.008),
            geopotential=9.81 * z_full,
            normalized_surface_pressure=jnp.ones((ncols,)),
            tracers={},
        )
        vdiff = VerticalDiffusionData.zeros((ncols,), nlev).copy(
            surface_evaporation=jnp.array([3.0e-5]),
            surface_sensible_heat=jnp.array([12.0]),
            surface_latent_heat=jnp.array([75.0]),
            surface_stress_u=jnp.array([0.08]),
            surface_stress_v=jnp.array([-0.02]),
            surface_exchange_heat=jnp.full((ncols, 3), 0.01),
            surface_exchange_moisture=jnp.full((ncols, 3), 0.01),
            surface_exchange_momentum=jnp.full((ncols, 3), 0.012),
        )
        diagnostics = {
            "_dt_seconds": 900.0,
            "pressure_full": p_full,
            "pressure_half": p_half,
            "height_full": z_full,
            "height_half": z_half,
            # A warmer-than-air SST with a nonzero exchange coefficient gives
            # the bulk formula a nonzero sensible/latent heat to compute.
            "surface": SurfaceData.zeros((ncols,), nlev).copy(
                surface_temperature=jnp.array([300.0]),
                roughness_length=jnp.array([1e-4]),
            ),
            "vertical_diffusion": vdiff,
            "radiation": SimpleNamespace(
                surface_sw_down=jnp.zeros(ncols),
                surface_lw_down=jnp.full(ncols, 350.0),
            ),
        }
        terrain = TerrainData.single_column(fmask=0.0)
        forcing = ForcingData.zeros((1, 1)).copy(
            sea_surface_temperature=jnp.full((1, 1), 300.0),
        )

        term = EchamSurface()
        _, out = term(state, diagnostics, forcing, terrain)
        surface = out["surface"]

        assert jnp.all(jnp.isfinite(surface.bulk_sensible_heat_flux))
        assert jnp.all(jnp.isfinite(surface.bulk_latent_heat_flux))
        assert jnp.all(jnp.isfinite(surface.bulk_evaporation))
        assert jnp.all(jnp.isfinite(surface.bulk_momentum_flux_u))
        assert jnp.all(jnp.isfinite(surface.bulk_momentum_flux_v))
        # A warmer ocean than the 290 K air, with a nonzero exchange
        # coefficient: the bulk sensible heat must be a real, nonzero
        # (positive, upward) estimate, not the zeros() default.
        assert float(surface.bulk_sensible_heat_flux[0]) > 0.0

    def test_surface_layer_diagnostics_are_published_not_discarded(self):
        """``compute_surface_diagnostics``'s bundle is now published
        (jax-gcm#1140), not computed-and-thrown-away every step.

        Same composed-step setup as the bulk-flux test above (300 K SST,
        290 K / 0.008 kg/kg air, nonzero exchange): every published field
        must be finite and physically sane, and different from the
        ``SurfaceData.zeros()`` default a step that never wired this bundle
        through would leave them at.
        """
        from types import SimpleNamespace

        from jcm.forcing import ForcingData
        from jcm.physics.surface.echam.surface_physics import EchamSurface
        from jcm.physics.surface.echam.surface_types import SurfaceData
        from jcm.physics.vertical_diffusion.tte_tke.vertical_diffusion_types import (
            VerticalDiffusionData,
        )
        from jcm.physics_interface import PhysicsState
        from jcm.terrain import TerrainData

        nlev, ncols = 6, 1
        z_half = jnp.linspace(6000.0, 0.0, nlev + 1)[:, None]
        z_full = 0.5 * (z_half[:-1] + z_half[1:])
        p_half = jnp.linspace(5.0e4, 1.0e5, nlev + 1)[:, None]
        p_full = 0.5 * (p_half[:-1] + p_half[1:])

        state = PhysicsState(
            u_wind=jnp.full((nlev, ncols), 8.0),
            v_wind=jnp.full((nlev, ncols), 2.0),
            temperature=jnp.full((nlev, ncols), 290.0),
            specific_humidity=jnp.full((nlev, ncols), 0.008),
            geopotential=9.81 * z_full,
            normalized_surface_pressure=jnp.ones((ncols,)),
            tracers={},
        )
        vdiff = VerticalDiffusionData.zeros((ncols,), nlev).copy(
            surface_evaporation=jnp.array([3.0e-5]),
            surface_sensible_heat=jnp.array([12.0]),
            surface_latent_heat=jnp.array([75.0]),
            surface_stress_u=jnp.array([0.08]),
            surface_stress_v=jnp.array([-0.02]),
            heat_2m_reduction=jnp.full((ncols, 3), 0.7),
            surface_exchange_heat=jnp.full((ncols, 3), 0.01),
            surface_exchange_moisture=jnp.full((ncols, 3), 0.01),
            surface_exchange_momentum=jnp.full((ncols, 3), 0.012),
        )
        diagnostics = {
            "_dt_seconds": 900.0,
            "pressure_full": p_full,
            "pressure_half": p_half,
            "height_full": z_full,
            "height_half": z_half,
            "surface": SurfaceData.zeros((ncols,), nlev).copy(
                surface_temperature=jnp.array([300.0]),
                roughness_length=jnp.array([1e-4]),
            ),
            "vertical_diffusion": vdiff,
            "radiation": SimpleNamespace(
                surface_sw_down=jnp.zeros(ncols),
                surface_lw_down=jnp.full(ncols, 350.0),
            ),
        }
        terrain = TerrainData.single_column(fmask=0.0)
        forcing = ForcingData.zeros((1, 1)).copy(
            sea_surface_temperature=jnp.full((1, 1), 300.0),
        )

        term = EchamSurface()
        _, out = term(state, diagnostics, forcing, terrain)
        surface = out["surface"]

        for name in ("temperature_2m", "humidity_2m", "dewpoint_2m",
                     "richardson_number", "energy_balance_residual",
                     "temperature_2m_tile", "humidity_2m_tile"):
            value = getattr(surface, name)
            assert jnp.all(jnp.isfinite(value)), name
            # None of these is left at its zeros() default: a warm 300 K
            # SST under 290 K air gives every one of them a real, nonzero
            # value in this setup.
            assert jnp.any(value != 0.0), name

        # With this profile factor the 2 m temperature lies between the
        # input temperatures; the tile-capped dew point cannot exceed it.
        assert 290.0 < float(surface.temperature_2m[0]) < 300.0
        assert float(surface.dewpoint_2m[0]) <= float(surface.temperature_2m[0])

    def test_bulk_land_flux_tracks_the_live_skin_not_stl_am(self):
        """The land tile's bulk flux reads the LIVE skin temperature.

        jax-gcm#1118 Codex review: the published ``surface.bulk_*`` land
        contribution used to read ``forcing.stl_am`` unconditionally, a
        synthetic value that silently diverges from the implicit solve's
        own (possibly prognostic) land state once the skin temperature
        evolves. ``forcing.stl_am`` is held FIXED across both runs below;
        only ``diagnostics["surface"].land_surface_temperature`` (what
        ``TteTkeVerticalDiffusion`` actually published this step) differs.
        If the fix regressed back to reading ``stl_am``, the two runs would
        be identical.
        """
        from types import SimpleNamespace

        from jcm.forcing import ForcingData
        from jcm.physics.surface.echam.surface_physics import EchamSurface
        from jcm.physics.surface.echam.surface_types import SurfaceData
        from jcm.physics.vertical_diffusion.tte_tke.vertical_diffusion_types import (
            VerticalDiffusionData,
        )
        from jcm.physics_interface import PhysicsState
        from jcm.terrain import TerrainData

        nlev, ncols = 6, 1
        z_half = jnp.linspace(6000.0, 0.0, nlev + 1)[:, None]
        z_full = 0.5 * (z_half[:-1] + z_half[1:])
        p_half = jnp.linspace(5.0e4, 1.0e5, nlev + 1)[:, None]
        p_full = 0.5 * (p_half[:-1] + p_half[1:])

        state = PhysicsState(
            u_wind=jnp.full((nlev, ncols), 5.0),
            v_wind=jnp.zeros((nlev, ncols)),
            temperature=jnp.full((nlev, ncols), 290.0),
            specific_humidity=jnp.full((nlev, ncols), 0.006),
            geopotential=9.81 * z_full,
            normalized_surface_pressure=jnp.ones((ncols,)),
            tracers={},
        )
        vdiff = VerticalDiffusionData.zeros((ncols,), nlev).copy(
            surface_exchange_heat=jnp.full((ncols, 3), 0.01),
            surface_exchange_moisture=jnp.full((ncols, 3), 0.01),
            surface_exchange_momentum=jnp.full((ncols, 3), 0.012),
        )
        terrain = TerrainData.single_column(fmask=1.0)  # all land
        # stl_am is identical in both runs; only the LIVE carried skin
        # temperature differs.
        forcing = ForcingData.zeros((1, 1)).copy(
            stl_am=jnp.full((1, 1), 280.0),
            sea_surface_temperature=jnp.full((1, 1), 280.0),
        )

        def run(live_land_temp):
            diagnostics = {
                "_dt_seconds": 900.0,
                "pressure_full": p_full,
                "pressure_half": p_half,
                "height_full": z_full,
                "height_half": z_half,
                "surface": SurfaceData.zeros((ncols,), nlev).copy(
                    surface_temperature=jnp.array([280.0]),
                    roughness_length=jnp.array([0.01]),
                    land_surface_temperature=jnp.array([live_land_temp]),
                    csat=jnp.array([0.6]),
                ),
                "vertical_diffusion": vdiff,
                "radiation": SimpleNamespace(
                    surface_sw_down=jnp.zeros(ncols),
                    surface_lw_down=jnp.full(ncols, 300.0),
                ),
            }
            term = EchamSurface()
            _, out = term(state, diagnostics, forcing, terrain)
            return out["surface"]

        # Scenario A: the carried skin happens to equal stl_am (280 K) — the
        # old, buggy behaviour and the fixed behaviour agree here by
        # coincidence, so this run alone would not catch a regression.
        cold = run(280.0)
        # Scenario B: a prognostic skin well above stl_am (320 K), as a real
        # land-energy-balance solve would produce under strong heating.
        warm = run(320.0)

        assert jnp.all(jnp.isfinite(cold.bulk_sensible_heat_flux))
        assert jnp.all(jnp.isfinite(warm.bulk_sensible_heat_flux))
        # If the land contribution still read stl_am (fixed at 280 K in both
        # runs), these would be equal. Reading the live skin instead, the
        # 320 K run must show substantially MORE upward sensible heat than
        # the 280 K run.
        assert (float(warm.bulk_sensible_heat_flux[0])
                > float(cold.bulk_sensible_heat_flux[0]) + 10.0)


class TestLongwaveReEmission:
    """The cached surface longwave follows the skin temperature (ECHAM ``radheat``, #979)."""

    def _radiation(self, ncols=2, nlev=4):
        from jcm.physics.radiation.radiation_types import RadiationData
        return RadiationData.zeros((ncols,), nlev).copy(
            surface_emissivity=jnp.full((ncols,), 0.95),
            surface_lw_up=jnp.full((ncols,), 450.0),
            lw_flux_up=jnp.full((nlev + 1, ncols), 450.0),
            toa_lw_up=jnp.full((ncols,), 240.0))

    def test_emission_change_heats_the_lowest_layer_and_holds_the_toa(self):
        import numpy as np

        import jcm.constants as c
        from jcm.physics.surface.echam.surface_physics import correct_surface_longwave

        rad = self._radiation()
        p_half = jnp.linspace(2.0e4, 1.0e5, 5)[:, None] * jnp.ones((1, 2))
        t_old, t_new = jnp.array([300.0, 300.0]), jnp.array([302.0, 300.0])
        out, heating = correct_surface_longwave(rad, t_old, t_new, p_half)
        d_up = 4.0 * 0.95 * c.sbc * 300.0 ** 3 * 2.0      # land_rad's zteffl4, linearised
        np.testing.assert_allclose(out.surface_lw_up, [450.0 + d_up, 450.0], rtol=1e-6)
        np.testing.assert_allclose(out.lw_flux_up[-1], out.surface_lw_up, rtol=1e-6)
        np.testing.assert_allclose(out.lw_flux_up[:-1], 450.0)
        # The extra emission is the lowest layer's: c_pd·Δp/g·heating = Δ.
        dp = float(p_half[-1, 0] - p_half[-2, 0])
        np.testing.assert_allclose(heating * c.cpd * dp / c.grav, [d_up, 0.0], rtol=1e-5)
        np.testing.assert_allclose(out.lw_heating_rate[-1], heating, rtol=1e-6)
        np.testing.assert_allclose(out.toa_lw_up, 240.0)
        # No clear-sky solve: the clear-sky profile keeps its zeros. With one,
        # its surface emission moves with the all-sky one.
        np.testing.assert_array_equal(out.lw_flux_up_clear, 0.0)
        clear = rad.copy(lw_flux_up_clear=jnp.full_like(rad.lw_flux_up, 430.0))
        out_c, _ = correct_surface_longwave(clear, t_old, t_new, p_half)
        np.testing.assert_allclose(out_c.lw_flux_up_clear[-1], [430.0 + d_up, 430.0], rtol=1e-6)
        np.testing.assert_allclose(out_c.lw_flux_up_clear[:-1], 430.0)
        # Successive corrections telescope from the last corrected value.
        out2, _ = correct_surface_longwave(out, t_new, t_old, p_half)
        np.testing.assert_allclose(
            out2.surface_lw_up[0],
            450.0 + d_up - 4.0 * 0.95 * c.sbc * 302.0 ** 3 * 2.0, rtol=1e-6)

    def test_term_publishes_the_new_skin_and_its_emission(self, monkeypatch):
        """``EchamSurface`` after the land balance: the grid surface temperature is
        the new skin, the only tendency is the lowest-level re-emission, and that
        heating advances the running thermodynamic state the later terms read.
        """
        import numpy as np
        from jcm.physics.surface.echam import surface_physics as surface_module
        import jcm.constants as c

        captured = {}
        original_step = surface_module.surface_physics_step

        def record_profiles(atmosphere, *args, **kwargs):
            captured["atmosphere"] = atmosphere
            result = original_step(atmosphere, *args, **kwargs)
            captured["diagnostics"] = result[2]
            return result

        monkeypatch.setattr(surface_module, "surface_physics_step", record_profiles)

        from jcm.forcing import ForcingData
        from jcm.physics.surface.echam.surface_physics import EchamSurface
        from jcm.physics.surface.echam.surface_types import SurfaceData
        from jcm.physics.vertical_diffusion.tte_tke.vertical_diffusion_types import (
            VerticalDiffusionData,
        )
        from jcm.physics_interface import PhysicsState
        from jcm.terrain import TerrainData

        nlev, ncols = 4, 1
        z_half = jnp.linspace(4000.0, 0.0, nlev + 1)[:, None]
        p_half = jnp.linspace(6.0e4, 1.0e5, nlev + 1)[:, None]
        state = PhysicsState(
            u_wind=jnp.full((nlev, ncols), 5.0), v_wind=jnp.zeros((nlev, ncols)),
            temperature=jnp.full((nlev, ncols), 295.0),
            specific_humidity=jnp.full((nlev, ncols), 0.008),
            geopotential=9.81 * 0.5 * (z_half[:-1] + z_half[1:]),
            normalized_surface_pressure=jnp.ones((ncols,)), tracers={})
        diagnostics = {
            "_dt_seconds": 720.0,
            "pressure_full": 0.5 * (p_half[:-1] + p_half[1:]), "pressure_half": p_half,
            "height_full": 0.5 * (z_half[:-1] + z_half[1:]),
            "height_half": z_half,
            "surface": SurfaceData.zeros((ncols,), nlev).copy(
                surface_temperature=jnp.array([300.0]),
                land_surface_temperature=jnp.array([303.0]),
                roughness_length=jnp.array([0.01])),
            "vertical_diffusion": VerticalDiffusionData.zeros((ncols,), nlev).copy(
                heat_2m_reduction=jnp.full((ncols, 3), .61),
                surface_exchange_heat=jnp.full((ncols, 3), 0.01),
                surface_exchange_moisture=jnp.full((ncols, 3), 0.01),
                surface_exchange_momentum=jnp.full((ncols, 3), 0.012)),
            "radiation": self._radiation(ncols, nlev),
            "thermo_run": {"temperature": state.temperature,
                           "specific_humidity": state.specific_humidity},
        }
        forcing = ForcingData.zeros((1, 1)).copy(
            sea_surface_temperature=jnp.full((1, 1), 290.0))
        tend, out = EchamSurface()(state, diagnostics, forcing,
                                   TerrainData.single_column(fmask=1.0))
        # ECHAM hydrostatic pgeom1/g from the lowest pressure interfaces,
        # independent of the deliberately different geometric-height fixture.
        p_upper, p_surface = np.asarray(p_half[-2:, 0], dtype=float)
        delta_p = p_surface - p_upper
        alpha = 1. - p_upper / delta_p * np.log(p_surface / p_upper)
        expected_height = c.rd * 295. * (1. + c.vtmpc1 * .008) * alpha / c.grav
        np.testing.assert_allclose(captured["atmosphere"].reference_height,
                                   expected_height, rtol=2e-6)
        np.testing.assert_allclose(captured["atmosphere"].heat_2m_reduction, .61)
        # SurfaceData.zeros supplies cair=csat=0, so the JSBACH boundary
        # here is q_surface=q_air. Both ends share the same moist cp.
        cp_moist = c.cpd * (1. + c.vtmpc2 * .008)
        expected_t2 = (303. + .61 * (295. - 303. + c.grav * expected_height / cp_moist)
                       - c.grav * 2. / cp_moist)
        np.testing.assert_allclose(captured["diagnostics"].temperature_2m,
                                   expected_t2, atol=3e-5)
        np.testing.assert_allclose(out["surface"].surface_temperature, 303.0)
        assert float(out["radiation"].surface_lw_up[0]) > 450.0 + 15.0
        heat = np.asarray(tend.temperature)
        assert heat[-1, 0] > 0.0 and np.all(heat[:-1] == 0.0)
        np.testing.assert_allclose(out["radiation"].lw_heating_rate[-1], heat[-1], rtol=1e-6)
        assert float(jnp.max(jnp.abs(tend.specific_humidity))) == 0.0
        # The running view is the step-start temperature plus the heating, in
        # the lowest level only: convection and the cloud scheme read it.
        run = out["thermo_run"]
        np.testing.assert_allclose(
            run["temperature"], np.asarray(state.temperature) + heat * 720.0, rtol=1e-6)
        assert run["temperature"][-1, 0] > state.temperature[-1, 0]
        np.testing.assert_array_equal(run["temperature"][:-1], state.temperature[:-1])
        np.testing.assert_array_equal(run["specific_humidity"], state.specific_humidity)

    def test_t_grid_new_mirrors_the_boundary_conditions_tile_weighting(self):
        """``t_grid_new`` (the re-emission's "new" surface temperature, #988)
        must use the SAME tile weighting as ``EchamBoundaryConditions``'
        ``surface_temperature`` -- a coastal cell with sea ice, checked
        against :func:`tile_weighted_radiative_temperature` directly.
        """
        import numpy as np

        from jcm.forcing import ForcingData
        from jcm.physics.forcing.echam_boundary_conditions import (
            tile_weighted_radiative_temperature,
        )
        from jcm.physics.surface.echam.surface_physics import EchamSurface
        from jcm.physics.surface.echam.surface_types import SurfaceData
        from jcm.physics.vertical_diffusion.tte_tke.vertical_diffusion_types import (
            VerticalDiffusionData,
        )
        from jcm.physics_interface import PhysicsState
        from jcm.terrain import TerrainData

        nlev, ncols = 4, 1
        z_half = jnp.linspace(4000.0, 0.0, nlev + 1)[:, None]
        p_half = jnp.linspace(6.0e4, 1.0e5, nlev + 1)[:, None]
        state = PhysicsState(
            u_wind=jnp.full((nlev, ncols), 5.0), v_wind=jnp.zeros((nlev, ncols)),
            temperature=jnp.full((nlev, ncols), 270.0),
            specific_humidity=jnp.full((nlev, ncols), 0.004),
            geopotential=9.81 * 0.5 * (z_half[:-1] + z_half[1:]),
            normalized_surface_pressure=jnp.ones((ncols,)), tracers={})
        new_land_skin = jnp.array([250.0])
        sst = jnp.array([[295.0]])
        diagnostics = {
            "_dt_seconds": 720.0,
            "pressure_full": 0.5 * (p_half[:-1] + p_half[1:]), "pressure_half": p_half,
            "height_full": 0.5 * (z_half[:-1] + z_half[1:]),
            "height_half": z_half,
            "surface": SurfaceData.zeros((ncols,), nlev).copy(
                surface_temperature=jnp.array([280.0]),  # last step's blend
                land_surface_temperature=new_land_skin,
                roughness_length=jnp.array([0.01])),
            "vertical_diffusion": VerticalDiffusionData.zeros((ncols,), nlev).copy(
                surface_exchange_heat=jnp.full((ncols, 3), 0.01),
                surface_exchange_moisture=jnp.full((ncols, 3), 0.01),
                surface_exchange_momentum=jnp.full((ncols, 3), 0.012)),
            "radiation": self._radiation(ncols, nlev),
            "thermo_run": {"temperature": state.temperature,
                           "specific_humidity": state.specific_humidity},
        }
        forcing = ForcingData.zeros((1, 1)).copy(
            sea_surface_temperature=sst, sice_am=jnp.array([[0.2]]))
        land_fraction, sea_ice_fraction = 0.4, 0.2
        terrain = TerrainData.single_column(fmask=land_fraction)
        out = EchamSurface()(state, diagnostics, forcing, terrain)[1]

        ice_surface_temp = jnp.minimum(sst.reshape(1), 271.38)  # CTFREEZ
        expect = tile_weighted_radiative_temperature(
            jnp.array([land_fraction]), jnp.array([sea_ice_fraction]),
            new_land_skin, ice_surface_temp, sst.reshape(1))
        np.testing.assert_allclose(
            np.asarray(out["surface"].surface_temperature), np.asarray(expect),
            rtol=1e-6)
        # Sanity: strictly between the coldest and warmest tile.
        got = float(out["surface"].surface_temperature[0])
        assert got > float(new_land_skin[0])
        assert got < float(sst[0, 0])


class TestSurfaceDiagnosticsLandInTheNetCDF:
    """The new ``surface.<field>`` diagnostics (jax-gcm#1140) get their CF
    attrs stamped onto the output Dataset through the SAME
    ``_apply_term_output_attrs`` mechanism ``surface.bulk_*`` (#1118) goes
    through — the function both trajectory paths call to merge a term's
    ``output_attrs`` onto whatever variables of that name the generic
    diagnostics-flattening pipeline already produced (one per
    ``SurfaceData`` field; that flattening is exercised generically
    elsewhere, and directly below for the tile fields specifically, since a
    #1151 Codex review found the SCALAR fields' bare names match what the
    flattener actually produces but the TILE fields' did not).
    """

    def test_new_diagnostics_get_their_cf_attrs_stamped(self):
        from types import SimpleNamespace

        import numpy as np
        import xarray as xr

        from jcm.physics.surface.echam.surface_physics import EchamSurface
        from jcm.predictions import _apply_term_output_attrs

        # ``_apply_term_output_attrs`` is always called with the
        # CALLABLE ``ComposablePhysics.output_attrs`` (it merges every
        # term's own, plain-dict ``output_attrs`` ClassVar); stand in for
        # that merge with the one term under test, the way
        # ``ComposablePhysics.output_attrs()`` would for a package that
        # included it.
        physics = SimpleNamespace(output_attrs=lambda: EchamSurface.output_attrs)

        # The tile variables are named ``surface.<field>_tile.<index>`` —
        # ONE variable per tile, no "tile" dimension — because that's what
        # the generic flattener actually produces (it expands a trailing
        # tile AXIS into separate per-index variables before this function
        # ever sees them); a hand-built Dataset that kept the axis as a
        # dimension (the pre-fix version of this test) exercised a shape
        # ``_apply_term_output_attrs`` never sees for real, which is why it
        # did not catch the #1151 Codex finding that the bare
        # ``surface.<field>_tile`` keys matched nothing.
        ds = xr.Dataset({
            "surface.temperature_2m": (("time",), np.array([282.0, 283.0])),
            "surface.humidity_2m": (("time",), np.array([5.0e-3, 5.1e-3])),
            "surface.dewpoint_2m": (("time",), np.array([278.0, 279.0])),
            "surface.richardson_number": (("time",), np.array([0.0, 0.1])),
            "surface.energy_balance_residual": (("time",), np.array([0.0, 1.0])),
            **{f"surface.temperature_2m_tile.{i}": (("time",), np.zeros(2))
               for i in range(3)},
            **{f"surface.humidity_2m_tile.{i}": (("time",), np.zeros(2))
               for i in range(3)},
            # Not EchamSurface's field: must be left untouched.
            "unrelated.variable": (("time",), np.array([1.0, 2.0])),
        })

        _apply_term_output_attrs(ds, physics)

        for name, expect_units in (
            ("surface.temperature_2m", "K"),
            ("surface.humidity_2m", "kg kg-1"),
            ("surface.dewpoint_2m", "K"),
            ("surface.richardson_number", "1"),
            ("surface.energy_balance_residual", "W m-2"),
        ):
            assert ds[name].attrs["units"] == expect_units
            assert "long_name" in ds[name].attrs

        from jcm.physics.surface.echam.surface_types import DIAG_TILE_NAMES

        for field, expect_units in (
            ("temperature_2m_tile", "K"), ("humidity_2m_tile", "kg kg-1"),
        ):
            for index, tile_name in enumerate(DIAG_TILE_NAMES):
                name = f"surface.{field}.{index}"
                assert ds[name].attrs["units"] == expect_units
                assert "long_name" in ds[name].attrs
                assert ds[name].attrs["surface_type"] == tile_name
                assert ds[name].attrs["tile_index"] == index

        assert ds["unrelated.variable"].attrs == {}


from typing import ClassVar as _ClassVar  # noqa: E402


def _stub_provider_term(provides):
    """Build a ``PhysicsTerm`` that satisfies ``_validate_ordering`` and
    nothing else -- never called, so its missing ``__call__`` implementation
    never matters.
    """
    from jcm.physics.physics_term import PhysicsTerm

    class _StubProvider(PhysicsTerm):
        name: _ClassVar[str] = "stub_provider"

    _StubProvider.provides = tuple(provides)
    return _StubProvider()


class TestSurfaceDiagnosticsTileExpansion:
    """Separate class: needs ``ComposablePhysics`` + a stub upstream term,
    unlike the hand-built-Dataset tests above.
    """

    def test_tile_axis_expands_through_the_real_flattener_with_attrs(self):
        """End-to-end, not hand-built: ``ComposablePhysics.data_struct_to_dict``
        (the SAME code path a real run's output goes through) must actually
        expand ``surface.temperature_2m_tile``/``humidity_2m_tile`` into the
        ``.0``/``.1``/``.2`` names ``DIAG_OUTPUT_ATTRS`` now declares — not
        just that the declared names happen to look right in isolation. A
        #1151 Codex review found the previous bare-name entries matched
        nothing the real flattener ever produced; this exercises that
        flattener directly, then stamps the result onto an actual
        serialized ``xr.Dataset`` and checks the expanded variables carry
        units.
        """
        import numpy as np
        import xarray as xr

        from jcm.physics.composable_physics import ComposablePhysics
        from jcm.physics.surface.echam.surface_physics import EchamSurface
        from jcm.physics.surface.echam.surface_types import (
            DIAG_TILE_FIELDS, DIAG_TILE_NAMES, SurfaceData,
        )
        from jcm.predictions import _apply_term_output_attrs

        nlev, nlat, nlon = 3, 2, 2
        surface = SurfaceData.zeros((nlat, nlon), nlev).copy(
            temperature_2m_tile=jnp.arange(nlat * nlon * 3, dtype=jnp.float32)
            .reshape(nlat, nlon, 3),
            humidity_2m_tile=1.0e-3 * jnp.arange(
                nlat * nlon * 3, dtype=jnp.float32).reshape(nlat, nlon, 3),
        )
        # A leading length-1 axis, exactly as a real run's "time" axis adds
        # one before ``Physics.data_struct_to_dict``'s tile-expansion step
        # looks for a trailing channel axis beyond ``nodal_shape``.
        diagnostics = {"surface": jax.tree_util.tree_map(
            lambda x: x[None], surface)}

        stub = _stub_provider_term(EchamSurface.requires)
        phys = ComposablePhysics(terms=[stub, EchamSurface()])
        flat = phys.data_struct_to_dict(
            diagnostics, nodal_shape=(nlev, nlat, nlon))

        for field in DIAG_TILE_FIELDS:
            assert f"surface.{field}" not in flat, (
                f"surface.{field} was not expanded by the real flattener")
            for index in range(len(DIAG_TILE_NAMES)):
                assert f"surface.{field}.{index}" in flat

        output_attrs = phys.output_attrs()
        ds = xr.Dataset({
            k: (("time",), np.asarray(v).reshape(-1))
            for k, v in flat.items() if k.startswith("surface.")
        })
        _apply_term_output_attrs(ds, phys)

        for field, expect_units in (
            ("temperature_2m_tile", "K"), ("humidity_2m_tile", "kg kg-1"),
        ):
            for index, tile_name in enumerate(DIAG_TILE_NAMES):
                key = f"surface.{field}.{index}"
                assert output_attrs[key]["units"] == expect_units
                assert ds[key].attrs["units"] == expect_units
                assert ds[key].attrs["surface_type"] == tile_name
                assert ds[key].attrs["tile_index"] == index


if __name__ == "__main__":
    pytest.main([__file__])
