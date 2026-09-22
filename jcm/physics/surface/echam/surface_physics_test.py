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
    
    def _wind_10m(self):
        """Build a diagnosed 10 m wind, as the vdiff carry supplies in a real run."""
        u = self.atmospheric_state.u_wind
        v = self.atmospheric_state.v_wind
        return 0.9 * jnp.sqrt(jnp.maximum(u ** 2 + v ** 2, 1.0e-30))

    def test_surface_physics_step_basic(self):
        """Test basic surface physics step."""
        fluxes, tendencies, diagnostics = surface_physics_step(
            self.atmospheric_state, self.surface_state, self.dt,
            self._wind_10m(),
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
            self._wind_10m(),
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
            self._wind_10m(),
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
            self._wind_10m(),
        )
        doubled = self.atmospheric_state._replace(
            exchange_coeff_momentum=self.atmospheric_state.exchange_coeff_momentum * 2.0
        )
        scaled, _, _ = surface_physics_step(doubled, self.surface_state, self.dt,
                                       self._wind_10m())

        assert jnp.allclose(scaled.momentum_u_mean, 2.0 * base.momentum_u_mean, rtol=1e-5)
        assert jnp.allclose(scaled.momentum_v_mean, 2.0 * base.momentum_v_mean, rtol=1e-5)


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
    the module's wind norms and the ``wind_speed_10m / wind_speed_atm``
    quotient are cone tips at calm wind, where no two-sided derivative
    exists, and a run reaches them (an initialised-at-rest spin-up starts
    there).
    """

    @staticmethod
    def _forcing(ncol, u_wind, v_wind):
        """Lowest-level atmospheric forcing for ``ncol`` columns."""
        nsfc_type = 3
        return AtmosphericForcing(
            temperature=jnp.linspace(290.0, 284.0, ncol),
            humidity=jnp.linspace(0.010, 0.007, ncol),
            u_wind=u_wind,
            v_wind=v_wind,
            pressure=jnp.linspace(101325.0, 97000.0, ncol),
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
        """Return ``f(u, v, T, q, CH, CM, CE, T_ocean, U_10m)``."""
        def f(u_wind, v_wind, temperature, humidity, exchange_heat,
              exchange_momentum, exchange_moisture, ocean_temp,
              wind_speed_10m):
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
            return surface_physics_step(state, surface, 3600.0,
                                        wind_speed_10m)

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
        u_wind = jnp.linspace(6.0, 3.0, ncol)
        v_wind = jnp.linspace(2.5, 4.5, ncol)
        atmospheric_state = self._forcing(ncol, u_wind, v_wind)
        surface_state = self._state(ncol)
        args = (u_wind, v_wind, atmospheric_state.temperature,
                atmospheric_state.humidity,
                atmospheric_state.exchange_coeff_heat,
                atmospheric_state.exchange_coeff_momentum,
                atmospheric_state.exchange_coeff_moisture,
                surface_state.ocean_temp,
                0.9 * jnp.sqrt(u_wind ** 2 + v_wind ** 2))
        check_gradients(self._step(ncol, atmospheric_state, surface_state),
                        args, rtol=1e-3)

    @pytest.mark.parametrize("wind_10m", [0.0, 0.3], ids=["w10=0", "w10>0"])
    def test_calm_wind_gradients_are_finite(self, wind_10m):
        """``u = v = 0``: every wind norm in the step is at its cone tip.

        ``surface_physics.py:154`` and ``turbulent_fluxes.py:220/311/318/322``
        all form ``sqrt(maximum(u^2 + v^2, 1e-30))``, and ``:313`` then
        divides the diagnosed 10 m wind by the result — 1e-15 here. The
        gradient is finite because the floor is 1e-30 rather than 0: below it
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
                surface_state.ocean_temp, jnp.full(ncol, wind_10m))

        def total(*a):
            return sum(jnp.sum(leaf ** 2)
                       for leaf in jax.tree.leaves(step(*a)))

        gradients = jax.grad(total, argnums=tuple(range(len(args))))(*args)
        names = ("u_wind", "v_wind", "temperature", "humidity",
                 "exchange_coeff_heat", "exchange_coeff_momentum",
                 "exchange_coeff_moisture", "ocean_temp", "wind_speed_10m")
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
                surface_state.ocean_temp,
                0.9 * jnp.sqrt(u_wind ** 2 + v_wind ** 2))

        def total(*a):
            return sum(jnp.sum(leaf ** 2)
                       for leaf in jax.tree.leaves(step(*a)))

        gradients = jax.grad(total, argnums=tuple(range(len(args))))(*args)
        names = ("u_wind", "v_wind", "temperature", "humidity",
                 "exchange_coeff_heat", "exchange_coeff_momentum",
                 "exchange_coeff_moisture", "ocean_temp", "wind_speed_10m")
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


if __name__ == "__main__":
    pytest.main([__file__])