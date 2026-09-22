"""Unit tests for turbulent flux calculations."""

import jax
import pytest
import jax.numpy as jnp

from jcm.physics.surface.echam.surface_physics import initialize_surface_state
from jcm.physics.surface.echam.turbulent_fluxes import (
    compute_bulk_richardson_number, compute_stability_functions,
    compute_exchange_coefficients, compute_surface_humidity,
    compute_surface_diagnostics, compute_surface_resistances
)
from jcm.physics.surface.echam.surface_types import (
    AtmosphericForcing, SurfaceFluxes, SurfaceParameters
)
from jcm.testing import check_gradients


def _atmospheric_forcing(ncol, nsfc_type, u_wind, v_wind):
    """Lowest-level forcing for a small block of columns."""
    return AtmosphericForcing(
        temperature=jnp.linspace(290.0, 285.0, ncol),
        humidity=jnp.linspace(0.010, 0.008, ncol),
        u_wind=u_wind,
        v_wind=v_wind,
        pressure=jnp.linspace(101325.0, 98000.0, ncol),
        sw_downward=jnp.linspace(300.0, 250.0, ncol),
        lw_downward=jnp.linspace(350.0, 320.0, ncol),
        rain_rate=jnp.full(ncol, 1.0e-6),
        snow_rate=jnp.zeros(ncol),
        exchange_coeff_heat=jnp.full((ncol, nsfc_type), 0.011),
        exchange_coeff_moisture=jnp.full((ncol, nsfc_type), 0.010),
        exchange_coeff_momentum=jnp.full((ncol, nsfc_type), 0.013),
    )


def _surface_state(ncol):
    """Build a mixed water/ice/land surface the way a run builds it."""
    fraction = jnp.stack([jnp.full(ncol, 0.5), jnp.full(ncol, 0.2),
                          jnp.full(ncol, 0.3)], axis=1)
    return initialize_surface_state(
        ncol, fraction, jnp.linspace(291.5, 286.5, ncol),
        jnp.full((ncol, 2), 268.0), jnp.full((ncol, 4), 283.0))


def _zero_momentum_fluxes(ncol, nsfc_type):
    """Fluxes whose momentum components are exactly zero (calm column)."""
    tile, mean = jnp.zeros((ncol, nsfc_type)), jnp.zeros(ncol)
    return SurfaceFluxes(
        sensible_heat=tile, latent_heat=tile, longwave_net=tile,
        shortwave_net=tile, ground_heat=tile, momentum_u=tile,
        momentum_v=tile, evaporation=tile, transpiration=tile,
        sensible_heat_mean=mean, latent_heat_mean=mean,
        momentum_u_mean=mean, momentum_v_mean=mean, evaporation_mean=mean,
    )


class TestBulkRichardsonNumber:
    """Test bulk Richardson number calculation."""
    
    def test_stable_conditions(self):
        """Test Richardson number for stable conditions."""
        ncol, nsfc_type = 3, 3
        
        # Cold surface, warm air (stable)
        temp_air = jnp.array([290.0, 295.0, 300.0])
        temp_surface = jnp.ones((ncol, nsfc_type)) * 280.0
        humidity_air = jnp.ones(ncol) * 0.01
        humidity_surface = jnp.ones((ncol, nsfc_type)) * 0.008
        wind_speed = jnp.ones(ncol) * 5.0
        
        ri_bulk = compute_bulk_richardson_number(
            temp_air, temp_surface, humidity_air, humidity_surface, wind_speed
        )
        
        assert ri_bulk.shape == (ncol, nsfc_type)
        # Should be positive for stable conditions
        assert jnp.all(ri_bulk > 0)
    
    def test_unstable_conditions(self):
        """Test Richardson number for unstable conditions."""
        ncol, nsfc_type = 3, 3
        
        # Warm surface, cold air (unstable)
        temp_air = jnp.array([280.0, 285.0, 290.0])
        temp_surface = jnp.ones((ncol, nsfc_type)) * 300.0
        humidity_air = jnp.ones(ncol) * 0.01
        humidity_surface = jnp.ones((ncol, nsfc_type)) * 0.012
        wind_speed = jnp.ones(ncol) * 5.0
        
        ri_bulk = compute_bulk_richardson_number(
            temp_air, temp_surface, humidity_air, humidity_surface, wind_speed
        )
        
        assert ri_bulk.shape == (ncol, nsfc_type)
        # Should be negative for unstable conditions
        assert jnp.all(ri_bulk < 0)
    
    def test_neutral_conditions(self):
        """Test Richardson number for neutral conditions."""
        ncol, nsfc_type = 2, 3
        
        # Same temperature (neutral)
        temp_air = jnp.array([290.0, 295.0])
        temp_surface = jnp.ones((ncol, nsfc_type)) * 290.0
        temp_surface = temp_surface.at[0, :].set(290.0)
        temp_surface = temp_surface.at[1, :].set(295.0)
        humidity_air = jnp.ones(ncol) * 0.01
        humidity_surface = jnp.ones((ncol, nsfc_type)) * 0.01
        wind_speed = jnp.ones(ncol) * 5.0
        
        ri_bulk = compute_bulk_richardson_number(
            temp_air, temp_surface, humidity_air, humidity_surface, wind_speed
        )
        
        assert ri_bulk.shape == (ncol, nsfc_type)
        # Should be near zero for neutral conditions
        assert jnp.all(jnp.abs(ri_bulk) < 0.1)
    
    def test_low_wind_conditions(self):
        """Test Richardson number with low wind speed."""
        ncol, nsfc_type = 2, 3
        
        temp_air = jnp.array([290.0, 295.0])
        temp_surface = jnp.ones((ncol, nsfc_type)) * 280.0
        humidity_air = jnp.ones(ncol) * 0.01
        humidity_surface = jnp.ones((ncol, nsfc_type)) * 0.008
        wind_speed = jnp.array([0.1, 0.05])  # Very low wind
        
        ri_bulk = compute_bulk_richardson_number(
            temp_air, temp_surface, humidity_air, humidity_surface, wind_speed
        )
        
        assert ri_bulk.shape == (ncol, nsfc_type)
        # Should be finite (no division by zero)
        assert jnp.all(jnp.isfinite(ri_bulk))
        # Should be large for low wind speeds
        assert jnp.all(ri_bulk > 1.0)


class TestStabilityFunctions:
    """Test stability function calculations."""
    
    def test_stable_stability_functions(self):
        """Test stability functions for stable conditions."""
        ncol, nsfc_type = 3, 3
        
        # Positive Richardson numbers (stable)
        ri_bulk = jnp.ones((ncol, nsfc_type)) * 0.1
        
        phi_h, phi_m = compute_stability_functions(ri_bulk)
        
        assert phi_h.shape == (ncol, nsfc_type)
        assert phi_m.shape == (ncol, nsfc_type)
        
        # Stability functions should be > 1 for stable conditions
        assert jnp.all(phi_h >= 1.0)
        assert jnp.all(phi_m >= 1.0)
        
        # Check specific values
        expected_phi = 1.0 + 5.0 * 0.1
        assert jnp.allclose(phi_h, expected_phi)
        assert jnp.allclose(phi_m, expected_phi)
    
    def test_unstable_stability_functions(self):
        """Test stability functions for unstable conditions.

        Businger-Dyer Φ_m, Φ_h appear in the denominator of the bulk
        exchange coefficients (``CH = κ²/(ln·Φ_m·Φ_h)·…``). To enhance
        turbulent mixing under unstable buoyancy (the standard textbook
        result for free convection), they must be **< 1** for ζ < 0.
        """
        ncol, nsfc_type = 3, 3

        # Negative Richardson numbers (unstable)
        ri_bulk = jnp.ones((ncol, nsfc_type)) * (-0.1)

        phi_h, phi_m = compute_stability_functions(ri_bulk)

        assert phi_h.shape == (ncol, nsfc_type)
        assert phi_m.shape == (ncol, nsfc_type)

        # Stability functions must be < 1 under unstable conditions so
        # that CH = κ²/(ln·Φ_m·Φ_h) is *larger* than neutral — the
        # boundary-layer enhancement of bulk exchange.
        assert jnp.all(phi_h < 1.0)
        assert jnp.all(phi_m < 1.0)
        # Finite & positive guards.
        assert jnp.all(phi_h > 0.0)
        assert jnp.all(phi_m > 0.0)
        assert jnp.all(jnp.isfinite(phi_h))
        assert jnp.all(jnp.isfinite(phi_m))
        # Specific values: ζ ≈ Ri = -0.1 → Φ_h = (1 + 16·0.1)^(-1/2)
        # ≈ 0.620, Φ_m = (1 + 16·0.1)^(-1/4) ≈ 0.787.
        expected_phi_h = (1.0 + 16.0 * 0.1) ** (-0.5)
        expected_phi_m = (1.0 + 16.0 * 0.1) ** (-0.25)
        assert jnp.allclose(phi_h, expected_phi_h, rtol=1e-4)
        assert jnp.allclose(phi_m, expected_phi_m, rtol=1e-4)
    
    def test_neutral_stability_functions(self):
        """Test stability functions for neutral conditions."""
        ncol, nsfc_type = 2, 3
        
        # Zero Richardson numbers (neutral)
        ri_bulk = jnp.zeros((ncol, nsfc_type))
        
        phi_h, phi_m = compute_stability_functions(ri_bulk)
        
        assert phi_h.shape == (ncol, nsfc_type)
        assert phi_m.shape == (ncol, nsfc_type)
        
        # Should be unity for neutral conditions
        assert jnp.allclose(phi_h, 1.0)
        assert jnp.allclose(phi_m, 1.0)
    
    def test_stability_function_limits(self):
        """Test stability function limits."""
        ncol, nsfc_type = 3, 3
        
        # Very stable conditions
        ri_bulk_stable = jnp.ones((ncol, nsfc_type)) * 1.0
        phi_h_stable, phi_m_stable = compute_stability_functions(ri_bulk_stable)
        
        # Should be limited
        stable_limit = 0.2
        expected_phi_stable = 1.0 + 5.0 * stable_limit
        assert jnp.allclose(phi_h_stable, expected_phi_stable)
        
        # Very unstable conditions
        ri_bulk_unstable = jnp.ones((ncol, nsfc_type)) * (-1.0)
        phi_h_unstable, phi_m_unstable = compute_stability_functions(ri_bulk_unstable)
        
        # Should be finite and positive
        assert jnp.all(jnp.isfinite(phi_h_unstable))
        assert jnp.all(jnp.isfinite(phi_m_unstable))
        assert jnp.all(phi_h_unstable > 0.0)
        assert jnp.all(phi_m_unstable > 0.0)


class TestExchangeCoefficients:
    """Test exchange coefficient calculations."""
    
    def test_exchange_coefficient_calculation(self):
        """Test basic exchange coefficient calculation."""
        ncol, nsfc_type = 3, 3
        
        wind_speed = jnp.array([2.0, 5.0, 10.0])
        roughness_momentum = jnp.ones((ncol, nsfc_type)) * 0.01
        roughness_heat = jnp.ones((ncol, nsfc_type)) * 0.001
        stability_heat = jnp.ones((ncol, nsfc_type)) * 1.0
        stability_momentum = jnp.ones((ncol, nsfc_type)) * 1.0
        
        cd, ch, cq = compute_exchange_coefficients(
            wind_speed, roughness_momentum, roughness_heat,
            stability_heat, stability_momentum, min_wind_speed=1.0, von_karman=0.4
        )
        
        assert cd.shape == (ncol, nsfc_type)
        assert ch.shape == (ncol, nsfc_type)
        assert cq.shape == (ncol, nsfc_type)
        
        # Should be positive
        assert jnp.all(cd > 0.0)
        assert jnp.all(ch > 0.0)
        assert jnp.all(cq > 0.0)
        
        # Should increase with wind speed
        assert jnp.all(cd[1, :] > cd[0, :])
        assert jnp.all(cd[2, :] > cd[1, :])
    
    def test_exchange_coefficient_roughness_dependence(self):
        """Test dependence on roughness length."""
        ncol, nsfc_type = 2, 3
        
        wind_speed = jnp.ones(ncol) * 5.0
        roughness_momentum_smooth = jnp.ones((ncol, nsfc_type)) * 1e-4
        roughness_momentum_rough = jnp.ones((ncol, nsfc_type)) * 1e-2
        roughness_heat = jnp.ones((ncol, nsfc_type)) * 1e-4
        stability_heat = jnp.ones((ncol, nsfc_type)) * 1.0
        stability_momentum = jnp.ones((ncol, nsfc_type)) * 1.0
        
        cd_smooth, _, _ = compute_exchange_coefficients(
            wind_speed, roughness_momentum_smooth, roughness_heat,
            stability_heat, stability_momentum, min_wind_speed=1.0, von_karman=0.4
        )
        
        cd_rough, _, _ = compute_exchange_coefficients(
            wind_speed, roughness_momentum_rough, roughness_heat,
            stability_heat, stability_momentum, min_wind_speed=1.0, von_karman=0.4
        )
        
        # Rougher surface should have higher exchange coefficients
        assert jnp.all(cd_rough > cd_smooth)
    
    def test_exchange_coefficient_stability_dependence(self):
        """Test dependence on stability."""
        ncol, nsfc_type = 2, 3
        
        wind_speed = jnp.ones(ncol) * 5.0
        roughness_momentum = jnp.ones((ncol, nsfc_type)) * 0.01
        roughness_heat = jnp.ones((ncol, nsfc_type)) * 0.001
        stability_heat_stable = jnp.ones((ncol, nsfc_type)) * 1.5
        stability_momentum_stable = jnp.ones((ncol, nsfc_type)) * 1.5
        stability_heat_unstable = jnp.ones((ncol, nsfc_type)) * 0.8
        stability_momentum_unstable = jnp.ones((ncol, nsfc_type)) * 0.8
        
        cd_stable, _, _ = compute_exchange_coefficients(
            wind_speed, roughness_momentum, roughness_heat,
            stability_heat_stable, stability_momentum_stable,
            min_wind_speed=1.0, von_karman=0.4
        )
        
        cd_unstable, _, _ = compute_exchange_coefficients(
            wind_speed, roughness_momentum, roughness_heat,
            stability_heat_unstable, stability_momentum_unstable,
            min_wind_speed=1.0, von_karman=0.4
        )
        
        # Unstable conditions should have higher exchange coefficients
        assert jnp.all(cd_unstable > cd_stable)
    
    def test_minimum_wind_speed(self):
        """Test minimum wind speed handling."""
        ncol, nsfc_type = 2, 3
        params = SurfaceParameters.default(min_wind_speed=1.0)
        
        wind_speed = jnp.array([0.1, 0.5])  # Below minimum
        roughness_momentum = jnp.ones((ncol, nsfc_type)) * 0.01
        roughness_heat = jnp.ones((ncol, nsfc_type)) * 0.001
        stability_heat = jnp.ones((ncol, nsfc_type)) * 1.0
        stability_momentum = jnp.ones((ncol, nsfc_type)) * 1.0
        
        cd, ch, cq = compute_exchange_coefficients(
            wind_speed, roughness_momentum, roughness_heat,
            stability_heat, stability_momentum, params.min_wind_speed, params.von_karman
        )
        
        # Should be finite and positive
        assert jnp.all(jnp.isfinite(cd))
        assert jnp.all(cd > 0.0)
        
        # Should be based on minimum wind speed
        cd_min, _, _ = compute_exchange_coefficients(
            jnp.ones(ncol) * params.min_wind_speed, 
            roughness_momentum, roughness_heat,
            stability_heat, stability_momentum, params.min_wind_speed, params.von_karman
        )
        
        assert jnp.allclose(cd, cd_min)


class TestSurfaceHumidity:
    """Test surface humidity calculations."""
    
    def test_surface_humidity_calculation(self):
        """Test surface humidity calculation."""
        ncol, nsfc_type = 3, 3
        
        temp_surface = jnp.ones((ncol, nsfc_type)) * 280.0
        pressure = jnp.ones(ncol) * 101325.0
        
        q_surface = compute_surface_humidity(temp_surface, pressure)
        
        assert q_surface.shape == (ncol, nsfc_type)
        assert jnp.all(q_surface > 0.0)
        assert jnp.all(q_surface < 0.1)  # Should be reasonable
    
    def test_surface_humidity_temperature_dependence(self):
        """Test temperature dependence of surface humidity."""
        ncol, nsfc_type = 3, 3
        
        temp_cold = jnp.ones((ncol, nsfc_type)) * 260.0
        temp_warm = jnp.ones((ncol, nsfc_type)) * 300.0
        pressure = jnp.ones(ncol) * 101325.0
        
        q_cold = compute_surface_humidity(temp_cold, pressure)
        q_warm = compute_surface_humidity(temp_warm, pressure)
        
        # Warmer surface should have higher humidity
        assert jnp.all(q_warm > q_cold)
    
    def test_surface_humidity_pressure_dependence(self):
        """Test pressure dependence of surface humidity."""
        ncol, nsfc_type = 2, 3
        
        temp_surface = jnp.ones((ncol, nsfc_type)) * 280.0
        pressure_low = jnp.ones(ncol) * 85000.0
        pressure_high = jnp.ones(ncol) * 101325.0
        
        q_low_p = compute_surface_humidity(temp_surface, pressure_low)
        q_high_p = compute_surface_humidity(temp_surface, pressure_high)
        
        # Lower pressure should have higher specific humidity
        assert jnp.all(q_low_p > q_high_p)
    
    def test_surface_humidity_bounds(self):
        """Test surface humidity bounds."""
        ncol, nsfc_type = 3, 3
        
        # Test extreme conditions
        temp_surface = jnp.ones((ncol, nsfc_type)) * 350.0  # Very hot
        pressure = jnp.ones(ncol) * 101325.0
        
        q_surface = compute_surface_humidity(temp_surface, pressure)
        
        # Should be clipped to reasonable bounds
        assert jnp.all(q_surface <= 0.1)  # Max 100 g/kg
        assert jnp.all(q_surface >= 0.0)


class TestTurbulentFluxGradients:
    """AD against a central difference for the bulk surface layer (#820).

    All green. The operating points are placed off this module's switches,
    and each test says which one:

     - ``compute_exchange_coefficients`` floors the roughnesses at 1e-5 m
       (``turbulent_fluxes.py:139/141``). ``initialize_surface_state`` sets
       ``roughness_heat = 0.1 * roughness_momentum``, so with the default
       ``z0_water = 1e-4`` the *water* tile's heat roughness lands on that
       floor exactly; a fixture built that way reports the floor's tie
       (``jnp.maximum`` splits 0.5/0.5, so AD returns the mean of the two
       one-sided slopes) rather than anything about the logarithmic profile.
       The roughnesses below are all well above it.
     - ``compute_stability_functions`` selects on ``Ri >= 0`` and clips the
       unstable branch at ``Ri = -0.5``, so the two regimes are checked
       separately and neither fixture touches 0 or -0.5.

    The calm-wind and zero-buoyancy cases are checked for *finiteness*
    instead: both are cone tips where no two-sided derivative exists, so a
    difference is not the reference to compare against — what matters is that
    the gradient is finite and does not poison the column.
    """

    PARAMS = SurfaceParameters.default()

    @staticmethod
    def _columns():
        """Three columns: one stable, one unstable, one near-neutral."""
        return (jnp.array([288.0, 293.0, 280.0]),          # air temperature
                jnp.array([[291.0, 272.0, 285.0],          # tile temperature
                           [296.0, 271.0, 290.0],
                           [278.5, 270.0, 276.0]]),
                jnp.array([0.009, 0.012, 0.004]),          # air humidity
                jnp.array([[0.013, 0.004, 0.010],          # tile humidity
                           [0.018, 0.004, 0.013],
                           [0.006, 0.003, 0.005]]),
                jnp.array([6.0, 3.5, 9.0]))                # wind speed

    def test_bulk_richardson_number(self):
        """Stable, unstable and near-neutral columns in one call."""
        check_gradients(compute_bulk_richardson_number, self._columns(),
                        rtol=1e-3)

    @pytest.mark.parametrize("richardson", [
        jnp.array([[0.02, 0.05, 0.11], [0.03, 0.08, 0.15]]),    # stable
        jnp.array([[-0.02, -0.08, -0.2], [-0.05, -0.12, -0.3]]),  # unstable
    ], ids=["stable", "unstable"])
    def test_stability_functions(self, richardson):
        """One regime per call: the branch switch at ``Ri = 0`` is a kink."""
        check_gradients(compute_stability_functions, (richardson,), rtol=1e-3)

    def test_exchange_coefficients(self):
        """Roughnesses an order of magnitude clear of the 1e-5 m floor."""
        _, _, _, _, wind_speed = self._columns()
        roughness_momentum = jnp.array([[2.0e-4, 1.0e-3, 0.10],
                                        [3.0e-4, 1.2e-3, 0.20],
                                        [5.0e-4, 8.0e-4, 0.05]])
        stability_heat, stability_momentum = compute_stability_functions(
            jnp.array([[0.02, -0.05, 0.11],
                       [-0.03, 0.08, -0.15],
                       [0.05, -0.2, 0.01]]))
        check_gradients(
            lambda u, z0m, z0h, ph, pm: compute_exchange_coefficients(
                u, z0m, z0h, ph, pm, self.PARAMS.min_wind_speed,
                self.PARAMS.von_karman),
            (wind_speed, roughness_momentum, 0.1 * roughness_momentum,
             stability_heat, stability_momentum),
            rtol=1e-3)

    def test_surface_humidity(self):
        """Tile temperatures spanning the freezing point, off both clips."""
        _, temperature_surface, _, _, _ = self._columns()
        check_gradients(compute_surface_humidity,
                        (temperature_surface,
                         jnp.array([1.0e5, 9.7e4, 1.01e5])),
                        rtol=1e-3)

    @pytest.mark.parametrize("wind_10m", [0.0, 0.3], ids=["w10=0", "w10>0"])
    def test_calm_wind_diagnostics_gradients_are_finite(self, wind_10m):
        """The cone tips of ``compute_surface_diagnostics`` at ``u = v = 0``.

        Three wind norms meet here at once: ``sqrt(u^2 + v^2)`` at
        ``turbulent_fluxes.py:311``, the momentum-flux magnitude at ``:318``
        and the friction velocity at ``:322``, plus the quotient
        ``wind_speed_10m / wind_speed_atm`` at ``:313`` whose denominator is
        then the 1e-30 floor's square root, 1e-15.

        Finite, and for the reason the floors are 1e-30 rather than 0: below
        the floor ``jnp.maximum`` sends the whole derivative to the constant
        branch, so the ``sqrt'`` singularity is multiplied by an exact zero
        instead of forming ``0 * inf``. A ``maximum(x, 0.0)`` there would tie
        at the tip and split 0.5/0.5 against ``sqrt'(0) = inf``, which is the
        shape that produces a NaN. ``wind_10m = 0.3`` with a calm column is
        the awkward combination — a finite 10 m wind over a 1e-15 denominator
        — and is checked alongside the fully calm one.
        """
        ncol, nsfc_type = 2, 3
        calm = jnp.zeros(ncol)
        atmospheric_state = _atmospheric_forcing(ncol, nsfc_type, calm, calm)
        surface_state = _surface_state(ncol)
        fluxes = _zero_momentum_fluxes(ncol, nsfc_type)
        resistances = compute_surface_resistances(
            atmospheric_state, surface_state,
            compute_bulk_richardson_number(
                atmospheric_state.temperature, surface_state.temperature,
                atmospheric_state.humidity,
                jnp.full((ncol, nsfc_type), 0.01), calm))

        def total(u_wind, v_wind, momentum_u, momentum_v, wind_speed_10m):
            state = atmospheric_state._replace(u_wind=u_wind, v_wind=v_wind)
            flux = fluxes._replace(momentum_u_mean=momentum_u,
                                   momentum_v_mean=momentum_v)
            diagnostics = compute_surface_diagnostics(
                state, surface_state, flux, resistances, wind_speed_10m)
            return sum(jnp.sum(leaf ** 2)
                       for leaf in jax.tree.leaves(diagnostics))

        args = (calm, calm, calm, calm, jnp.full(ncol, wind_10m))
        gradients = jax.grad(total, argnums=tuple(range(len(args))))(*args)
        names = ("u_wind", "v_wind", "momentum_u_mean", "momentum_v_mean",
                 "wind_speed_10m")
        for name, gradient in zip(names, gradients):
            assert jnp.all(jnp.isfinite(gradient)), (
                f"d/d{name} is not finite at calm wind: {gradient}")

    def test_zero_buoyancy_gradients_are_finite(self):
        """``Ri = 0`` exactly: air and surface at the same virtual temperature.

        The branch switch in ``compute_stability_functions`` sits at
        ``Ri >= 0`` and the unstable branch carries ``jnp.abs(Ri)``, so this
        point is a kink — the check is finiteness, not agreement with a
        difference. It is also the point an aquaplanet spin-up passes through.
        """
        ncol, nsfc_type = 2, 3
        temperature_air = jnp.array([290.0, 285.0])
        humidity_air = jnp.array([0.010, 0.008])

        def total(temperature_surface, humidity_surface, wind_speed):
            richardson = compute_bulk_richardson_number(
                temperature_air, temperature_surface, humidity_air,
                humidity_surface, wind_speed)
            stability_heat, stability_momentum = compute_stability_functions(
                richardson)
            coefficients = compute_exchange_coefficients(
                wind_speed,
                jnp.full((ncol, nsfc_type), 1.0e-3),
                jnp.full((ncol, nsfc_type), 1.0e-4),
                stability_heat, stability_momentum,
                self.PARAMS.min_wind_speed, self.PARAMS.von_karman)
            return (jnp.sum(richardson ** 2)
                    + sum(jnp.sum(x ** 2) for x in coefficients))

        args = (jnp.broadcast_to(temperature_air[:, None],
                                 (ncol, nsfc_type)) * jnp.ones(1),
                jnp.broadcast_to(humidity_air[:, None],
                                 (ncol, nsfc_type)) * jnp.ones(1),
                jnp.array([6.0, 4.0]))
        gradients = jax.grad(total, argnums=(0, 1, 2))(*args)
        for name, gradient in zip(
                ("temperature_surface", "humidity_surface", "wind_speed"),
                gradients):
            assert jnp.all(jnp.isfinite(gradient)), (
                f"d/d{name} is not finite at zero buoyancy: {gradient}")


if __name__ == "__main__":
    pytest.main([__file__])