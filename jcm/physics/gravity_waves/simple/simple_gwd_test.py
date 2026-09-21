"""Unit tests for the simple monochromatic mountain-wave drag scheme."""

import jax.numpy as jnp
import jax
from .simple_gwd import (
    SimpleGwdParameters, brunt_vaisala_frequency, orographic_source,
    simple_gwd,
)
from jcm.constants import grav, cpd, rd
from jcm.testing import check_gradients


def _sheared_column(nlev=30, top=30000.0):
    """Build a stable column with a real surface wind, on the top-first frame.

    ``height[0]`` is the model top and ``height[-1]`` the surface, matching the
    physics-internal ordering the composable term hands the scheme.
    """
    height = jnp.linspace(0.0, top, nlev)[::-1]
    pressure = 100000.0 * jnp.exp(-height / 8000.0)
    temperature = jnp.maximum(288.0 - 0.0065 * height, 210.0)
    air_density = pressure / (rd * temperature)
    return height, pressure, temperature, air_density


def test_simple_gwd_smoke():
    """A westerly column launches a wave and the drag opposes the wind."""
    height, pressure, temperature, air_density = _sheared_column()
    u_wind = jnp.ones_like(height) * 20.0
    v_wind = jnp.zeros_like(height)

    tend, state = simple_gwd(
        u_wind, v_wind, temperature, pressure,
        height, air_density, 500.0, 1800.0,
    )
    assert jnp.all(jnp.isfinite(tend.dudt))
    # Drag never accelerates the wind, and the wave actually deposits.
    assert jnp.all(tend.dudt * u_wind <= 1e-12)
    assert jnp.max(jnp.abs(tend.dudt)) > 0.0
    assert state.wave_stress[-1] > 0.0


class TestBruntVaisalaFrequency:
    """Test Brunt-Väisälä frequency calculation"""

    def test_stable_atmosphere(self):
        """Test N² in stably stratified atmosphere"""
        # Create stable profile
        nlev = 20
        height = jnp.linspace(20000, 0, nlev)
        pressure = 100000 * jnp.exp(-height / 8000)

        # Stable temperature profile (decreasing with height)
        temperature = 288 - 0.0065 * height

        n2 = brunt_vaisala_frequency(temperature, pressure, height)

        # Should be positive for stable stratification
        assert jnp.all(n2 > 0)

        # Typical tropospheric values ~1e-4 s^-2
        assert jnp.all(n2 < 1e-3)
        assert jnp.mean(n2) > 1e-5

    def test_isothermal_atmosphere(self):
        """Test N² in isothermal atmosphere"""
        nlev = 10
        height = jnp.linspace(10000, 0, nlev)
        pressure = 100000 * jnp.exp(-height / 8000)
        temperature = jnp.ones(nlev) * 273.0

        n2 = brunt_vaisala_frequency(temperature, pressure, height)

        # Isothermal atmosphere has N² = g²/cp/T
        expected = grav**2 / (cpd * 273.0)

        # Should be approximately constant
        assert jnp.std(n2[1:-1]) / jnp.mean(n2[1:-1]) < 0.1

        # Check approximate value
        assert jnp.abs(jnp.mean(n2) - expected) / expected < 0.2


class TestOrographicSource:
    """Test the linear mountain-wave surface stress."""

    def test_source_magnitude(self):
        """Stress scales with wind and with the square of orography height."""
        config = SimpleGwdParameters.default()

        u_sfc = jnp.array(10.0)
        v_sfc = jnp.array(0.0)
        n_sfc = jnp.array(0.01)   # 0.01 s^-1
        rho_sfc = jnp.array(1.0)
        h_std = jnp.array(100.0)  # 100 m mountains

        tau_x1, _ = orographic_source(u_sfc, v_sfc, n_sfc, rho_sfc, h_std, config)

        # Double wind speed -> larger stress.
        tau_x2, _ = orographic_source(
            2 * u_sfc, v_sfc, n_sfc, rho_sfc, h_std, config)
        assert jnp.abs(tau_x2) > jnp.abs(tau_x1)

        # Double mountain height -> stress scales with h^2 (>3x, ~4x).
        tau_x3, _ = orographic_source(
            u_sfc, v_sfc, n_sfc, rho_sfc, 2 * h_std, config)
        assert jnp.abs(tau_x3) > 3 * jnp.abs(tau_x1)

    def test_source_magnitude_is_physical(self):
        """A 300 m peak in a 10 m/s wind gives an O(0.1 Pa) launch stress.

        The pre-#842 formula dropped the ``rho_s k`` factor and returned
        ~4e2 Pa here — three to four orders above real sub-grid orographic
        drag. This pins the corrected order of magnitude.
        """
        config = SimpleGwdParameters.default()
        tau_x, _ = orographic_source(
            jnp.array(10.0), jnp.array(0.0), jnp.array(0.01),
            jnp.array(1.2), jnp.array(300.0), config)
        assert 0.01 < jnp.abs(tau_x) < 1.0

    def test_source_direction(self):
        """Test that source opposes wind direction"""
        config = SimpleGwdParameters.default()

        n_sfc = jnp.array(0.01)
        rho_sfc = jnp.array(1.0)
        h_std = jnp.array(100.0)

        # Westerly wind
        u_sfc = jnp.array(10.0)
        v_sfc = jnp.array(0.0)
        tau_x, tau_y = orographic_source(
            u_sfc, v_sfc, n_sfc, rho_sfc, h_std, config)

        # Stress should oppose wind
        assert tau_x < 0
        assert jnp.abs(tau_y) < 1e-10

        # Northerly wind
        u_sfc = jnp.array(0.0)
        v_sfc = jnp.array(10.0)
        tau_x, tau_y = orographic_source(
            u_sfc, v_sfc, n_sfc, rho_sfc, h_std, config)

        assert jnp.abs(tau_x) < 1e-10
        assert tau_y < 0


class TestGravityWaveDrag:
    """Test the complete gravity wave drag scheme"""

    def test_momentum_deposition(self):
        """A saturating column decelerates the flow with non-zero drag.

        With no wind reversal the deposition comes entirely from McFarlane
        saturation: as the wave climbs into thinner air the flux is capped at
        ``rho k G U^3 / N`` and its convergence brakes the wind. The drag must
        be non-zero, oppose the wind everywhere, and remove kinetic energy.
        """
        config = SimpleGwdParameters.default()
        height, pressure, temperature, air_density = _sheared_column(
            nlev=40, top=40000.0)
        u_wind = jnp.ones_like(height) * 20.0
        v_wind = jnp.zeros_like(height)

        tendencies, state = simple_gwd(
            u_wind, v_wind, temperature, pressure, height, air_density,
            300.0, 1800.0, config)

        assert state.wave_stress[-1] > 0
        assert jnp.all(jnp.isfinite(tendencies.dudt))
        # Non-trivial deposition, decelerating, energy-losing.
        assert jnp.max(jnp.abs(tendencies.dudt)) > 1e-6
        assert jnp.all(tendencies.dudt * u_wind <= 1e-12)
        assert jnp.sum(u_wind * tendencies.dudt) < 0.0
        # It saturates aloft, not in the lowest levels: deposition grows upward.
        assert jnp.any(state.breaking_level > 0)

    def test_critical_level_deposition_and_filtering(self):
        """A wind reversal absorbs the wave and deposits on the source side.

        Above the reversal the flux is fully absorbed (``tau_x`` -> 0). The
        absorbed momentum is deposited just below the critical level, where the
        wind is still aligned with the launch, so it decelerates that flow —
        the sign the pre-#842 ``u.tau < 0`` test got exactly backwards, which
        left the tendency identically zero.
        """
        config = SimpleGwdParameters.default()
        height, pressure, temperature, air_density = _sheared_column(
            nlev=30, top=30000.0)

        # Westerly below 10 km, easterly above -> critical level at 10 km.
        u_wind = jnp.where(height < 10000, 20.0, -20.0)
        v_wind = jnp.zeros_like(height)

        tendencies, state = simple_gwd(
            u_wind, v_wind, temperature, pressure, height, air_density,
            300.0, 1800.0, config)

        # Flux removed above the critical level.
        above = height > 12000
        assert jnp.all(jnp.abs(state.tau_x[above]) < 1e-3)

        # Deposition is non-zero, decelerating, and sits below the reversal.
        assert jnp.max(jnp.abs(tendencies.dudt)) > 1e-6
        assert jnp.all(tendencies.dudt * u_wind <= 1e-12)
        depo = jnp.abs(tendencies.dudt)
        assert jnp.all(depo[height > 10500] < 1e-9)

    def test_extreme_drag_cannot_reverse_wind_in_one_step(self):
        """The ECHAM overshoot guard bounds the increment at extreme drag.

        The sharpest deposition is a high-altitude critical level: a strong
        launch stress (40 m/s low-level flow over 500 m orography, ~1.4 Pa)
        surviving to ~32 km is absorbed across one thin layer where the density
        is ~1e-2 kg/m^3, and the unbounded acceleration there (~0.06 m/s^2)
        would remove ~110 m/s from a 40 m/s wind in an 1800 s step. The guard
        (``mo_ssodrag``'s ``rover = 0.25`` cap, as in the Lott-Miller port)
        limits ``|dU/dt|`` to ``0.25 |U| / dt`` for every dt: the applied
        increment removes at most a quarter of the local wind speed, the wind
        never changes sign, and kinetic energy still strictly decreases.
        """
        config = SimpleGwdParameters.default()
        height, pressure, temperature, air_density = _sheared_column(
            nlev=40, top=40000.0)
        # Westerly to 32 km, reversed above -> critical level in thin air.
        u_wind = jnp.where(height < 32000, 40.0, -40.0)
        v_wind = jnp.zeros_like(height)

        for dt in (900.0, 1800.0, 3600.0):
            tendencies, _ = simple_gwd(
                u_wind, v_wind, temperature, pressure, height, air_density,
                500.0, dt, config)

            speed = jnp.abs(u_wind)
            increment = jnp.abs(tendencies.dudt) * dt
            frac = increment / speed
            assert jnp.all(frac <= 0.25 * (1 + 1e-5))
            # The guard genuinely engaged (the unbounded increment is ~3x the
            # wind), the post-step wind keeps its sign, drag still acts, and
            # energy is still lost.
            assert jnp.max(frac) > 0.249
            u_after = u_wind + dt * tendencies.dudt
            assert jnp.all(u_after * u_wind > 0)
            assert jnp.sum(u_wind * tendencies.dudt) < 0.0

    def test_calm_column_is_zero(self):
        """A wind-free column launches nothing and stays exactly zero."""
        config = SimpleGwdParameters.default()
        height, pressure, temperature, air_density = _sheared_column()
        calm = jnp.zeros_like(height)

        tendencies, state = simple_gwd(
            calm, calm, temperature, pressure, height, air_density,
            300.0, 1800.0, config)

        assert jnp.all(tendencies.dudt == 0.0)
        assert jnp.all(tendencies.dvdt == 0.0)
        assert jnp.all(tendencies.dtedt == 0.0)
        assert jnp.all(jnp.isfinite(tendencies.dudt))
        assert float(state.wave_stress[-1]) == 0.0

    def test_column_matches_vmapped_block(self):
        """The scheme is broadcasting/vmap-agnostic: one column == a block.

        The composable term vmaps :func:`simple_gwd` over ``(nlev, ncols)``;
        each column must match the same profile run on its own.
        """
        config = SimpleGwdParameters.default()
        height, pressure, temperature, air_density = _sheared_column(
            nlev=40, top=40000.0)
        profiles = [
            (jnp.ones_like(height) * 20.0, jnp.zeros_like(height)),
            (jnp.where(height < 10000, 20.0, -20.0), jnp.zeros_like(height)),
            (jnp.linspace(5.0, 40.0, height.size), height * 0.0 + 3.0),
        ]
        hstds = jnp.array([300.0, 200.0, 500.0])

        u_block = jnp.stack([p[0] for p in profiles], axis=1)
        v_block = jnp.stack([p[1] for p in profiles], axis=1)
        col = lambda a: jnp.broadcast_to(a[:, None], u_block.shape)

        tend_b, _ = jax.vmap(
            simple_gwd, in_axes=(1, 1, 1, 1, 1, 1, 0, None, None),
            out_axes=(0, 0),
        )(u_block, v_block, col(temperature), col(pressure), col(height),
          col(air_density), hstds, 1800.0, config)

        for j, (u, v) in enumerate(profiles):
            tend_c, _ = simple_gwd(
                u, v, temperature, pressure, height, air_density,
                float(hstds[j]), 1800.0, config)
            # rtol against the O(1e-4) deposition signal; atol covers the
            # float32 noise tail where vmap reorders the reduction.
            assert jnp.allclose(tend_b.dudt[j], tend_c.dudt, rtol=1e-4, atol=1e-8)
            assert jnp.allclose(tend_b.dvdt[j], tend_c.dvdt, rtol=1e-4, atol=1e-8)

    def test_height_limits(self):
        """Test that GWD only applies within height limits"""
        config = SimpleGwdParameters.default(zmin=5000.0, zmax=25000.0)

        nlev = 40
        height = jnp.linspace(40000, 0, nlev)
        pressure = 100000 * jnp.exp(-height / 8000)
        temperature = jnp.maximum(288 - 0.0065 * height, 210.0)

        u_wind = jnp.ones(nlev) * 20.0
        v_wind = jnp.zeros(nlev)

        air_density = pressure / (rd * temperature)

        tendencies, _ = simple_gwd(
            u_wind, v_wind, temperature, pressure, height, air_density,
            300.0, 1800.0, config)

        # No tendencies below zmin
        below_mask = height < float(config.zmin)
        assert jnp.all(tendencies.dudt[below_mask] == 0)

        # No tendencies above zmax
        above_mask = height > float(config.zmax)
        assert jnp.all(tendencies.dudt[above_mask] == 0)

    def test_energy_conservation(self):
        """Dissipated kinetic energy reappears as heat, dT/dt = -dKE/dt / cp."""
        config = SimpleGwdParameters.default()
        height, pressure, temperature, air_density = _sheared_column(
            nlev=40, top=40000.0)

        u_wind = jnp.ones_like(height) * 25.0
        v_wind = jnp.zeros_like(height)

        tendencies, _ = simple_gwd(
            u_wind, v_wind, temperature, pressure, height, air_density,
            400.0, 1800.0, config)

        # Kinetic energy is removed, never added.
        ke_loss = u_wind * tendencies.dudt + v_wind * tendencies.dvdt
        assert jnp.sum(ke_loss) <= 0.0

        # Heating matches the dissipated KE where drag acts.
        expected_heating = -ke_loss / cpd
        mask = jnp.abs(ke_loss) > 1e-10
        assert jnp.any(mask)
        assert jnp.allclose(
            tendencies.dtedt[mask], expected_heating[mask], rtol=1e-3)

    def test_jax_transformations(self):
        """Test JAX transformations"""
        config = SimpleGwdParameters.default()

        def gwd_loss(u_wind):
            nlev = len(u_wind)
            height = jnp.linspace(20000, 0, nlev)
            pressure = 100000 * jnp.exp(-height / 8000)
            temperature = 288 - 0.0065 * height
            v_wind = jnp.zeros(nlev)
            air_density = pressure / (rd * temperature)

            tend, _ = simple_gwd(
                u_wind, v_wind, temperature, pressure,
                height, air_density, 300.0, 1800.0, config
            )

            return jnp.sum(tend.dudt ** 2)

        # Test JIT
        jitted = jax.jit(gwd_loss)
        u = jnp.ones(20) * 20.0
        loss = jitted(u)
        assert jnp.isfinite(loss)

        # Test gradient
        grad_fn = jax.grad(gwd_loss)
        grad = grad_fn(u)
        assert grad.shape == u.shape
        assert jnp.all(jnp.isfinite(grad))


class TestGradients:
    """AD against a central difference, per issue #820.

    The scheme is built almost entirely out of Euclidean norms and hard
    switches, so these checks exist mainly to fence the norms: ``_safe_hypot``
    replaced four bare ``sqrt(x**2 + y**2)`` whose derivative is ``0/0`` at the
    origin, and the origin is where a calm column and every wave-free level sit.
    """

    @staticmethod
    def _column(nlev=28, u0=18.0, v0=7.0):
        """Build a sheared mid-latitude column that genuinely saturates.

        The gradient checks need ``dudt`` to be a real signal, not float32
        noise: the column carries a strong low-level wind (so a wave launches)
        and reaches 45 km (so the wave saturates and deposits well inside the
        domain). A shallow, weak-wind column instead deposits at the rounding
        floor, and the adjoint identity contracted against that noise is
        meaningless.
        """
        height = jnp.linspace(0.0, 45000.0, nlev)[::-1]
        pressure = 100000.0 * jnp.exp(-height / 8000.0)
        temperature = jnp.maximum(288.0 - 0.0065 * height, 215.0)
        air_density = pressure / (rd * temperature)
        jet = jnp.exp(-((height - 12000.0) / 9000.0) ** 2)
        u_wind = u0 * (0.8 + 0.4 * jet)
        v_wind = v0 * (0.6 + 0.3 * jet)
        return u_wind, v_wind, temperature, pressure, height, air_density

    @staticmethod
    def _outputs(tend, state):
        """Collect the differentiable outputs worth projecting.

        ``breaking_level`` is a cast boolean mask, piecewise constant in every
        input, so a finite difference would report its staircase rather than a
        derivative; it is left out deliberately. Everything else is continuous.
        """
        return (tend.dudt, tend.dvdt, tend.dtedt,
                state.tau_x, state.tau_y, state.wave_stress,
                state.deposited_momentum)

    def test_gradient_check_column(self):
        """One ``(nlev,)`` column: AD matches the central difference."""
        u_wind, v_wind, temperature, pressure, height, air_density = self._column()

        def f(u, v, temp, h_std):
            return self._outputs(*simple_gwd(
                u, v, temp, pressure, height, air_density, h_std, 1800.0))

        # Height, pressure and density are held out of the direction rather
        # than frozen with ``fixed_inputs``: in the composable term they are
        # moist-air diagnostics derived from temperature and surface pressure,
        # so perturbing them independently of ``temp`` would displace the column
        # off its own hydrostatic profile — a state the scheme is never handed.
        check_gradients(
            f, (u_wind, v_wind, temperature, jnp.asarray(420.0)), rtol=2e-3)

    def test_gradient_check_block(self):
        """A ``(nlev, ncols)`` block, vmapped exactly as ``SimpleGwd`` does."""
        columns = [self._column(u0=u0, v0=v0)
                   for u0, v0 in ((18.0, 7.0), (-12.0, 4.0), (25.0, -9.0))]
        stack = lambda i: jnp.stack([col[i] for col in columns], axis=1)
        u_wind, v_wind, temperature = stack(0), stack(1), stack(2)
        pressure, height, air_density = stack(3), stack(4), stack(5)
        h_std = jnp.array([420.0, 180.0, 900.0])

        def f(u, v, temp, hs):
            tend, state = jax.vmap(
                simple_gwd, in_axes=(1, 1, 1, 1, 1, 1, 0, None),
                out_axes=(0, 0),
            )(u, v, temp, pressure, height, air_density, hs, 1800.0)
            return self._outputs(tend, state)

        check_gradients(f, (u_wind, v_wind, temperature, h_std), rtol=2e-3)

    def test_gradients_are_finite_on_a_calm_column(self):
        """A calm column sits exactly on the cone tip of every wind norm.

        Before ``_safe_hypot`` this returned NaN for every input, and because
        the surface norm feeds the whole column's flux, a single calm column in
        a batch took the rest of the batch's gradient with it.
        """
        _, _, temperature, pressure, height, air_density = self._column()
        calm = jnp.zeros_like(temperature)

        def loss(u, v, temp, h_std):
            tend, state = simple_gwd(
                u, v, temp, pressure, height, air_density, h_std, 1800.0)
            return sum(jnp.sum(x ** 2) for x in self._outputs(tend, state))

        grads = jax.grad(loss, argnums=(0, 1, 2, 3))(
            calm, calm, temperature, jnp.asarray(420.0))
        for g in grads:
            assert jnp.all(jnp.isfinite(g))
