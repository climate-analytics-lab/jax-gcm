"""Tests for the ECHAM-Louis surface-layer scheme.

Cross-checks the JAX port against analytical limits and expected behaviour.
The Fortran-against-Fortran agreement for the same column lives in
``fortran_harness/compare_vdiff.py`` (on the ``origin/fortran-harness-vdiff``
branch, not in this tree) — that's the more rigorous test, but
these are the fast unit tests for CI.
"""
import jax.numpy as jnp
import numpy as np

from .vertical_diffusion_types import VDiffParameters, VDiffState
from .surface_layer import compute_surface_exchange_coefficients_echam_louis
from .turbulence_coefficients import compute_surface_exchange_coefficients


def _build_state(T_air, T_sfc, q_air=0.005, u=5.0, p_sfc=101325.0,
                 p_full_offset=1000.0, z_air=60.0, z0=5.0e-4, ncol=1):
    """Build a minimal (ncol, nsfc_type=3) VDiffState for surface tests."""
    nsfc = 3
    nlev = 4   # arbitrary, only the lowest level matters for these tests
    T_arr = jnp.full((ncol, nlev), T_air)
    qv = jnp.full((ncol, nlev), q_air)
    zero = jnp.zeros((ncol, nlev))
    p_full = jnp.full((ncol, nlev), p_sfc - p_full_offset)
    p_half = jnp.full((ncol, nlev + 1), p_sfc)
    height_full = jnp.full((ncol, nlev), z_air)
    height_half = jnp.zeros((ncol, nlev + 1))
    state = VDiffState(
        u=jnp.full((ncol, nlev), u), v=zero,
        temperature=T_arr,
        qv=qv, qc=zero, qi=zero, cloud_fraction=zero,
        pressure_full=p_full, pressure_half=p_half,
        geopotential=height_full * 9.80665,
        air_mass=jnp.full((ncol, nlev), 100.0),
        surface_temperature=jnp.full((ncol, nsfc), T_sfc),
        surface_fraction=jnp.array([[1.0, 0.0, 0.0]] * ncol),
        roughness_length=jnp.full((ncol, nsfc), z0),
        roughness_heat=jnp.full((ncol, nsfc), z0),
        surface_wetness=jnp.ones((ncol, nsfc)),
        height_full=height_full, height_half=height_half,
        tke=jnp.full((ncol, nlev), 0.1),
        thv_variance=jnp.full((ncol, nlev), 0.01),
        ocean_u=jnp.zeros(ncol), ocean_v=jnp.zeros(ncol),
    )
    return state


class TestECHAMLouisScheme:
    """Behavioural tests for the ECHAM-Louis surface-layer scheme."""

    def test_default_scheme_is_echam_louis(self):
        """The faithful ECHAM-Louis scheme is the default (drives flux + damping)."""
        assert (VDiffParameters.default().surface_layer_scheme
                == VDiffParameters.SCHEME_ECHAM_LOUIS)

    def test_neutral_limit_is_finite_and_positive(self):
        # T_air ≈ T_sfc, q_air ≈ q_sat(T_sfc) → near-neutral
        state = _build_state(T_air=300.0, T_sfc=300.0, q_air=0.020)
        params = VDiffParameters.default(surface_layer_scheme="echam_louis")
        wind = jnp.array([5.0])
        sCH, sCM, sCMom = compute_surface_exchange_coefficients_echam_louis(
            state, params, wind,
            state.surface_temperature, state.temperature[:, -1],
        )
        assert jnp.all(jnp.isfinite(sCH))
        assert jnp.all(sCH > 0)
        assert sCH.shape == (1, 3)
        # Momentum coefficient (CM·|U|) is now returned, finite and positive.
        assert jnp.all(jnp.isfinite(sCMom)) and jnp.all(sCMom > 0)
        assert sCMom.shape == (1, 3)

    def test_unstable_enhances_neutral(self):
        # Cold air over warm surface: CH should exceed neutral
        state_neutral = _build_state(T_air=300.0, T_sfc=300.0, q_air=0.020)
        state_unstable = _build_state(T_air=200.0, T_sfc=300.0, q_air=0.001)
        params = VDiffParameters.default(surface_layer_scheme="echam_louis")
        wind = jnp.array([5.0])
        sCH_neut, _, _ = compute_surface_exchange_coefficients_echam_louis(
            state_neutral, params, wind,
            state_neutral.surface_temperature, state_neutral.temperature[:, -1],
        )
        sCH_unst, _, _ = compute_surface_exchange_coefficients_echam_louis(
            state_unstable, params, wind,
            state_unstable.surface_temperature, state_unstable.temperature[:, -1],
        )
        # Louis-1979 gives ~3-5× enhancement at very negative Ri,
        # not the unbounded growth of Businger-Dyer.
        assert sCH_unst[0, 0] > 1.5 * sCH_neut[0, 0]
        assert sCH_unst[0, 0] < 10.0 * sCH_neut[0, 0]

    def test_stable_suppresses_neutral(self):
        # Warm air over cold surface: CH should be below neutral
        state_neutral = _build_state(T_air=300.0, T_sfc=300.0, q_air=0.020)
        state_stable = _build_state(T_air=303.0, T_sfc=300.0, q_air=0.020)
        params = VDiffParameters.default(surface_layer_scheme="echam_louis")
        wind = jnp.array([5.0])
        sCH_neut, _, _ = compute_surface_exchange_coefficients_echam_louis(
            state_neutral, params, wind,
            state_neutral.surface_temperature, state_neutral.temperature[:, -1],
        )
        sCH_stab, _, _ = compute_surface_exchange_coefficients_echam_louis(
            state_stable, params, wind,
            state_stable.surface_temperature, state_stable.temperature[:, -1],
        )
        assert sCH_stab[0, 0] < sCH_neut[0, 0]

    def test_strongly_unstable_louis_caps_around_5x(self):
        # F1-style cold-air-over-warm (T_air ≈ 200 K, T_sfc ≈ 300 K).
        # Louis (1979) doesn't grow without bound — verify the cap.
        state_neutral = _build_state(T_air=300.0, T_sfc=300.0, q_air=0.020)
        state_strongly_unstable = _build_state(
            T_air=200.0, T_sfc=300.0, q_air=0.0001
        )
        params = VDiffParameters.default(surface_layer_scheme="echam_louis")
        wind = jnp.array([5.0])
        sCH_neut, _, _ = compute_surface_exchange_coefficients_echam_louis(
            state_neutral, params, wind,
            state_neutral.surface_temperature, state_neutral.temperature[:, -1],
        )
        sCH_strong, _, _ = compute_surface_exchange_coefficients_echam_louis(
            state_strongly_unstable, params, wind,
            state_strongly_unstable.surface_temperature,
            state_strongly_unstable.temperature[:, -1],
        )
        ratio = sCH_strong[0, 0] / sCH_neut[0, 0]
        # Louis at very negative Ri asymptotes — empirically ~3-7× over
        # neutral, never the 14× Businger-Dyer would give for Ri ≈ −12.
        assert ratio > 1.5, f"unstable ratio {ratio} too low"
        assert ratio < 8.0, f"unstable ratio {ratio} too high (Louis cap?)"

    def test_returns_heat_equals_moisture(self):
        state = _build_state(T_air=290.0, T_sfc=295.0, q_air=0.010)
        params = VDiffParameters.default(surface_layer_scheme="echam_louis")
        wind = jnp.array([3.0])
        sCH, sCM, sCMom = compute_surface_exchange_coefficients_echam_louis(
            state, params, wind,
            state.surface_temperature, state.temperature[:, -1],
        )
        # The Louis form sets heat- and moisture-CH equal (no separate
        # moisture roughness in this port). Momentum is a distinct drag.
        np.testing.assert_allclose(np.asarray(sCH), np.asarray(sCM))
        assert jnp.all(sCMom > 0)

    def test_existing_businger_dyer_still_works(self):
        # Regression: don't break the original scheme.
        state = _build_state(T_air=290.0, T_sfc=295.0, q_air=0.010)
        params = VDiffParameters.default(surface_layer_scheme="businger_dyer")
        wind = jnp.array([5.0])
        sCH, sCM, sCMom = compute_surface_exchange_coefficients(
            state, params, wind,
            state.surface_temperature, state.temperature[:, -1],
        )
        assert jnp.all(jnp.isfinite(sCH))
        assert jnp.all(sCH > 0)
        assert jnp.all(jnp.isfinite(sCMom)) and jnp.all(sCMom > 0)

    def test_two_schemes_agree_within_factor_of_3_near_neutral(self):
        # Near neutral, Louis and Businger-Dyer should be close (both
        # asymptote to ``CHN`` ≈ ``κ²/log(z/z0)²·U``).
        state = _build_state(T_air=300.05, T_sfc=300.0, q_air=0.020)
        wind = jnp.array([5.0])

        params_bd = VDiffParameters.default(surface_layer_scheme="businger_dyer")
        params_el = VDiffParameters.default(surface_layer_scheme="echam_louis")
        sCH_bd, _, _ = compute_surface_exchange_coefficients(
            state, params_bd, wind,
            state.surface_temperature, state.temperature[:, -1],
        )
        sCH_el, _, _ = compute_surface_exchange_coefficients_echam_louis(
            state, params_el, wind,
            state.surface_temperature, state.temperature[:, -1],
        )
        ratio = sCH_el[0, 0] / sCH_bd[0, 0]
        assert 0.33 < ratio < 3.0, f"near-neutral ratio {ratio} should be O(1)"

    def test_jit_compiles(self):
        # The function is decorated @jax.jit; first call traces.
        state = _build_state(T_air=300.0, T_sfc=300.0, q_air=0.005)
        params = VDiffParameters.default(surface_layer_scheme="echam_louis")
        wind = jnp.array([5.0])
        sCH1, _, _ = compute_surface_exchange_coefficients_echam_louis(
            state, params, wind,
            state.surface_temperature, state.temperature[:, -1],
        )
        # Second call hits the cache
        sCH2, _, _ = compute_surface_exchange_coefficients_echam_louis(
            state, params, wind,
            state.surface_temperature, state.temperature[:, -1],
        )
        np.testing.assert_allclose(np.asarray(sCH1), np.asarray(sCH2))


class TestEchamLouisNeutralDrag:
    """``echam_louis_neutral_drag`` vs ECHAM6.3's own ``ln(1 + z/z0)`` (#1056).

    Reference: land ``mo_surface_land.f90:238`` (``zcdnl``), ocean
    ``mo_surface_ocean.f90:378-380`` (``zalo``/``zcdnw``), sea ice
    ``mo_surface_ice.f90:639,642`` (``zalo``/``zcdni``) — all three tiles
    build the neutral log term the same way, with no cap on ``z/z0``.
    """

    @staticmethod
    def _bn_and_cdn(z_ref, z0_values, wind=5.0):
        """Run :func:`echam_louis_neutral_drag` on one column per ``z0``."""
        from .surface_layer import echam_louis_neutral_drag
        import jcm.constants as c
        ncol = len(z0_values)
        nlev, nsfc = 2, 1
        state = VDiffState(
            u=jnp.zeros((ncol, nlev)), v=jnp.zeros((ncol, nlev)),
            temperature=jnp.full((ncol, nlev), 280.0),
            qv=jnp.zeros((ncol, nlev)), qc=jnp.zeros((ncol, nlev)),
            qi=jnp.zeros((ncol, nlev)), cloud_fraction=jnp.zeros((ncol, nlev)),
            pressure_full=jnp.full((ncol, nlev), 100000.0),
            pressure_half=jnp.full((ncol, nlev + 1), 101325.0),
            geopotential=jnp.zeros((ncol, nlev)),
            air_mass=jnp.full((ncol, nlev), 100.0),
            surface_temperature=jnp.full((ncol, nsfc), 280.0),
            surface_fraction=jnp.ones((ncol, nsfc)),
            roughness_length=jnp.asarray(z0_values).reshape(ncol, nsfc),
            roughness_heat=jnp.asarray(z0_values).reshape(ncol, nsfc),
            surface_wetness=jnp.ones((ncol, nsfc)),
            height_full=jnp.full((ncol, nlev), z_ref),
            height_half=jnp.zeros((ncol, nlev + 1)),
            tke=jnp.full((ncol, nlev), 0.1),
            thv_variance=jnp.full((ncol, nlev), 0.01),
            ocean_u=jnp.zeros(ncol), ocean_v=jnp.zeros(ncol),
        )
        params = VDiffParameters.default(surface_layer_scheme="echam_louis")
        wind_arr = jnp.full((ncol,), wind)
        bn, cfn_m = echam_louis_neutral_drag(state, params, wind_arr)
        karman = c.karman_const
        cdn = (karman / bn[:, 0]) ** 2
        return np.asarray(bn[:, 0]), np.asarray(cdn)

    def test_matches_ln_one_plus_z_over_z0_to_round_off(self):
        """``cdn`` built from ``bn`` must equal ``kappa**2/ln(1+z/z0)**2``
        exactly (the function and the reference formula are the same
        expression; this guards against a future accidental re-floor).
        """
        import jcm.constants as c
        z_ref = 30.0
        z0_values = [1e-4, 1e-3, 0.1, 1.0, 5.0, 10.0]
        bn, cdn = self._bn_and_cdn(z_ref, z0_values)
        karman = c.karman_const
        expect_bn = np.log1p(z_ref / np.asarray(z0_values))
        expect_cdn = (karman / expect_bn) ** 2
        # float32 round-off (jax defaults to x32 unless the test runner
        # enables x64); the two sides compute the identical expression, so
        # agreement is bounded by float32 epsilon, not a formula mismatch.
        np.testing.assert_allclose(bn, expect_bn, rtol=1e-6)
        np.testing.assert_allclose(cdn, expect_cdn, rtol=1e-6)

    def test_relative_difference_vs_the_old_e2_floored_form(self):
        """Reproduces #1056's table: relative difference of the OLD
        ``ln(max(z/z0, e^2))`` form against the current ``ln(1+z/z0)``
        form, at z_ref=30 m across the issue's z0 sweep.
        """
        import jcm.constants as c
        z_ref = 30.0
        karman = c.karman_const
        cases = [
            (1e-4, 5.3e-7), (1e-3, 6.5e-6), (0.1, 1.17e-3),
            (1.0, 1.94e-2), (5.0, -5.34e-2), (10.0, -0.520),
        ]
        z0_values = [z0 for z0, _ in cases]
        _, cdn_new = self._bn_and_cdn(z_ref, z0_values)
        old_bn = np.log(np.maximum(z_ref / np.asarray(z0_values),
                                    np.exp(2.0)))
        cdn_old = (karman / old_bn) ** 2
        reldiff = (cdn_old - cdn_new) / cdn_new
        for (_, expect), got in zip(cases, reldiff):
            assert abs(got - expect) < 2e-3 * max(abs(expect), 1.0), (
                f"expected relative difference {expect}, got {got}")

    def test_smooth_surface_unchanged_to_1e_minus_5(self):
        """Over ocean/sea-ice-like roughness (z0 <~ 1e-3 m) the fix changes
        nothing a production run would notice: old and new forms agree to
        well inside the 1e-5 ceiling the issue sets for 'negligible'.
        """
        z_ref = 30.0
        z0_values = [1.0e-4, 3.0e-4, 1.0e-3]
        _, cdn_new = self._bn_and_cdn(z_ref, z0_values)
        karman = 0.4
        old_bn = np.log(np.maximum(z_ref / np.asarray(z0_values),
                                    np.exp(2.0)))
        cdn_old = (karman / old_bn) ** 2
        reldiff = np.abs((cdn_old - cdn_new) / cdn_new)
        assert np.all(reldiff < 1e-5), reldiff

    def test_no_cap_binds_over_rough_land(self):
        """z/z0 < e^2 (rough land/orography) must no longer be floored: the
        new ``bn`` keeps rising as z0 grows past z_ref/e^2, unlike the old
        form which pinned ``bn`` at exactly 2.0 there.
        """
        z_ref = 30.0
        bn_at_5, _ = self._bn_and_cdn(z_ref, [5.0])
        bn_at_10, _ = self._bn_and_cdn(z_ref, [10.0])
        # Old form: both would give bn == 2.0 (the e^2 floor binds for both,
        # since z_ref/z0 = 6.0 and 3.0 are both below e^2 ≈ 7.39).
        assert bn_at_5[0] != 2.0 and bn_at_10[0] != 2.0
        assert bn_at_10[0] < bn_at_5[0]  # larger z0 -> smaller ln(1+z/z0)


class TestWind10mReduction:
    """ECHAM ``nsurf_diag`` 10 m wind reduction (#723)."""

    def _neutral(self, z_ref, z0):
        """|U(10)|/|U(z_ref)| = ln(1 + (z/z0 - 1)*10/z) / ln(z/z0)."""
        return (np.log1p((z_ref / z0 - 1.0) * 10.0 / z_ref)
                / np.log(z_ref / z0))

    def test_neutral_matches_the_log_profile(self):
        import jcm.constants as c
        from .surface_layer import wind_10m_reduction
        karman = c.karman_const
        z_ref, z0, wind = 32.6, 1.5e-4, 6.0
        bn = np.log(z_ref / z0)
        cfnc = wind * karman ** 2 / bn ** 2
        red = wind_10m_reduction(
            jnp.full((1, 3), cfnc),                # neutral: CM|U| = CM_n|U|
            jnp.full((1, 3), cfnc), jnp.full((1, 3), bn),
            jnp.asarray([z_ref]))
        expect = self._neutral(z_ref, z0)
        assert abs(float(red[0, 0]) - expect) < 1e-5
        # ~0.90 at L47's lowest level over the ocean: the |u|^3.41 sea-salt
        # source is inflated ~1.4x by using the model level instead.
        assert 0.88 < expect < 0.92

    def test_ocean_roughness_range(self):
        for z0, want in ((5e-5, 0.9124), (1.5e-4, 0.9036), (5e-4, 0.8935)):
            assert abs(self._neutral(32.6, z0) - want) < 1e-3

    def test_each_scheme_uses_its_own_neutral_drag(self):
        """A shared reference reads stability at exact neutral (#783 review)."""
        import jcm.constants as c
        from .surface_layer import wind_10m_reduction
        z_ref, wind = 32.6, 6.0
        # Businger-Dyer's hard-coded ocean z0 against a Charnock state z0.
        bn_bd = np.log(max(z_ref, 1.0) / 1e-4)
        bn_louis = np.log(z_ref / 2.2e-4)
        cfnc_bd = wind * c.karman_const ** 2 / bn_bd ** 2
        own = float(wind_10m_reduction(
            jnp.full((1, 1), cfnc_bd), jnp.full((1, 1), cfnc_bd),
            jnp.full((1, 1), bn_bd), jnp.asarray([z_ref]))[0, 0])
        mixed = float(wind_10m_reduction(
            jnp.full((1, 1), cfnc_bd),
            jnp.full((1, 1), wind * c.karman_const ** 2 / bn_louis ** 2),
            jnp.full((1, 1), bn_louis), jnp.asarray([z_ref]))[0, 0])
        assert abs(own - self._neutral(z_ref, 1e-4)) < 1e-4   # true neutral
        assert mixed < own - 0.03                             # spuriously stable

    def test_unstable_is_closer_to_one_than_stable(self):
        import jcm.constants as c
        from .surface_layer import wind_10m_reduction
        z_ref, z0, wind = 32.6, 1.5e-4, 6.0
        bn = np.log(z_ref / z0)
        cfnc = wind * c.karman_const ** 2 / bn ** 2
        args = (jnp.full((1, 1), cfnc), jnp.full((1, 1), bn),
                jnp.asarray([z_ref]))
        red = lambda f: float(
            wind_10m_reduction(jnp.full((1, 1), f * cfnc), *args)[0, 0])
        stable, neutral, unstable = red(0.4), red(1.0), red(3.0)
        assert stable < neutral < unstable <= 1.0

    def test_continuous_across_the_stability_branch(self):
        import jcm.constants as c
        from .surface_layer import wind_10m_reduction
        z_ref, z0, wind = 32.6, 1.5e-4, 6.0
        bn = np.log(z_ref / z0)
        cfnc = wind * c.karman_const ** 2 / bn ** 2
        args = (jnp.full((1, 1), cfnc), jnp.full((1, 1), bn),
                jnp.asarray([z_ref]))
        lo = float(wind_10m_reduction(
            jnp.full((1, 1), cfnc * (1 - 1e-6)), *args)[0, 0])
        hi = float(wind_10m_reduction(
            jnp.full((1, 1), cfnc * (1 + 1e-6)), *args)[0, 0])
        assert abs(lo - hi) < 1e-6

    def test_no_reduction_below_10m_and_bounded(self):
        from .surface_layer import wind_10m_reduction
        bn = np.log(np.asarray([5.0, 40.0])[:, None] / 1e-3)
        red = wind_10m_reduction(
            jnp.full((2, 3), 0.05), jnp.full((2, 3), 0.05),
            jnp.asarray(bn * np.ones((1, 3))), jnp.asarray([5.0, 40.0]))
        assert np.all(np.asarray(red) <= 1.0) and np.all(np.asarray(red) >= 0.0)
        # z_ref below the diagnostic height: zrat is capped at 1, no reduction.
        assert abs(float(red[0, 0]) - 1.0) < 1e-6

    def test_gradient_is_finite(self):
        import jax
        from .surface_layer import wind_10m_reduction
        bn = float(np.log(32.6 / 1.5e-4))
        g = jax.grad(lambda cm: jnp.sum(wind_10m_reduction(
            cm, jnp.full((1, 3), 6.0 * 0.4 ** 2 / bn ** 2),
            jnp.full((1, 3), bn), jnp.asarray([32.6]))))(jnp.full((1, 3), 0.01))
        assert np.all(np.isfinite(np.asarray(g)))


class TestFrozenSurfacePhase:
    """ECHAM ``tlucua``: a frozen tile saturates over ice (#672)."""

    @staticmethod
    def _neutral_frozen_state(q_factor):
        """Air at the tile's potential temperature, qv = q_factor·qsat_ice.

        With equal potential temperatures and no condensate the bulk
        Richardson number is set by the humidity contrast alone, so it is
        zero exactly when the surface saturation equals the air humidity.
        """
        from jcm.constants import PhysicalConstants
        from jcm.physics.thermodynamics import saturation_specific_humidity

        const = PhysicalConstants()
        t_sfc, p_sfc, offset = 260.0, 101325.0, 1000.0
        t_air = t_sfc * ((p_sfc - offset) / p_sfc) ** (const.rd / const.cpd)
        q_ice = float(saturation_specific_humidity(t_sfc, p_sfc, phase="ice"))
        state = _build_state(T_air=t_air, T_sfc=t_sfc, q_air=q_factor * q_ice,
                             p_sfc=p_sfc, p_full_offset=offset)
        return state

    def _exchange(self, state):
        params = VDiffParameters.default(surface_layer_scheme="echam_louis")
        sCH, _, _ = compute_surface_exchange_coefficients_echam_louis(
            state, params, jnp.array([5.0]),
            state.surface_temperature, state.temperature[:, -1],
        )
        return float(sCH[0, 1])

    def test_air_at_ice_saturation_is_neutral(self):
        """At ``q = qsat_ice`` the exchange is exactly the neutral coefficient.

        ``CH·|U| = |U|·κ²/(ln(z/z0m)·ln(z/z0h))`` at ``Ri = 0``; drier air is
        unstable (more exchange), moister stable (less). The Sundqvist
        mixed-phase blend puts the 260 K surface ~8 % above ice saturation,
        which made this case unstable and ~1 % above neutral.
        """
        import jcm.constants as c

        at_sat = self._exchange(self._neutral_frozen_state(1.0))
        drier = self._exchange(self._neutral_frozen_state(0.97))
        moister = self._exchange(self._neutral_frozen_state(1.03))
        assert moister < at_sat < drier
        log_z = np.log(60.0 / 5.0e-4)   # z_ref / z0 (m and h) of _build_state
        neutral = 5.0 * c.karman_const ** 2 / log_z ** 2
        np.testing.assert_allclose(at_sat, neutral, rtol=2e-4)


class TestBusingerDyerGradients:
    """The Businger-Dyer option differentiates on both sides of neutral (#663)."""

    def test_exchange_gradients_are_finite_on_stable_and_unstable_surfaces(self):
        import jax

        params = VDiffParameters.default(surface_layer_scheme="businger_dyer")
        # Unstable, near neutral, strongly stable (Ri > 1/16, where the
        # unstable branch's base would be negative) and strongly unstable
        # (Ri < -0.2, where the stable branch's would divide by zero).
        for t_air, t_sfc in ((290.0, 295.0), (300.0, 300.0),
                             (300.0, 290.0), (250.0, 300.0)):
            state = _build_state(T_air=t_air, T_sfc=t_sfc, q_air=0.010)

            def total(t_surface, wind):
                ch, ce, cm = compute_surface_exchange_coefficients(
                    state, params, wind, t_surface, state.temperature[:, -1])
                return jnp.sum(ch) + jnp.sum(ce) + jnp.sum(cm)

            grads = jax.grad(total, argnums=(0, 1))(
                state.surface_temperature, jnp.array([5.0]))
            for g in grads:
                assert np.all(np.isfinite(np.asarray(g))), (t_air, t_sfc)

