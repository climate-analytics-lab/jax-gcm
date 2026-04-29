"""Tests for the ECHAM-Louis surface-layer scheme.

Cross-checks the JAX port against analytical limits and expected behaviour.
The Fortran-against-Fortran agreement for the same column lives in
``fortran_harness/compare_vdiff.py`` — that's the more rigorous test, but
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
        qv=qv, qc=zero, qi=zero,
        pressure_full=p_full, pressure_half=p_half,
        geopotential=height_full * 9.80665,
        air_mass=jnp.full((ncol, nlev), 100.0),
        dry_air_mass=jnp.full((ncol, nlev), 100.0),
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

    def test_neutral_limit_is_finite_and_positive(self):
        # T_air ≈ T_sfc, q_air ≈ q_sat(T_sfc) → near-neutral
        state = _build_state(T_air=300.0, T_sfc=300.0, q_air=0.020)
        params = VDiffParameters.default(surface_layer_scheme="echam_louis")
        wind = jnp.array([5.0])
        sCH, sCM = compute_surface_exchange_coefficients_echam_louis(
            state, params, wind,
            state.surface_temperature, state.temperature[:, -1],
        )
        assert jnp.all(jnp.isfinite(sCH))
        assert jnp.all(sCH > 0)
        assert sCH.shape == (1, 3)

    def test_unstable_enhances_neutral(self):
        # Cold air over warm surface: CH should exceed neutral
        state_neutral = _build_state(T_air=300.0, T_sfc=300.0, q_air=0.020)
        state_unstable = _build_state(T_air=200.0, T_sfc=300.0, q_air=0.001)
        params = VDiffParameters.default(surface_layer_scheme="echam_louis")
        wind = jnp.array([5.0])
        sCH_neut, _ = compute_surface_exchange_coefficients_echam_louis(
            state_neutral, params, wind,
            state_neutral.surface_temperature, state_neutral.temperature[:, -1],
        )
        sCH_unst, _ = compute_surface_exchange_coefficients_echam_louis(
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
        sCH_neut, _ = compute_surface_exchange_coefficients_echam_louis(
            state_neutral, params, wind,
            state_neutral.surface_temperature, state_neutral.temperature[:, -1],
        )
        sCH_stab, _ = compute_surface_exchange_coefficients_echam_louis(
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
        sCH_neut, _ = compute_surface_exchange_coefficients_echam_louis(
            state_neutral, params, wind,
            state_neutral.surface_temperature, state_neutral.temperature[:, -1],
        )
        sCH_strong, _ = compute_surface_exchange_coefficients_echam_louis(
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
        sCH, sCM = compute_surface_exchange_coefficients_echam_louis(
            state, params, wind,
            state.surface_temperature, state.temperature[:, -1],
        )
        # The Louis form sets heat- and moisture-CH equal (no separate
        # moisture roughness in this port).
        np.testing.assert_allclose(np.asarray(sCH), np.asarray(sCM))

    def test_existing_businger_dyer_still_works(self):
        # Regression: don't break the original scheme.
        state = _build_state(T_air=290.0, T_sfc=295.0, q_air=0.010)
        params = VDiffParameters.default(surface_layer_scheme="businger_dyer")
        wind = jnp.array([5.0])
        sCH, sCM = compute_surface_exchange_coefficients(
            state, params, wind,
            state.surface_temperature, state.temperature[:, -1],
        )
        assert jnp.all(jnp.isfinite(sCH))
        assert jnp.all(sCH > 0)

    def test_two_schemes_agree_within_factor_of_3_near_neutral(self):
        # Near neutral, Louis and Businger-Dyer should be close (both
        # asymptote to ``CHN`` ≈ ``κ²/log(z/z0)²·U``).
        state = _build_state(T_air=300.05, T_sfc=300.0, q_air=0.020)
        wind = jnp.array([5.0])

        params_bd = VDiffParameters.default(surface_layer_scheme="businger_dyer")
        params_el = VDiffParameters.default(surface_layer_scheme="echam_louis")
        sCH_bd, _ = compute_surface_exchange_coefficients(
            state, params_bd, wind,
            state.surface_temperature, state.temperature[:, -1],
        )
        sCH_el, _ = compute_surface_exchange_coefficients_echam_louis(
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
        sCH1, _ = compute_surface_exchange_coefficients_echam_louis(
            state, params, wind,
            state.surface_temperature, state.temperature[:, -1],
        )
        # Second call hits the cache
        sCH2, _ = compute_surface_exchange_coefficients_echam_louis(
            state, params, wind,
            state.surface_temperature, state.temperature[:, -1],
        )
        np.testing.assert_allclose(np.asarray(sCH1), np.asarray(sCH2))
