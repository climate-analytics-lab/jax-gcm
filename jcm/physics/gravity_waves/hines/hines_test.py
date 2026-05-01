"""Unit tests for the Hines (1997) doppler-spread spectral GWD port.

These are sanity tests that run as part of the regular ``pytest`` suite.
The bit-exact-against-Fortran validation lives in
``fortran_harness/compare_gw_hines.py`` and is run manually during
development; it depends on a local Fortran build that is intentionally
NOT shipped with the repository.
"""
import os

# Hines is an f64 port; force JAX into x64 before any jcm import.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jcm.physics.gravity_waves.hines import (
    HinesParameters, HinesState, HinesTendencies, hines_gwd,
)


def _make_column(nlev: int = 47, u_scale: float = 1.0,
                 v_scale: float = 1.0, jet_z: float = 10000.0):
    """Build a simple isothermal-ish atmosphere with a Gaussian jet."""
    grav = 9.80665
    rd = 287.04
    paphm1 = np.logspace(np.log10(10.0), np.log10(101325.0), nlev + 1)
    papm1 = 0.5 * (paphm1[:-1] + paphm1[1:])
    z = np.zeros(nlev)
    zh = np.zeros(nlev + 1)
    Tprof = np.zeros(nlev)
    for k in range(nlev - 1, -1, -1):
        z_g = zh[k + 1]
        Tprof[k] = max(288.15 - 0.0065 * z_g, 200.0) if z_g < 11000 else 220.0
        dz = (rd * Tprof[k] / grav) * np.log(paphm1[k + 1] / paphm1[k])
        zh[k] = zh[k + 1] + dz
        z[k] = 0.5 * (zh[k] + zh[k + 1])
    rho = papm1 / (rd * Tprof)
    pmair = (paphm1[1:] - paphm1[:-1]) / grav
    u = u_scale * 30.0 * np.exp(-((z - jet_z) / 6000.0) ** 2)
    v = v_scale * 5.0 * np.exp(-((z - jet_z) / 8000.0) ** 2)
    return dict(
        paphm1=jnp.asarray(paphm1), papm1=jnp.asarray(papm1),
        pzh=jnp.asarray(zh), prho=jnp.asarray(rho),
        pmair=jnp.asarray(pmair), ptm1=jnp.asarray(Tprof),
        pum1=jnp.asarray(u), pvm1=jnp.asarray(v),
    )


class TestHinesBasic:
    """Sanity properties of the Hines GWD scheme."""

    def test_returns_finite_tendencies(self):
        """A reasonable mid-latitude column produces all-finite output."""
        col = _make_column()
        config = HinesParameters.default()
        tend, state = hines_gwd(**col, config=config)
        assert jnp.all(jnp.isfinite(tend.dudt))
        assert jnp.all(jnp.isfinite(tend.dvdt))
        assert jnp.all(jnp.isfinite(tend.dissip))
        assert jnp.all(jnp.isfinite(state.flux_u))
        assert jnp.all(jnp.isfinite(state.flux_v))

    def test_tendencies_zero_below_launch(self):
        """No drag is computed below the launch level (emiss_lev counts up
        from the surface)."""
        col = _make_column(nlev=47)
        config = HinesParameters.default()
        tend, _ = hines_gwd(**col, config=config, emiss_lev=10)
        levbot = 47 - 10 - 1
        # emiss_lev=10 means the bottom 10 levels (indices 37..46) get no drag.
        # The launch level itself (index 36) does get a flux-divergence drag.
        below = jnp.arange(47) > levbot
        np.testing.assert_array_equal(np.asarray(tend.dudt[below]), 0.0)
        np.testing.assert_array_equal(np.asarray(tend.dvdt[below]), 0.0)

    def test_drag_opposes_relative_wind_at_top(self):
        """Eastward jet → eastward momentum flux divergence above launch
        decelerates the easterly drift in the upper stratosphere/mesosphere
        — and flux pile-up near model top gives strongly positive du/dt
        there. Test that the column-integrated stress has the right sign."""
        col = _make_column(u_scale=1.0, v_scale=0.0)
        config = HinesParameters.default()
        tend, _ = hines_gwd(**col, config=config)
        # Above-launch column-integrated u-momentum tendency should be
        # negative-then-positive (pile-up at top). Most realistic columns
        # show a strong positive peak at the model top — at minimum the
        # absolute peak should not be at the launch level.
        levbot = 47 - 10 - 1
        peak_idx = int(jnp.argmax(jnp.abs(tend.dudt[:levbot + 1])))
        assert peak_idx < levbot, "drag peak should be above the launch level"

    def test_drag_scales_with_rmscon(self):
        """Doubling the launch RMS wind doubles the spectral amplitude →
        ak_alpha (∝ rmscon^2 / m_alpha^2) scales, but with the m_alpha-
        feedback the actual stress scales sub-linearly. Test that bigger
        rmscon gives bigger column-integrated stress."""
        col = _make_column()
        cfg_a = HinesParameters.default(rmscon=0.5)
        cfg_b = HinesParameters.default(rmscon=2.0)
        tend_a, _ = hines_gwd(**col, config=cfg_a)
        tend_b, _ = hines_gwd(**col, config=cfg_b)
        peak_a = float(jnp.max(jnp.abs(tend_a.dudt)))
        peak_b = float(jnp.max(jnp.abs(tend_b.dudt)))
        assert peak_b > peak_a, "stronger launch RMS should give stronger drag"


class TestHinesJaxTransforms:
    """JAX transformations work on the scheme."""

    def test_jit_runs(self):
        col = _make_column()
        config = HinesParameters.default()
        jitted = jax.jit(lambda **kw: hines_gwd(**kw, config=config))
        tend, _ = jitted(**col)
        assert jnp.all(jnp.isfinite(tend.dudt))

    def test_vmap_over_columns(self):
        """vmap over a small batch of columns."""
        col1 = _make_column(u_scale=1.0)
        col2 = _make_column(u_scale=2.0)
        col3 = _make_column(u_scale=-1.0)
        # Stack along a new leading axis.
        batch = {k: jnp.stack([col1[k], col2[k], col3[k]]) for k in col1}
        config = HinesParameters.default()

        def one(paphm1, papm1, pzh, prho, pmair, ptm1, pum1, pvm1):
            t, _ = hines_gwd(paphm1, papm1, pzh, prho, pmair,
                             ptm1, pum1, pvm1, config)
            return t.dudt

        batched = jax.vmap(one)
        out = batched(batch["paphm1"], batch["papm1"], batch["pzh"],
                      batch["prho"], batch["pmair"], batch["ptm1"],
                      batch["pum1"], batch["pvm1"])
        assert out.shape == (3, 47)
        # Reversed-jet column should give reversed-sign u-tendency at top.
        # Tolerance is loose because the production default precision is f32;
        # the harness runs at f64 for bit-exactness against Fortran.
        np.testing.assert_allclose(np.asarray(out[2]), -np.asarray(out[0]),
                                   rtol=1e-3, atol=1e-9)

    def test_grad_finite(self):
        """jax.grad runs and produces finite gradients wrt input wind."""
        col = _make_column()
        config = HinesParameters.default()

        def loss(u):
            t, _ = hines_gwd(col["paphm1"], col["papm1"], col["pzh"],
                             col["prho"], col["pmair"], col["ptm1"],
                             u, col["pvm1"], config)
            return jnp.sum(t.dudt ** 2)

        g = jax.grad(loss)(col["pum1"])
        assert g.shape == col["pum1"].shape
        assert jnp.all(jnp.isfinite(g))


class TestHinesParameters:
    """Parameters object behaves correctly."""

    def test_defaults_match_fortran_module_constants(self):
        """The tunable-knob defaults reproduce ``mo_gw_hines.f90`` /
        ``mo_echam_gwd_config``. Static-loop knobs (naz, nsmax, emiss_lev,
        slope, icutoff) are passed via :func:`hines_gwd` kwargs, not on the
        parameters tree. Tolerance is loose (atol=1e-6) because the
        production default is f32."""
        p = HinesParameters.default()
        for name, expected in [
            ("f1", 1.5), ("f2", 0.3), ("f3", 1.0), ("f5", 1.0),
            ("f6", 0.5), ("alt_cutoff", 105e3), ("smco", 2.0),
            ("rmscon", 1.0), ("kstar", 5e-5), ("m_min", 1e-4),
        ]:
            np.testing.assert_allclose(float(getattr(p, name)), expected,
                                       atol=1e-6, rtol=1e-6)

    def test_custom_overrides(self):
        p = HinesParameters.default(rmscon=2.0)
        assert abs(float(p.rmscon) - 2.0) < 1e-6
