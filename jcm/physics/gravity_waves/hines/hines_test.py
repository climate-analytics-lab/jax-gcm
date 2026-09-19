"""Unit tests for the Hines (1997) doppler-spread spectral GWD port.

These are sanity tests that run as part of the regular ``pytest`` suite.
The bit-exact-against-Fortran validation lives in
``fortran_harness/compare_gw_hines.py`` (on the
``origin/fortran-harness-vdiff`` branch, not in this tree) and is run
manually during development; it depends on a local Fortran build that
is intentionally NOT shipped with the repository.
"""
import os

# Hines is an f64 port; force JAX into x64 before any jcm import.
os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jcm.physics.gravity_waves.hines import (
    HinesParameters, hines_gwd,
)
from jcm.testing import check_gradients


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
        pressure_half=jnp.asarray(paphm1),
        pressure_full=jnp.asarray(papm1),
        height_half=jnp.asarray(zh),
        density=jnp.asarray(rho),
        layer_mass=jnp.asarray(pmair),
        temperature=jnp.asarray(Tprof),
        u_wind=jnp.asarray(u),
        v_wind=jnp.asarray(v),
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
        """No drag is computed below the launch level (launch_level counts
        up from the surface).
        """
        col = _make_column(nlev=47)
        config = HinesParameters.default()
        tend, _ = hines_gwd(**col, config=config, launch_level=10)
        launch_idx = 47 - 10 - 1
        # launch_level=10 means the bottom 10 levels (indices 37..46) get no
        # drag. The launch level itself (index 36) does get a flux-divergence
        # drag.
        below = jnp.arange(47) > launch_idx
        np.testing.assert_array_equal(np.asarray(tend.dudt[below]), 0.0)
        np.testing.assert_array_equal(np.asarray(tend.dvdt[below]), 0.0)

    def test_drag_opposes_relative_wind_at_top(self):
        """Eastward jet → eastward momentum flux divergence above launch
        decelerates the easterly drift in the upper stratosphere/mesosphere
        — and flux pile-up near model top gives strongly positive du/dt
        there. Test that the column-integrated stress has the right sign.
        """
        col = _make_column(u_scale=1.0, v_scale=0.0)
        config = HinesParameters.default()
        tend, _ = hines_gwd(**col, config=config)
        # Above-launch column-integrated u-momentum tendency should be
        # negative-then-positive (pile-up at top). Most realistic columns
        # show a strong positive peak at the model top — at minimum the
        # absolute peak should not be at the launch level.
        launch_idx = 47 - 10 - 1
        peak_idx = int(jnp.argmax(jnp.abs(tend.dudt[:launch_idx + 1])))
        assert peak_idx < launch_idx, \
            "drag peak should be above the launch level"

    def test_drag_scales_with_rms_launch_wind(self):
        """Doubling the launch RMS wind doubles the spectral amplitude →
        with the m_alpha feedback the actual stress scales sub-linearly.
        Test that a bigger launch wind gives stronger column drag.
        """
        col = _make_column()
        cfg_a = HinesParameters.default(rms_launch_wind=0.5)
        cfg_b = HinesParameters.default(rms_launch_wind=2.0)
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
        """Vmap over a small batch of columns."""
        col1 = _make_column(u_scale=1.0)
        col2 = _make_column(u_scale=2.0)
        col3 = _make_column(u_scale=-1.0)
        keys = list(col1.keys())
        batch = {k: jnp.stack([col1[k], col2[k], col3[k]]) for k in keys}
        config = HinesParameters.default()

        def one(*args):
            t, _ = hines_gwd(*args, config=config)
            return t.dudt

        out = jax.vmap(one)(*[batch[k] for k in keys])
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
            t, _ = hines_gwd(
                col["pressure_half"], col["pressure_full"],
                col["height_half"], col["density"], col["layer_mass"],
                col["temperature"], u, col["v_wind"], config,
            )
            return jnp.sum(t.dudt ** 2)

        g = jax.grad(loss)(col["u_wind"])
        assert g.shape == col["u_wind"].shape
        assert jnp.all(jnp.isfinite(g))


class TestHinesParameters:
    """Parameters object behaves correctly."""

    def test_defaults_match_echam_namelist(self):
        """The default tunable knobs reproduce ECHAM-A's namelist values.
        Static loop knobs (launch_level, num_azimuths, smoothing_passes)
        live as :func:`hines_gwd` kwargs. Tolerance is loose because the
        project default precision is f32.
        """
        p = HinesParameters.default()
        for name, expected in [
            ("wave_amplitude_factor", 1.5),
            ("spectrum_width_factor", 0.3),
            ("mol_diffusion_factor", 1.0),
            ("heating_efficiency", 1.0),
            ("diffusion_efficiency", 0.5),
            ("cutoff_altitude", 105e3),
            ("smoothing_coeff", 2.0),
            ("rms_launch_wind", 1.0),
            ("typical_horizontal_wavenumber", 5e-5),
            ("min_vertical_wavenumber", 1e-4),
        ]:
            np.testing.assert_allclose(float(getattr(p, name)), expected,
                                       atol=1e-6, rtol=1e-6)

    def test_custom_overrides(self):
        p = HinesParameters.default(rms_launch_wind=2.0)
        assert abs(float(p.rms_launch_wind) - 2.0) < 1e-6


class TestHinesGradients:
    """AD against a central difference through the Hines column (#820).

    The scheme comes out clean wherever a wind spectrum exists. What it does
    not survive is a column with *no* wind at all: the azimuthal sums at
    ``hines.py:253-258`` take the square root of a quantity that is identically
    0 when ``u = v = 0`` everywhere, so the launch spectrum is degenerate and
    the response to a displacement is a jump rather than a slope. The
    gradients stay finite there — which is what matters for a model that can
    pass through a calm column — but no secant can be taken, and that is
    recorded rather than tuned away.

    Hines is an f64 port and this module sets ``JAX_ENABLE_X64`` before
    importing jcm, but issue #729's conftest pins float32 for the suite, so
    inside pytest these run in single precision like everything else. Hence
    ``rtol=1e-2`` rather than the 1e-3 the same check reaches standalone under
    x64: the consistency search settles on a coarse rung (1.6e-5) here, where
    the central secant's own truncation plus float32 round-off leaves the
    reference about 0.8% from the AD value. The tolerance measures the
    reference's precision, not the scheme's derivative.
    """

    @staticmethod
    def _scheme_fn(column, config):
        """Return f(temperature, u, v, density) -> the three tendencies."""
        def f(temperature, u_wind, v_wind, density):
            tend, _ = hines_gwd(
                column["pressure_half"], column["pressure_full"],
                column["height_half"], density, column["layer_mass"],
                temperature, u_wind, v_wind, config)
            return (tend.dudt, tend.dvdt, tend.dissip)

        return f

    @staticmethod
    def _args(column):
        return (column["temperature"], column["u_wind"], column["v_wind"],
                column["density"])

    @pytest.mark.parametrize("seed", [0, 3])
    def test_jet_column_gradients_match_a_central_difference(self, seed):
        """A mid-latitude jet — the state the scheme is written for."""
        column = _make_column(nlev=30)
        check_gradients(self._scheme_fn(column, HinesParameters.default()),
                        self._args(column), rtol=1e-2, seed=seed)

    @pytest.mark.parametrize(
        "u_scale, v_scale", [(0.0, 0.0), (0.02, 0.02), (1.0, 0.0)])
    def test_gradients_are_finite_for_any_wind_profile(self, u_scale, v_scale):
        """Calm, near-calm and unidirectional columns all stay finite."""
        column = _make_column(nlev=30, u_scale=u_scale, v_scale=v_scale)
        f = self._scheme_fn(column, HinesParameters.default())
        args = self._args(column)
        grads = jax.grad(
            lambda *a: sum(jnp.sum(x ** 2) for x in f(*a)),
            argnums=tuple(range(len(args))),
        )(*args)
        names = ("temperature", "u_wind", "v_wind", "density")
        for name, grad in zip(names, grads):
            assert jnp.all(jnp.isfinite(grad)), (
                f"d/d{name} is not finite at u_scale={u_scale}, "
                f"v_scale={v_scale}")

    @pytest.mark.xfail(
        strict=True,
        reason="an identically calm column has no launch spectrum: the "
               "azimuthal root-mean-square winds at hines.py:253-258 are "
               "sqrt of a sum that is exactly 0 there, so displacing the "
               "column produces a jump and the secant grows as jump/eps at "
               "every rung (measured D doubling from 4.6e6 to 1.8e7 over the "
               "last two rungs). The gradients themselves stay finite.")
    def test_calm_column_has_no_two_sided_derivative(self):
        """Record the calm column as a degenerate point, not a tolerance."""
        column = _make_column(nlev=30, u_scale=0.0, v_scale=0.0)
        check_gradients(self._scheme_fn(column, HinesParameters.default()),
                        self._args(column), rtol=1e-2)
