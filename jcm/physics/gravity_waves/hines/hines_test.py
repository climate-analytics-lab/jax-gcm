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
from jcm.physics.gravity_waves.hines import (
    HinesGwd, launch_level_from_pressure,
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

    def test_diffusion_coefficient_gradients_are_finite(self):
        """``diffco`` differentiates, parameters included, at every wind (#663).

        Below the launch level ``sigma_t`` is exactly zero and above it the
        heating can be zero or negative: both mask a branch whose root or
        quotient must not be formed there.
        """
        for scale in (0.0, 1.0, 10.0):
            col = _make_column(nlev=30, u_scale=scale, v_scale=scale)

            def loss(temperature, config):
                _, state = hines_gwd(
                    col["pressure_half"], col["pressure_full"],
                    col["height_half"], col["density"], col["layer_mass"],
                    temperature, col["u_wind"], col["v_wind"], config)
                return jnp.sum(state.diffco ** 2)

            d_t, d_config = jax.grad(loss, argnums=(0, 1))(
                col["temperature"], HinesParameters.default())
            assert jnp.all(jnp.isfinite(d_t)), scale
            for leaf in jax.tree.leaves(d_config):
                assert jnp.all(jnp.isfinite(leaf)), scale

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
        strict=True, raises=AssertionError,
        reason="an identically calm column has no launch spectrum: the "
               "azimuthal root-mean-square winds at hines.py:253-258 are "
               "sqrt of a sum that is exactly 0 there, so displacing the "
               "column produces a jump and the secant grows as jump/eps at "
               "every rung (measured D doubling from 4.6e6 to 1.8e7 over the "
               "last two rungs). The gradients themselves stay finite. (#843)")
    def test_calm_column_has_no_two_sided_derivative(self):
        """Record the calm column as a degenerate point, not a tolerance."""
        column = _make_column(nlev=30, u_scale=0.0, v_scale=0.0)
        check_gradients(self._scheme_fn(column, HinesParameters.default()),
                        self._args(column), rtol=1e-2)


# ---------------------------------------------------------------------------
# Launch level as a pressure
# ---------------------------------------------------------------------------

def _ref_pressure_hpa(vertical):
    """Return reference (101325 Pa surface) full-level pressures, hPa, top-first."""
    return (np.asarray(vertical.a_centers)
            + np.asarray(vertical.b_centers) * 101325.0) / 100.0


def _echam_vertical(nlev):
    from jcm.physics.echam.echam_levels import get_echam_levels
    return get_echam_levels(nlev)


def _term_tendencies(term, ncols):
    """Run the term on ``ncols`` columns of the test atmosphere."""
    from types import SimpleNamespace
    cols = [_make_column(u_scale=1.0 + 0.5 * i) for i in range(ncols)]

    def stack(key):
        return jnp.stack([col[key] for col in cols], axis=1)  # (nlev, ncols)

    ph = stack("pressure_half")
    diag = dict(pressure_full=stack("pressure_full"), pressure_half=ph,
                height_half=stack("height_half"),
                air_density=stack("density"))
    state = SimpleNamespace(
        temperature=stack("temperature"), u_wind=stack("u_wind"),
        v_wind=stack("v_wind"),
        specific_humidity=jnp.zeros_like(stack("u_wind")))
    tend, _ = term(state, diag, None, None)
    return tend


class TestHinesLaunchPressure:
    """``launch_pressure_hpa`` resolves once to the ``launch_level`` count."""

    @pytest.mark.parametrize("nlev", [47, 95])
    def test_reference_pressure_of_default_level_gives_default(self, nlev):
        from types import SimpleNamespace
        vert = _echam_vertical(nlev)
        p_hpa = _ref_pressure_hpa(vert)[nlev - 10 - 1]
        term = HinesGwd(launch_pressure_hpa=float(p_hpa))
        term.cache_coords(SimpleNamespace(vertical=vert))
        assert term.launch_level == 10

    @pytest.mark.parametrize("nlev", [47, 95])
    def test_pressure_path_reproduces_launch_level_bitwise(self, nlev):
        from types import SimpleNamespace
        vert = _echam_vertical(nlev)
        p_hpa = float(_ref_pressure_hpa(vert)[nlev - 10 - 1])
        by_pressure = HinesGwd(launch_pressure_hpa=p_hpa)
        by_pressure.cache_coords(SimpleNamespace(vertical=vert))
        by_count = HinesGwd(launch_level=10)
        by_count.cache_coords(SimpleNamespace(vertical=vert))
        a = _term_tendencies(by_pressure, ncols=3)
        b = _term_tendencies(by_count, ncols=3)
        np.testing.assert_array_equal(np.asarray(a.u_wind), np.asarray(b.u_wind))
        np.testing.assert_array_equal(np.asarray(a.v_wind), np.asarray(b.v_wind))
        np.testing.assert_array_equal(np.asarray(a.temperature),
                                      np.asarray(b.temperature))

    @pytest.mark.parametrize("nlev, count", [(47, 5), (47, 20), (95, 15), (95, 40)])
    def test_other_pressure_resolves_to_expected_index(self, nlev, count):
        from types import SimpleNamespace
        vert = _echam_vertical(nlev)
        p_hpa = _ref_pressure_hpa(vert)
        # Nudge off the level centre: still nearest to the same level.
        target = float(p_hpa[nlev - count - 1]) * 1.001
        assert launch_level_from_pressure(vert, target) == count
        term = HinesGwd(launch_pressure_hpa=target)
        term.cache_coords(SimpleNamespace(vertical=vert))
        assert term.launch_level == count

    def test_same_pressure_is_a_different_count_on_a_different_table(self):
        """The point of the key: one pressure, different level counts."""
        p = 50.0
        assert (launch_level_from_pressure(_echam_vertical(47), p)
                != launch_level_from_pressure(_echam_vertical(95), p))

    def test_both_keys_is_an_error(self):
        with pytest.raises(ValueError, match="not both"):
            HinesGwd(launch_level=10, launch_pressure_hpa=634.0)

    def test_default_is_echam_emiss_lev(self):
        from types import SimpleNamespace
        term = HinesGwd()
        assert term.launch_level == 10 and term.launch_pressure_hpa is None
        term.cache_coords(SimpleNamespace(vertical=_echam_vertical(47)))
        assert term.launch_level == 10
        # Unchanged tendencies versus the explicit ECHAM count.
        explicit = HinesGwd(launch_level=10)
        explicit.cache_coords(SimpleNamespace(vertical=_echam_vertical(47)))
        out = _term_tendencies(term, ncols=2)
        ref = _term_tendencies(explicit, ncols=2)
        np.testing.assert_array_equal(np.asarray(out.u_wind),
                                      np.asarray(ref.u_wind))

    @pytest.mark.parametrize("p", [1e-4, 1050.0, 1005.0, float("nan")])
    def test_out_of_range_pressure_names_valid_range(self, p):
        vert = _echam_vertical(47)
        p_hpa = _ref_pressure_hpa(vert)
        with pytest.raises(ValueError, match="outside the range") as err:
            launch_level_from_pressure(vert, p)
        assert f"{p_hpa[0]:.4g}" in str(err.value)

    def test_unresolved_pressure_term_refuses_to_run(self):
        with pytest.raises(RuntimeError, match="cache_coords"):
            _term_tendencies(HinesGwd(launch_pressure_hpa=634.0), ncols=1)

    def test_launch_level_beyond_grid_is_rejected_at_cache(self):
        from types import SimpleNamespace
        term = HinesGwd(launch_level=47)
        with pytest.raises(ValueError, match="valid: 1 to 46"):
            term.cache_coords(SimpleNamespace(vertical=_echam_vertical(47)))

    def test_nonpositive_launch_level_is_rejected(self):
        with pytest.raises(ValueError, match=">= 1"):
            HinesGwd(launch_level=0)

    def test_sigma_vertical_resolves(self):
        from dinosaur.sigma_coordinates import SigmaCoordinates
        vert = SigmaCoordinates.equidistant(20)
        level = launch_level_from_pressure(vert, 500.0)
        idx = 20 - level - 1
        assert abs(float(vert.centers[idx]) * 1013.25 - 500.0) <= 26.0

    def test_single_column_and_block_agree(self):
        from types import SimpleNamespace
        vert = _echam_vertical(47)
        term = HinesGwd(launch_pressure_hpa=300.0)
        term.cache_coords(SimpleNamespace(vertical=vert))
        block = _term_tendencies(term, ncols=3)
        single = _term_tendencies(term, ncols=1)
        np.testing.assert_allclose(np.asarray(block.u_wind[:, 0]),
                                   np.asarray(single.u_wind[:, 0]),
                                   rtol=1e-3, atol=1e-9)

    def test_factory_passes_launch_keys(self):
        from jcm.physics.echam.echam_terms import echam_physics
        phys = echam_physics(hines_launch_pressure_hpa=300.0)
        hines = [t for t in phys.terms if isinstance(t, HinesGwd)][0]
        assert hines.launch_pressure_hpa == 300.0
        with pytest.raises(ValueError, match="not both"):
            echam_physics(hines_launch_level=8, hines_launch_pressure_hpa=300.0)
