"""Unit tests for the Lott-Miller (1997) SSO drag port.

The bit-exact Fortran-comparison harness lives in
``fortran_harness/compare_ssodrag.py`` (on the
``origin/fortran-harness-vdiff`` branch, not in this tree) and depends
on a local Fortran build that is intentionally not shipped with the
repository. These tests
are sanity checks that run as part of the regular ``pytest`` suite.
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jcm.constants import grav, rd
from jcm.physics.gravity_waves.sso import (
    SSOParameters, sso_drag,
)
from jcm.physics.gravity_waves.sso.lott_miller import _energy_conserving_cap
from jcm.testing import check_gradients


def _make_alps_column(nlev: int = 47, **overrides):
    """Mid-latitude column with Alps-like sub-grid orography.

    Returns a dict suitable for ``sso_drag(**col, config=...)``.
    """
    pressure_half = np.logspace(np.log10(10.0), np.log10(101325.0), nlev + 1)
    pressure_full = 0.5 * (pressure_half[:-1] + pressure_half[1:])
    z = np.zeros(nlev)
    zh = np.zeros(nlev + 1)
    Tprof = np.zeros(nlev)
    for k in range(nlev - 1, -1, -1):
        z_g = zh[k + 1]
        if z_g < 11000.0:
            Tprof[k] = 288.15 - 0.0065 * z_g
        elif z_g < 50000.0:
            Tprof[k] = max(220.0 - 0.001 * (z_g - 20000.0), 200.0)
        else:
            Tprof[k] = max(270.0 - 0.0035 * (z_g - 50000.0), 180.0)
        dz = (rd * Tprof[k] / grav) * np.log(pressure_half[k + 1]
                                             / pressure_half[k])
        zh[k] = zh[k + 1] + dz
        z[k] = 0.5 * (zh[k] + zh[k + 1])
    layer_mass = (pressure_half[1:] - pressure_half[:-1]) / grav
    u = 30.0 * np.exp(-((z - 10000.0) / 6000.0) ** 2) + 5.0
    v = 5.0 * np.exp(-((z - 10000.0) / 8000.0) ** 2) + 1.0
    inputs = dict(
        dt=jnp.asarray(1800.0),
        coriolis=jnp.asarray(1.0e-4),
        height_full=jnp.asarray(z),
        surface_height=jnp.asarray(500.0),
        pressure_half=jnp.asarray(pressure_half),
        pressure_full=jnp.asarray(pressure_full),
        layer_mass=jnp.asarray(layer_mass),
        temperature=jnp.asarray(Tprof),
        u_wind=jnp.asarray(u),
        v_wind=jnp.asarray(v),
        mean_orography=jnp.asarray(1500.0),
        orography_std=jnp.asarray(400.0),
        orography_slope=jnp.asarray(0.07),
        orography_anisotropy=jnp.asarray(0.4),
        orography_orientation=jnp.asarray(30.0),
        peak_elevation=jnp.asarray(2500.0),
        valley_elevation=jnp.asarray(900.0),
        land_fraction=jnp.asarray(1.0),
    )
    inputs.update({k: jnp.asarray(v) for k, v in overrides.items()})
    return inputs


class TestSSOBasic:
    """Sanity properties of the SSO scheme."""

    def test_returns_finite_tendencies(self):
        col = _make_alps_column()
        tend, _ = sso_drag(**col, config=SSOParameters.default())
        assert jnp.all(jnp.isfinite(tend.dudt))
        assert jnp.all(jnp.isfinite(tend.dvdt))
        assert jnp.all(jnp.isfinite(tend.dissip))

    def test_inactive_when_orography_below_threshold(self):
        """Activation gate: setting std-dev below ``min_orog_std`` and
        peak below ``min_peak_minus_mean_elevation`` should disable the
        scheme entirely.
        """
        col = _make_alps_column(orography_std=0.5, peak_elevation=600.0)
        tend, _ = sso_drag(**col, config=SSOParameters.default())
        np.testing.assert_array_equal(np.asarray(tend.dudt), 0.0)
        np.testing.assert_array_equal(np.asarray(tend.dvdt), 0.0)
        np.testing.assert_array_equal(np.asarray(tend.dissip), 0.0)

    def test_drag_opposes_low_level_wind(self):
        """The column-integrated zonal stress should oppose the mean wind."""
        col = _make_alps_column()
        _, state = sso_drag(**col, config=SSOParameters.default())
        assert float(state.u_stress) < 0.0   # westerly column

    def test_dissipation_non_negative(self):
        """Energy dissipation should be non-negative (KE → heat). Tolerance
        is loose because the project default precision is f32.
        """
        col = _make_alps_column()
        tend, _ = sso_drag(**col, config=SSOParameters.default())
        peak_dissip = float(jnp.max(jnp.abs(tend.dissip)))
        assert jnp.all(tend.dissip >= -1e-4 * peak_dissip)

    def test_land_fraction_scaling(self):
        """Halving land_fraction halves the tendencies."""
        config = SSOParameters.default()
        tend_full, _ = sso_drag(**_make_alps_column(), config=config)
        col_half = _make_alps_column(land_fraction=0.5)
        tend_half, _ = sso_drag(**col_half, config=config)
        np.testing.assert_allclose(np.asarray(tend_half.dudt),
                                   0.5 * np.asarray(tend_full.dudt),
                                   rtol=1e-6, atol=1e-12)


def _captured_blowup_columns():
    """Load the twelve worst columns of the t63-echam-1m CPU blow-up (#981).

    ``jcm/data/test/sso_cpu_blowup_cols.npz`` holds the SSO inputs at the
    western-Tibet / Himalaya / Andes columns where, from the last sane state of
    the cold start (step 2 of 12 minutes, float32), the CPU-compiled scheme
    returned 0.3-2.5 m/s² at a level the drag did not touch (a 32 m/s wind
    over 12 minutes) while the same function run op-by-op returned < 3e-3.
    Everything is the model's own state: the terrain fields are the T63
    descriptors of those columns, ``pressure_*`` and ``height_full`` the
    diagnostics the term reads.
    """
    from importlib import resources

    path = resources.files("jcm.data.test") / "sso_cpu_blowup_cols.npz"
    with resources.as_file(path) as f:
        d = np.load(f)
        return {k: np.asarray(d[k]) for k in d.files}


def _run_captured_columns(cols):
    """``sso_drag`` vmapped over the captured columns under ``jax.jit``.

    Mirrors how :class:`LottMillerSso` calls it (``surface_height`` and
    ``mean_orography`` are both the grid-mean orography); float32 whatever the
    process default, as the model runs it.
    """
    config = SSOParameters.default()

    def one(pf, ph, hf, T, u, v, orog, std, sig, gam, the, pic, val, fmask):
        mass = (ph[1:] - ph[:-1]) / grav
        t, _ = sso_drag(
            jnp.asarray(cols["dt"]), jnp.zeros((), jnp.float32), hf, orog,
            ph, pf, mass, T, u, v, orog, std, sig, gam, the, pic, val, fmask,
            config, nktopg=1, ntop=1)
        return t

    args = [jnp.asarray(cols[k], jnp.float32) for k in (
        "pressure_full", "pressure_half", "height_full", "temperature",
        "u_wind", "v_wind", "orog", "orostd", "orosig", "orogam", "orothe",
        "oropic", "oroval", "fmask")]
    return jax.jit(jax.vmap(one))(*args)


class TestSSOEnergyCap:
    """The drag never accelerates the wind, however XLA compiles the cap (#981).

    The cap's branch test is a difference of nearly equal squares, which is
    exactly zero in exact arithmetic at every level the drag does not touch;
    XLA:CPU evaluated it differently in each of its consumers, so the cap's
    rescale ran on a placeholder denominator and a calm level was accelerated
    by ~2 m/s². These tests pin the invariants the Fortran guarantees by
    construction rather than a compiler's rounding.
    """

    def test_captured_columns_never_accelerate_the_wind(self):
        cols = _captured_blowup_columns()
        t = _run_captured_columns(cols)
        dt = float(cols["dt"])
        u, v = cols["u_wind"], cols["v_wind"]
        speed_old = np.hypot(u, v)
        speed_new = np.hypot(u + dt * np.asarray(t.dudt),
                             v + dt * np.asarray(t.dvdt))
        # ``|u*| <= |u|`` per level, to the round-off of forming u + dt*a.
        assert np.all(speed_new <= speed_old * (1.0 + 1e-5) + 1e-4), (
            "drag accelerated the wind: worst level gained "
            f"{float(np.max(speed_new - speed_old)):.3g} m/s")
        # ... which bounds the tendency itself by twice the wind per step.
        bound = 2.0 * speed_old / dt * (1.0 + 1e-5) + 1e-9
        assert np.all(np.hypot(np.asarray(t.dudt), np.asarray(t.dvdt)) <= bound)

    def test_frictional_heating_is_non_negative_and_zero_where_no_drag(self):
        cols = _captured_blowup_columns()
        t = _run_captured_columns(cols)
        dudt, dvdt, dis = (np.asarray(x) for x in (t.dudt, t.dvdt, t.dissip))
        assert np.all(dis >= 0.0)
        untouched = (dudt == 0.0) & (dvdt == 0.0)
        assert untouched.any(), "fixture has no drag-free level to test"
        # A level the scheme leaves alone has no frictional heating: no
        # rounding residue of 0.5*(|u|² - |u + dt*0|²).
        assert np.all(dis[untouched] == 0.0)

    @staticmethod
    def _echam_cap(u, v, du, dv, dt):
        """ECHAM's ``IF (zdis < 0)`` branch (mo_ssortns.f90::orodrag), float64."""
        zust, zvst = u + dt * du, v + dt * dv
        zdis = 0.5 * (u ** 2 + v ** 2 - zust ** 2 - zvst ** 2)
        gain = zdis < 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            zred = np.sqrt((u ** 2 + v ** 2) / (zust ** 2 + zvst ** 2))
        zust2, zvst2 = zust * zred, zvst * zred
        du2 = np.where(gain, (zust2 - u) / dt, du)
        dv2 = np.where(gain, (zvst2 - v) / dt, dv)
        zdis2 = np.where(
            gain, 0.5 * (u ** 2 + v ** 2 - zust2 ** 2 - zvst2 ** 2), zdis)
        return du2, dv2, zdis2 / dt

    def test_cap_is_the_echam_branch(self):
        """Same tendencies and heating as ECHAM's branch, both sides of it."""
        rng = np.random.default_rng(981)
        n = 4000
        dt = 720.0
        u = rng.normal(0.0, 20.0, n).astype(np.float32)
        v = rng.normal(0.0, 20.0, n).astype(np.float32)
        # Increment as a multiple of the wind: < 2 slows it (no cap), > 2
        # reverses it past its own speed and gains energy (the cap acts); a
        # random crosswind part makes the speed gain depend on direction.
        alpha = rng.uniform(0.0, 3.0, n).astype(np.float32)
        cross = rng.normal(0.0, 0.5, n).astype(np.float32)
        du = (-alpha * u - cross * v) / dt
        dv = (-alpha * v + cross * u) / dt
        # A quarter of the levels carry no drag at all, as most of a column does.
        none = rng.random(n) < 0.25
        du = np.where(none, 0.0, du).astype(np.float32)
        dv = np.where(none, 0.0, dv).astype(np.float32)
        ref = self._echam_cap(*(x.astype(np.float64) for x in (u, v, du, dv)), dt)
        got = [np.asarray(x) for x in jax.jit(_energy_conserving_cap)(
            jnp.asarray(u), jnp.asarray(v), jnp.asarray(du), jnp.asarray(dv),
            jnp.float32(dt))]
        wind_scale = np.max(np.hypot(u, v)) / dt
        np.testing.assert_allclose(got[0], ref[0], rtol=1e-4, atol=1e-5 * wind_scale)
        np.testing.assert_allclose(got[1], ref[1], rtol=1e-4, atol=1e-5 * wind_scale)
        np.testing.assert_allclose(got[2], ref[2], rtol=1e-3,
                                   atol=1e-5 * wind_scale ** 2 * dt)
        # The branch really was exercised: some levels were capped back.
        capped = (got[0] != du) | (got[1] != dv)
        assert capped.sum() > 100 and (~capped).sum() > 100

    def test_cap_leaves_a_calm_drag_free_level_alone(self):
        z = jnp.zeros(3, jnp.float32)
        du, dv, dis = jax.jit(_energy_conserving_cap)(
            z, z, z, z, jnp.float32(720.0))
        for x in (du, dv, dis):
            np.testing.assert_array_equal(np.asarray(x), 0.0)


class TestSSOJaxTransforms:
    def test_jit_runs(self):
        col = _make_alps_column()
        config = SSOParameters.default()
        jitted = jax.jit(lambda **kw: sso_drag(**kw, config=config))
        tend, _ = jitted(**col)
        assert jnp.all(jnp.isfinite(tend.dudt))

    def test_vmap_over_columns(self):
        col1 = _make_alps_column()
        col2 = _make_alps_column(peak_elevation=1500.0)   # smaller peak
        col3 = _make_alps_column(peak_elevation=4000.0)   # larger peak
        keys = list(col1.keys())
        batch = {k: jnp.stack([col1[k], col2[k], col3[k]]) for k in keys}
        config = SSOParameters.default()

        def one(*args):
            t, _ = sso_drag(*args, config=config)
            return t.dudt

        out = jax.vmap(one)(*[batch[k] for k in keys])
        assert out.shape == (3, 47)
        # Taller peaks → bigger surface stress (when not Froude-blocked).
        peak1 = float(jnp.max(jnp.abs(out[0])))
        peak2 = float(jnp.max(jnp.abs(out[1])))
        assert peak1 > peak2, (
            f"larger peak should give larger drag: small={peak2}, "
            f"medium={peak1}"
        )


class TestSSOParameters:
    def test_defaults_match_echam_namelist(self):
        """Tunable knobs match echam6 defaults. Static knobs (nktopg, ntop)
        live as :func:`sso_drag` kwargs.
        """
        p = SSOParameters.default()
        for name, expected in [
            ("min_peak_minus_mean_elevation", 1.0),
            ("min_orog_std", 1.0),
            ("wave_drag_coeff", 0.2),
            ("blocked_flow_drag_coeff", 1.0),
            ("mountain_lift_coeff", 0.0),
        ]:
            np.testing.assert_allclose(float(getattr(p, name)), expected,
                                       atol=1e-6, rtol=1e-6)


class TestSSOGradients:
    """AD against a central difference through the SSO column (#820).

    The scheme is green at every operating point tried, including the two the
    ``_safe_denom`` floors and the ``argmax``​es were expected to be
    brittle at: an aquaplanet column where ``orography_std`` and every other
    orographic descriptor is exactly 0, so ``lott_miller.py:114/537/554/606/
    692`` all sit on their floors at once, and a column just above the
    activation threshold where the drag is switched on but vanishingly small.
    Worth pinning precisely because those are the states an aquaplanet
    configuration spends all of its time in.

    ``rtol=1e-2``: this module sets ``JAX_ENABLE_X64`` before importing jcm,
    but issue #729's conftest pins float32 for the suite, so the reference is
    a float32 secant at whatever rung the consistency search settles on.
    """

    KEYS = ("temperature", "u_wind", "v_wind", "orography_std",
            "orography_slope", "peak_elevation")

    AQUAPLANET = dict(
        orography_std=0.0, orography_slope=0.0, orography_anisotropy=0.0,
        peak_elevation=0.0, valley_elevation=0.0, mean_orography=0.0,
        surface_height=0.0, land_fraction=0.0,
    )

    def _scheme_fn(self, column, config):
        """Return f(T, u, v, std, slope, peak) -> the three tendencies."""
        def f(*values):
            inputs = dict(column)
            inputs.update(dict(zip(self.KEYS, values)))
            tend, _ = sso_drag(**inputs, config=config)
            return (tend.dudt, tend.dvdt, tend.dissip)

        return f

    @pytest.mark.parametrize(
        "label, overrides",
        [("alps", {}),
         ("aquaplanet", AQUAPLANET)],
    )
    @pytest.mark.parametrize("seed", [0, 3])
    def test_gradients_match_a_central_difference(self, label, overrides,
                                                  seed):
        """Every orographic regime, including an entirely flat one."""
        column = _make_alps_column(nlev=30, **overrides)
        check_gradients(
            self._scheme_fn(column, SSOParameters.default()),
            tuple(column[k] for k in self.KEYS), rtol=1e-2, seed=seed)

    @pytest.mark.parametrize(
        "label, overrides",
        [("alps", {}),
         ("aquaplanet", AQUAPLANET),
         ("near-calm wind", {"u_wind": 1.0e-4, "v_wind": 1.0e-4})],
    )
    def test_gradients_are_finite(self, label, overrides):
        """No orographic or wind regime may return a non-finite gradient."""
        column = _make_alps_column(nlev=30)
        for key, value in overrides.items():
            column[key] = jnp.full_like(column[key], value) \
                if jnp.ndim(column[key]) else jnp.asarray(value)
        f = self._scheme_fn(column, SSOParameters.default())
        args = tuple(column[k] for k in self.KEYS)
        grads = jax.grad(
            lambda *a: sum(jnp.sum(x ** 2) for x in f(*a)),
            argnums=tuple(range(len(args))),
        )(*args)
        for name, grad in zip(self.KEYS, grads):
            assert jnp.all(jnp.isfinite(grad)), (
                f"d/d{name} is not finite for {label}")

    def test_calm_column_wind_gradient_is_finite(self):
        """An exactly calm column (u = v = 0 everywhere) differentiates.

        The updated wind is zero at every level there, so the KE-conserving
        rescale's quotient must not be formed over its zero denominator
        (#663).
        """
        column = _make_alps_column(nlev=30)
        column["u_wind"] = jnp.zeros_like(column["u_wind"])
        column["v_wind"] = jnp.zeros_like(column["v_wind"])
        f = self._scheme_fn(column, SSOParameters.default())
        args = tuple(column[k] for k in self.KEYS)
        grads = jax.grad(
            lambda *a: sum(jnp.sum(x ** 2) for x in f(*a)),
            argnums=tuple(range(len(args))),
        )(*args)
        for name, grad in zip(self.KEYS, grads):
            assert jnp.all(jnp.isfinite(grad)), f"d/d{name} is not finite"

    @pytest.mark.parametrize("orography_std", [1.0e-8, 1.0e-6, 1.0e-5])
    def test_gradient_at_the_orography_floor_is_finite(self, orography_std):
        """The band around ``_MIN_OROG_STD`` differentiates, against a reference.

        Just above the floor the blocked-flow drag stops the wind in one step,
        so the updated wind is zero — the same zero denominator of the
        KE-conserving rescale that the calm column above reaches (#663).
        """
        column = _make_alps_column(nlev=30, orography_std=orography_std)
        check_gradients(
            self._scheme_fn(column, SSOParameters.default()),
            tuple(column[k] for k in self.KEYS), rtol=1e-2)
