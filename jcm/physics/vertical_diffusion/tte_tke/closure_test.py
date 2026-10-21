"""ECHAM6.3's TKE closure, pinned against the compiled ``vdiff.f90``.

The reference is ECHAM itself: ``jcm/data/test/echam_vdiff_reference/
vdiff_closure_T63L47.npz`` holds what the compiled, unmodified ``vdiff.f90``
r7492 statements return for the 384 columns of ``vdiff_T63L47.npz`` — the
closure constants (l.560-604), the PBL extension (l.737-765), the closure loop
(mixing length, Louis functions, TKE source, exchange coefficients, θ_v
variance; l.777-866), the TKE and variance transport solve (l.1235-1314) and
the momentum solve with its dissipation (l.1347-1425) — and what
``mo_surface_ocean.f90::precalc_ocean``'s TKE boundary condition (l.387-388,
422-457) returns for 192 surface cells. Provenance in the README there. jcm runs
with ECHAM's constants for the comparison, as the buoyancy tests do.

Tolerances (measured, then given a margin): float64 agrees to 7e-14 of each
field's largest value (the TKE solve; the closure fields to 3e-14, the indices
exactly); the bound is 1e-12. Float32 rounds the inputs, and the implicit TKE
update and the Louis functions amplify that near their branch points, so the
float32 comparison is of the fields' bulk (see ``test_float32_tracks_echam``).
"""

import contextlib
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jcm.constants as c
from .closure import (
    closure_constants, geopotential_above_surface, louis_stability, mixing_length,
    pbl_extension,
)
from .vertical_diffusion_types import VDiffParameters, VDiffState

_DIR = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir,
                    "data", "test", "echam_vdiff_reference")
_TOL = 1.0e-12


@pytest.fixture(scope="module")
def ref():
    out = {}
    for name in ("vdiff_T63L47.npz", "vdiff_closure_T63L47.npz"):
        with np.load(os.path.join(_DIR, name)) as z:
            out.update({k: z[k] for k in z.files})
    return out


@contextlib.contextmanager
def _echam(ref):
    """ECHAM's constants and float64 for the duration."""
    saved = c.physical_constants
    c.set_constants(grav=float(ref["echam_grav"]), alhc=float(ref["echam_alv"]),
                    alhs=float(ref["echam_als"]), rv=float(ref["echam_rv"]),
                    cpd=float(ref["echam_cpd"]), tmelt=float(ref["echam_tmelt"]))
    try:
        with jax.enable_x64(True):
            yield
    finally:
        c.set_constants(saved)


def _close(got, want, tol=_TOL):
    got, want = np.asarray(got, np.float64), np.asarray(want, np.float64)
    scale = max(float(np.max(np.abs(want))), 1e-300)
    np.testing.assert_allclose(got, want, rtol=0, atol=tol * scale)


def _geopotential(ref):
    """Model geopotential: ECHAM's above-surface field plus an orography."""
    phis = np.random.default_rng(0).uniform(0.0, 3.0e4, ref["in_ptm1"].shape[1])
    return ref["in_pgeom1_above_surface"] + phis


def _state(ref, dtype=jnp.float64):
    """Return the reference columns as a (ncol, nlev) VDiffState, top first."""
    g = lambda k: jnp.asarray(ref["in_" + k].T, dtype)  # noqa: E731
    ncol, nlev = ref["in_ptm1"].shape[::-1]
    zero = jnp.zeros((ncol, nlev), dtype)
    tiles = jnp.zeros((ncol, 3), dtype)
    return VDiffState(
        u=g("pum1"), v=g("pvm1"), temperature=g("ptm1"), qv=g("pqm1"),
        qc=g("pxlm1"), qi=g("pxim1"), cloud_fraction=g("paclc"),
        pressure_full=g("papm1"), pressure_half=g("paphm1"),
        geopotential=jnp.asarray(_geopotential(ref).T, dtype),
        air_mass=jnp.diff(g("paphm1"), axis=1) / c.grav,
        surface_temperature=tiles, surface_fraction=tiles, roughness_length=tiles,
        roughness_heat=tiles, surface_wetness=tiles, height_full=zero,
        height_half=jnp.zeros((ncol, nlev + 1), dtype), tke=g("ptkem1"),
        thv_variance=g("pthvvar"), ocean_u=jnp.zeros(ncol, dtype),
        ocean_v=jnp.zeros(ncol, dtype))


class TestReferenceCoverage:
    """The reference set exercises every branch the closure turns on."""

    def test_branches(self, ref):
        ri, tke = ref["echam_zri"], ref["echam_ztkevn"][:-1]
        assert 0.01 < (ri < 0).mean() < 0.5               # Louis unstable branch
        assert 0.05 < (tke <= 1.0e-10).mean() < 0.9       # zktest <= 1 branch
        c_lev, d_lev = ref["echam_ihpblc"], ref["echam_ihpbld"]
        assert (c_lev < d_lev).any() and (d_lev < c_lev).any()  # both set ihpbl
        k = np.arange(ri.shape[0])[:, None] + 1
        inside = k >= ref["echam_ihpbl"][None, :]
        assert inside.any() and (~inside).any()            # both mixing lengths
        assert (np.abs(ref["in_coriol"]) < 5.0e-5).any()   # the zepcor floor
        wst = ref["sfc_echam_zwstw"]
        assert (wst > 1e-10).any() and (wst <= 1e-10).any()
        assert np.isclose(ref["sfc_echam_zstabf"], 3.0 * ref["sfc_echam_zustf"]).any()


class TestStabilityFunctions:
    """The closure constants and Louis's functions against ``vdiff.f90``."""

    def test_closure_constants(self, ref):
        with jax.enable_x64(True):
            k = closure_constants(VDiffParameters.default())
        assert float(k.shn) == pytest.approx(float(ref["echam_zshn"]), rel=1e-15)
        assert float(k.smn) == pytest.approx(float(ref["echam_zsmn"]), rel=1e-15)
        assert float(k.da1) == pytest.approx(float(ref["echam_zda1"]), rel=1e-15)
        assert 1.0 / float(k.da1) == pytest.approx(0.16646, abs=1e-5)

    def test_louis_functions_of_echams_ri_and_length(self, ref):
        """``zsm``, ``zsh`` from ECHAM's own ``zri`` and ``zmix`` (l.819-832)."""
        with _echam(ref):
            sm, sh = louis_stability(
                jnp.asarray(ref["echam_zri"]), jnp.asarray(ref["echam_zmix"]),
                jnp.asarray(ref["in_pgeom1_above_surface"]), VDiffParameters.default())
            sm, sh = np.asarray(sm), np.asarray(sh)
        _close(sm, ref["echam_zsm"])
        _close(sh, ref["echam_zsh"])

    def test_neutral_stability_is_the_neutral_coefficients(self):
        params = VDiffParameters.default()
        k = closure_constants(params)
        pgeom1 = jnp.array([3000.0, 2000.0, 1000.0])
        zmix = jnp.array([40.0, 30.0])
        sm, sh = louis_stability(jnp.zeros(2), zmix, pgeom1, params)
        np.testing.assert_allclose(sm, k.smn * zmix, rtol=1e-6)
        np.testing.assert_allclose(sh, k.shn * zmix, rtol=1e-6)
        # continuous across Ri = 0, enhanced when unstable, damped when stable
        for eps in (1e-6, 1e-3):
            up = louis_stability(jnp.full(2, -eps), zmix, pgeom1, params)
            dn = louis_stability(jnp.full(2, eps), zmix, pgeom1, params)
            assert jnp.all(up[0] > sm) and jnp.all(dn[0] < sm)
        np.testing.assert_allclose(
            louis_stability(jnp.full(2, -1e-9), zmix, pgeom1, params)[1], sh, rtol=1e-6)

    def test_broadcasting_native(self, ref):
        """A (nlev,) column and a (nlev, nlon, nlat) grid give the block's values."""
        params = VDiffParameters.default()
        ri = jnp.asarray(ref["echam_zri"], jnp.float32)
        zmix = jnp.asarray(ref["echam_zmix"], jnp.float32)
        pgeom1 = jnp.asarray(ref["in_pgeom1_above_surface"], jnp.float32)
        block = louis_stability(ri, zmix, pgeom1, params)
        one = louis_stability(ri[:, 7], zmix[:, 7], pgeom1[:, 7], params)
        grid = louis_stability(ri.reshape(-1, 24, 16), zmix.reshape(-1, 24, 16),
                               pgeom1.reshape(-1, 24, 16), params)
        for b, o, g in zip(block, one, grid):
            np.testing.assert_array_equal(np.asarray(o), np.asarray(b)[:, 7])
            np.testing.assert_array_equal(np.asarray(g).reshape(b.shape), np.asarray(b))


class TestMixingLengthAndPblExtension:
    """The PBL extension and the mixing length against ``vdiff.f90``."""

    def _inputs(self, ref):
        pgeom1 = jnp.asarray(ref["in_pgeom1_above_surface"])
        t, q = jnp.asarray(ref["in_ptm1"]), jnp.asarray(ref["in_pqm1"])
        s = pgeom1 + t * c.cpd * (1.0 + c.vtmpc2 * q)                 # zcptgz
        return pgeom1, s

    def test_geopotential_above_the_surface(self, ref):
        with _echam(ref):
            st = _state(ref)
            tv = st.temperature * (1.0 + c.vtmpc1 * st.qv - st.qc - st.qi)
            got = np.asarray(geopotential_above_surface(
                st.geopotential.T, tv.T, st.pressure_half.T))
        _close(got, ref["in_pgeom1_above_surface"])

    def test_pbl_extension(self, ref):
        with _echam(ref):
            pgeom1, s = self._inputs(ref)
            out = pbl_extension(pgeom1, s, jnp.asarray(ref["in_ustarm"]),
                                jnp.asarray(ref["in_coriol"]), VDiffParameters.default())
            out = [np.asarray(x) for x in out]
        level, convective, dynamic, geopotential, height = out
        np.testing.assert_array_equal(convective + 1, ref["echam_ihpblc"])
        np.testing.assert_array_equal(dynamic + 1, ref["echam_ihpbld"])
        np.testing.assert_array_equal(level + 1, ref["echam_ihpbl"])
        _close(geopotential, ref["echam_zghabl"])
        _close(height, ref["echam_zhdyn"])

    def test_mixing_length(self, ref):
        with _echam(ref):
            got = np.asarray(mixing_length(
                jnp.asarray(ref["in_pgeom1_above_surface"]),
                jnp.asarray(ref["echam_ihpbl"].astype(int) - 1),
                VDiffParameters.default()))
        _close(got, ref["echam_zmix"])

    def test_blackadar_inside_and_holtslag_boville_above(self):
        """λ = clam inside the PBL extension; above it λ decays towards 1 m."""
        params = VDiffParameters.default()
        g = c.grav
        z = jnp.linspace(10000.0, 25.0, 30)
        pgeom1 = g * z
        level = jnp.asarray(20)
        zmix = mixing_length(pgeom1, level, params)
        zi = 0.5 * (z[:-1] + z[1:])
        kz = c.karman_const * zi
        inside = jnp.arange(29) >= 20
        np.testing.assert_allclose(zmix[inside], (kz / (1 + kz / 150.0))[inside], rtol=1e-5)
        lam = 1.0 + 149.0 * jnp.exp(1.0 - z[:-1] / z[20])
        np.testing.assert_allclose(zmix[~inside], (kz / (1 + kz / lam))[~inside], rtol=1e-5)
        # λ decays with height above the extension: at 10 km l is far below
        # the λ = clam the same height would have inside it.
        assert float(zmix[0]) < 0.3 * float(kz[0] / (1 + kz[0] / 150.0))

    def test_pbl_extension_takes_the_higher_of_the_two(self):
        params = VDiffParameters.default()
        g = c.grav
        z = jnp.linspace(5000.0, 50.0, 20)
        pgeom1 = g * z
        # a well-mixed layer below 2 km capped by an inversion
        s = jnp.where(z < 2000.0, 3.0e5, 3.0e5 + 10.0 * (z - 2000.0) + 1.0)
        out = pbl_extension(pgeom1, s, jnp.asarray(0.3), jnp.asarray(1.0e-4), params)
        assert float(out.dynamic_height) == pytest.approx(900.0, rel=1e-5)
        assert float(z[out.convective_level]) > 2000.0
        assert int(out.level) == int(out.convective_level)
        # strong wind: the dynamic height is the higher one
        out = pbl_extension(pgeom1, s, jnp.asarray(2.0), jnp.asarray(1.0e-4), params)
        assert int(out.level) == int(out.dynamic_level) < int(out.convective_level)
        # nothing above the lowest level is warmer: ihpblc is the lowest level
        out = pbl_extension(pgeom1, jnp.full(20, 3.0e5), jnp.asarray(0.3),
                            jnp.asarray(1.0e-4), params)
        assert int(out.convective_level) == 19
        assert float(out.geopotential) == pytest.approx(min(50000.0, float(pgeom1[out.level])))

    def test_a_grid_of_columns(self, ref):
        """A (nlev, nlon, nlat) host gives the (nlev, ncols) block's values."""
        params = VDiffParameters.default()
        pgeom1 = jnp.asarray(ref["in_pgeom1_above_surface"], jnp.float32)
        level = jnp.asarray(ref["echam_ihpbl"].astype(int) - 1)
        block = mixing_length(pgeom1, level, params)
        grid = mixing_length(pgeom1.reshape(pgeom1.shape[0], 24, 16),
                             level.reshape(24, 16), params)
        np.testing.assert_array_equal(np.asarray(grid).reshape(block.shape),
                                      np.asarray(block))
