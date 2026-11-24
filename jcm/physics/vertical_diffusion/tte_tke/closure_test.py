"""ECHAM6.3's TKE closure, pinned against the compiled ``vdiff.f90``.

The reference is ECHAM itself: ``jcm/data/test/echam_vdiff_reference/
vdiff_closure_T63L47.npz`` holds what the compiled, unmodified ``vdiff.f90``
r7492 statements return for the 384 columns of ``vdiff_T63L47.npz`` — the
closure constants (l.560-604), the PBL extension (l.737-765), the closure loop
(mixing length, Louis functions, TKE source, exchange coefficients, θ_v
variance; l.777-866), the heat, moisture and hydrometeor solve of the dry
static energy (l.877-941, 1212-1219, 1439-1457), the TKE and variance transport
solve (l.1235-1314) and the momentum solve with its dissipation (l.1347-1425) —
and what
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
from jcm.testing import check_gradients
from .closure import (
    closure_constants, geopotential_above_surface, interior_closure, louis_stability,
    lowest_level_height_above_surface, mixing_length, pbl_extension,
    surface_tke_boundary,
)
from .moist_buoyancy import interior_buoyancy_and_shear, richardson_number
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


def _closure(ref, dtype=jnp.float64, params=None, columns=slice(None)):
    # ECHAM's ztkemin = 1e-10, the parameter's default, stated: the reference
    # was computed with it.
    params = params or VDiffParameters.default(totte_min=1.0e-10)
    st = _state(ref, dtype)
    buoyancy, shear = interior_buoyancy_and_shear(st)
    ri = richardson_number(buoyancy, shear)
    f = lambda a: jnp.asarray(a, dtype)[..., columns]  # noqa: E731
    return interior_closure(
        temperature=f(st.temperature.T), qv=f(st.qv.T), qc=f(st.qc.T), qi=f(st.qi.T),
        pressure_full=f(st.pressure_full.T), pressure_half=f(st.pressure_half.T),
        geopotential=f(st.geopotential.T), buoyancy=f(buoyancy.T), shear=f(shear.T),
        ri=f(ri.T), prev_tke=f(ref["in_ptkem1"][:-1]),
        prev_thv_variance=f(ref["in_pthvvar"][:-1]),
        friction_velocity=f(ref["in_ustarm"]), coriolis=f(ref["in_coriol"]),
        dt=float(ref["in_time_step_len"]), params=params)


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

    def test_lowest_level_height_matches_geopotential_above_surface(self, ref):
        """``lowest_level_height_above_surface`` == ``geopotential_above_surface``'s
        own lowest row / g (jax-gcm#1171), on this reference's real,
        non-degenerate 384-column pressure_half profile -- an independent
        cross-check (not the land-reference fixture's own solved-for p_up,
        which only confirms the solve inverts cleanly) that the surface
        layer's new reference height is exactly the same quantity the TKE
        boundary condition (``closure.surface_tke_boundary``, #996) already
        reads off ``InteriorClosure.geopotential_above_surface[-1]``.
        """
        with _echam(ref):
            st = _state(ref)
            tv = st.temperature * (1.0 + c.vtmpc1 * st.qv - st.qc - st.qi)
            pgeom1_profile = geopotential_above_surface(
                st.geopotential.T, tv.T, st.pressure_half.T)
            want = np.asarray(pgeom1_profile[-1]) / float(c.grav)
            got = np.asarray(lowest_level_height_above_surface(
                tv[:, -1], st.pressure_half[:, -2], st.pressure_half[:, -1]))
        np.testing.assert_allclose(got, want, rtol=1e-9)

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

    def test_pbl_extension_under_lrce(self, ref):
        """ECHAM's RCE switch: the convective level alone (l.756-760).

        Pinned on the same inputs run through the compiled statements with
        ``lrce`` on, which changes ``ihpbl`` in 299 of the 384 columns; the
        mixing length reads the new extension.
        """
        assert (ref["echam_lrce_ihpbl"] != ref["echam_ihpbl"]).sum() > 100
        params = VDiffParameters.default(lrce=True)
        with _echam(ref):
            pgeom1, s = self._inputs(ref)
            out = pbl_extension(pgeom1, s, jnp.asarray(ref["in_ustarm"]),
                                jnp.asarray(ref["in_coriol"]), params)
            zmix = np.asarray(mixing_length(pgeom1, out.level, params))
        np.testing.assert_array_equal(np.asarray(out.level) + 1,
                                      ref["echam_lrce_ihpbl"])
        _close(np.asarray(out.geopotential), ref["echam_lrce_zghabl"])
        _close(zmix, ref["echam_lrce_zmix"])

    def test_lrce_off_is_the_default_and_selects_the_minimum(self):
        """Off, the extension is the higher of the two levels, as in 3D runs."""
        assert VDiffParameters.default().lrce is False
        # Dry static energy falling with height except at level 9, the only
        # one above the lowest whose s exceeds the lowest level's: the
        # convective level is 9 (about 760 m) and the dynamic height
        # 0.3·u*/|f| = 3 km is the higher.
        pgeom1 = c.grav * jnp.linspace(8000.0, 30.0, 12)
        s = c.cpd * 300.0 - 0.1 * pgeom1
        s = s.at[9].add(1.0e5)
        args = (pgeom1, s, jnp.asarray(1.0), jnp.asarray(1.0e-4))
        off = pbl_extension(*args, VDiffParameters.default())
        on = pbl_extension(*args, VDiffParameters.default(lrce=True))
        assert int(on.convective_level) == 9
        assert int(off.dynamic_level) < 9
        assert int(off.level) == int(off.dynamic_level)
        assert int(on.level) == 9

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


class TestInteriorClosure:
    """The whole closure of the interior interfaces against ``vdiff.f90``."""

    def test_pbl_extension(self, ref):
        with _echam(ref):
            out = _closure(ref)
        np.testing.assert_array_equal(np.asarray(out.pbl.level) + 1, ref["echam_ihpbl"])
        _close(out.pbl.geopotential, ref["echam_zghabl"])

    @pytest.mark.parametrize("field,name", [
        ("mixing_length", "zmix"), ("sm", "zsm"), ("sh", "zsh"),
        ("tke", "ztkevn"), ("thv_variance", "zthvvar")])
    def test_interior_fields(self, ref, field, name):
        with _echam(ref):
            out = _closure(ref)
        _close(getattr(out, field), ref[f"echam_{name}"][:ref["echam_zmix"].shape[0]])

    @pytest.mark.parametrize("field,name", [
        ("conductance_momentum", "zcfm"), ("conductance_heat", "zcfh"),
        ("conductance_tke", "zcdum"), ("conductance_thv", "zcthv")])
    def test_exchange_coefficients(self, ref, field, name):
        """ECHAM's ``zcf = cvdifts·Δt·G`` of the interior interfaces (l.852-856)."""
        scale = 1.5 * float(ref["in_time_step_len"])
        with _echam(ref):
            got = np.asarray(getattr(_closure(ref), field)) * scale
        _close(got, ref[f"echam_{name}"][:-1])

    def test_surface_tke_boundary(self, ref):
        s = lambda k: jnp.asarray(ref["sfc_in_" + k])  # noqa: E731
        with _echam(ref):
            du2 = s("du2")
            tv = s("ptm1") * (1.0 + c.vtmpc1 * s("pqm1") - s("zx"))
            # ECHAM's zcfmw = zcons12·p/T_v·C_m|U| and zchw = C_h.
            cm_u = s("zcfmw") / (float(ref["sfc_echam_zcons12"]) * s("paphm1") / tv)
            got = np.asarray(surface_tke_boundary(
                momentum_exchange=cm_u[:, None],
                heat_exchange=(s("zchw") * jnp.sqrt(du2))[:, None],
                wind_speed=jnp.sqrt(du2), thv_surface=s("thv_sfc")[:, None],
                thv_air=s("thv_air"), fraction=jnp.ones((du2.shape[0], 1)),
                pbl_geopotential=s("zghabl"), lowest_geopotential=s("pgeom1"),
                params=VDiffParameters.default()))
        _close(got, ref["sfc_echam_ztkevw"])

    def test_float32_tracks_echam(self, ref):
        """The model's precision: the PBL extension in 97 %, the rest in bulk.

        Float32 inputs move the implicit TKE update and the Louis functions
        near their branch points (``zktest = 1``, ``Ri = 0``), so the check is
        on the bulk: 99 % of every interior field within 5e-3 of ECHAM's
        (measured: 1.0e-3 for the TKE, the largest).
        """
        saved = c.physical_constants
        c.set_constants(grav=float(ref["echam_grav"]), alhc=float(ref["echam_alv"]),
                        alhs=float(ref["echam_als"]), rv=float(ref["echam_rv"]),
                        cpd=float(ref["echam_cpd"]), tmelt=float(ref["echam_tmelt"]))
        try:
            out = _closure(ref, jnp.float32)
        finally:
            c.set_constants(saved)
        agree = (np.asarray(out.pbl.level) + 1 == ref["echam_ihpbl"]).mean()
        assert agree > 0.97
        for field, name in (("mixing_length", "zmix"), ("sm", "zsm"), ("sh", "zsh"),
                            ("tke", "ztkevn")):
            got = np.asarray(getattr(out, field), np.float64)
            want = ref[f"echam_{name}"][:got.shape[0]]
            rel = np.abs(got - want) / np.maximum(np.abs(want), 1e-9)
            assert np.quantile(rel, 0.99) < 5e-3, (field, np.quantile(rel, 0.99))

    def test_single_column_matches_the_block(self, ref):
        with _echam(ref):
            block = _closure(ref)
            for j in (0, 17, 383):
                one = _closure(ref, columns=j)
                for field in ("mixing_length", "sm", "sh", "tke", "thv_variance",
                              "conductance_momentum", "conductance_tke"):
                    np.testing.assert_allclose(np.asarray(getattr(one, field)),
                                               np.asarray(getattr(block, field))[:, j],
                                               rtol=1e-13)
                assert int(one.pbl.level) == int(block.pbl.level[j])


class TestGradients:
    """The closure's constants are differentiable leaves, and live."""

    LEAVES = ("closure_h1", "closure_h2", "closure_m1", "closure_m2", "closure_m4",
              "mixing_length_asymptote", "louis_cb", "louis_cc", "totte_min")

    def test_closure_constants_reach_the_fields(self, ref):
        """AD against a central difference through the whole interior closure.

        The PBL extension is a level index, so ``zchneu`` (which moves it in
        steps) is held fixed; every other closure constant is checked.
        """
        cols = slice(0, 384, 24)
        base = VDiffParameters.default()

        def f(*leaves):
            p = base.replace(**dict(zip(self.LEAVES, leaves)))
            out = _closure(ref, jnp.float32, params=p, columns=cols)
            return out.mixing_length, out.sm, out.sh, out.tke, out.thv_variance

        leaves = tuple(getattr(base, n) for n in self.LEAVES)
        check_gradients(f, leaves, rtol=1e-2,
                        live_inputs=tuple(f"[{i}]" for i in range(len(leaves) - 1)))

    def test_surface_tke_is_differentiable_through_the_convective_branch(self):
        params = VDiffParameters.default()

        def f(convective_factor, thv_s):
            p = params.replace(surface_tke_convective_factor=convective_factor)
            return surface_tke_boundary(
                momentum_exchange=jnp.array([[0.02, 0.03]]),
                heat_exchange=jnp.array([[0.015, 0.02]]), wind_speed=jnp.array([4.0]),
                thv_surface=thv_s, thv_air=jnp.array([300.0]),
                fraction=jnp.array([[0.6, 0.4]]), pbl_geopotential=jnp.array([8000.0]),
                lowest_geopotential=jnp.array([300.0]), params=p)

        check_gradients(f, (jnp.asarray(0.2), jnp.array([[302.0, 299.5]])), rtol=1e-3,
                        live_inputs=("[0]", "[1]"))


class TestImplicitSolve:
    """The implicit solve against ``vdiff.f90``: TKE, variance, momentum, heat, moisture."""

    def _solve(self, ref):
        """Run the implicit solve (l.877-941, 1212-1219, 1235-1425, 1439-1457)."""
        from .matrix_solver import (
            _surface_air_density, back_substitute, compute_tendencies_from_solution,
            forward_sweep, setup_matrix_system, solve_tridiagonal_system,
        )
        from .vertical_diffusion_types import InterfaceConductances

        params = VDiffParameters.default()
        dt = float(ref["in_time_step_len"])
        dtf = float(params.tpfac1) * dt
        out = _closure(ref)
        st = _state(ref)
        post_tke = jnp.concatenate([out.tke, ref["in_tkevn"][None]], axis=0)
        post_var = jnp.concatenate([out.thv_variance, out.thv_variance[-1:]], axis=0)
        st = st._replace(tke=post_tke.T, thv_variance=post_var.T)
        cond = InterfaceConductances(
            momentum=out.conductance_momentum.T, heat=out.conductance_heat.T,
            tke=jnp.concatenate([out.conductance_tke, ref["in_cdum"][None] / dtf], 0).T,
            thv=jnp.concatenate([out.conductance_thv, ref["in_cduh"][None] / dtf], 0).T)
        # The momentum row's surface drag is ECHAM's cdum = cvdifts·Δt·g·ρ_s·C_m|U|.
        c_mom = ref["in_cdum"] / (dtf * _surface_air_density(st) * c.grav)
        zero = jnp.zeros(st.u.shape[0])
        ms = setup_matrix_system(st, params, cond, dt, surface_momentum=(c_mom, zero, zero))
        sol = solve_tridiagonal_system(ms.matrix_coeffs, ms.rhs_vectors, ms.variable_to_matrix)
        # Heat and moisture: the column eliminated from the top, the lowest row
        # replaced by update_surface's hand-over ztdif_new/zqdif_new (l.1101-
        # 1102, given as inputs as cdum and tkevn are), then back-substituted
        # (l.1212-1219) -- what ``couple_surface_tiles`` does with the bottom
        # value its tiles produce. jcm carries the dry static energy over c_pd.
        for ivar, imat, bottom in ((2, 1, ref["in_ztdif_new"] / c.cpd),
                                   (3, 2, ref["in_zqdif_new"])):
            cp, dp, _ = forward_sweep(ms.matrix_coeffs[:, :, 0, imat],
                                      ms.matrix_coeffs[:, :, 1, imat],
                                      ms.matrix_coeffs[:, :, 2, imat],
                                      ms.rhs_vectors[:, :, ivar])
            sol = sol.at[:, :, ivar].set(back_substitute(cp, dp, jnp.asarray(bottom)))
        tend = compute_tendencies_from_solution(sol, st, params, dt)
        tke = post_tke.T + dt * tend.tke_tendency
        var = post_var.T + dt * tend.thv_var_tendency
        return st, tend, tke, var.at[:, -1].set(var[:, -2])

    def test_tke_and_variance_transport(self, ref):
        with _echam(ref):
            _, _, tke, var = self._solve(ref)
            tke, var = np.asarray(tke.T), np.asarray(jnp.maximum(var, 1.0e-10).T)
        _close(tke, ref["echam_ptke"])
        _close(var, ref["echam_pthvvar"])

    def test_momentum_solve_and_its_dissipation(self, ref):
        """The momentum solve on interface coefficients, and the heat it returns.

        ``zdis = ½·(|u|² − |u_new|²)`` (l.1418-1421) is the kinetic energy the
        solve removes; the temperature tendency carries it as heat.
        """
        with _echam(ref):
            st, tend, _, _ = self._solve(ref)
            dt = float(ref["in_time_step_len"])
            un = st.u + dt * tend.u_tendency
            vn = st.v + dt * tend.v_tendency
            zdis = np.asarray(0.5 * (st.u ** 2 + st.v ** 2 - un ** 2 - vn ** 2)).T
            u_tend, v_tend = np.asarray(tend.u_tendency).T, np.asarray(tend.v_tendency).T
        _close(u_tend, ref["echam_pvom"])
        _close(v_tend, ref["echam_pvol"])
        _close(zdis, ref["echam_zdis"], tol=1e-10)

    def test_heat_and_moisture_solve(self, ref):
        """Dry static energy, humidity and the hydrometeors, end to end.

        ECHAM diffuses ``zcptgz = φ + c_pd·T·(1 + vtmpc2·q)`` (l.669-670, 890)
        and recovers the temperature with the dissipated kinetic energy and
        the step-start humidity's heat capacity (l.1444-1446); ``zcpten``
        (l.1447) is the dry static energy change, which jcm publishes per
        layer as ``heating_rate`` (W/m², i.e. times the layer mass). The
        hydrometeors share ``zcfh`` with no surface flux (l.933-941).
        """
        with _echam(ref):
            st, tend, _, _ = self._solve(ref)
            got = {name: np.asarray(getattr(tend, field)).T for name, field in (
                ("ptte", "temperature_tendency"), ("pqte", "qv_tendency"),
                ("pxlte", "qc_tendency"), ("pxite", "qi_tendency"))}
            heating = np.asarray(tend.heating_rate / st.air_mass).T
        for name, value in got.items():
            _close(value, ref["echam_" + name])
        _close(heating, ref["echam_zcpten"])
