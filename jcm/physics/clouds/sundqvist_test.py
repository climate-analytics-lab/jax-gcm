"""Tests of the ECHAM cloud cover (``sundqvist.py``, ``mo_cover.f90::cover``).

Value tests compare against :func:`_echam_cover_reference`, a NumPy
transcription of ``mo_cover.f90`` l.164-252 written as the Fortran loops run
(the upward inversion scan with ECHAM's ``FSEL`` update rule, then the level
loop), at designed points: relative humidity below, at and above the critical
value, exact saturation, each inversion case, each surface and convective
type, and cold cells with and without ice. The comparison against the
unmodified Fortran itself lives in ``echam_fortran_reference_test.py``.

Derivative tests follow ``docs/source/design/surrogate_gradients.md``: for
each surrogate, ``check_surrogate_gradient`` on the wrapped function,
``check_gradients`` on the surrogate alone, and a bound on the distance
between surrogate and reference.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import jcm.constants as c
from jcm.physics import thermodynamics
from jcm.physics.clouds import echam_saturation as es
from jcm.physics.clouds.sundqvist import (
    CloudParameters,
    SundqvistCloudFraction,
    _cover_exact,
    _cover_surrogate,
    _zsat_exact,
    _zsat_surrogate,
    _inversion_lapse,
    calculate_cloud_fraction,
    cover_from_b0,
    cover_saturation_specific_humidity,
    critical_relative_humidity,
    saturation_specific_humidity,
    stratocumulus_saturation_factor,
)
from jcm.physics.echam.echam_levels import get_echam_levels
from jcm.physics.surrogate_gradient import with_surrogate_gradient
from jcm.testing import check_gradients, check_surrogate_gradient

L47_RANGE = (39, 44)   # ECHAM's jbmin/jbmax = 40/45, 0-based


# ---------------------------------------------------------------------------
# Column construction and the NumPy transcription of mo_cover.f90
# ---------------------------------------------------------------------------

def _l47_pressures(ps=101325.0):
    hc = get_echam_levels(47)
    a = np.asarray(hc.a_boundaries, np.float64)
    b = np.asarray(hc.b_boundaries, np.float64)
    ph = a + b * ps
    return ph, 0.5 * (ph[:-1] + ph[1:])


def _geopotential(temperature, p_half, p_full):
    """Hydrostatic full-level geopotential above the surface [m2 s-2]."""
    t = np.asarray(temperature, np.float64)
    n = t.shape[0]
    phi = np.zeros(n)
    phi_half_below = 0.0
    for k in range(n - 1, -1, -1):
        phi[k] = phi_half_below + c.rd * t[k] * np.log(p_half[k + 1] / p_full[k])
        phi_half_below += c.rd * t[k] * np.log(
            p_half[k + 1] / max(p_half[k], 1e-3))
    return phi


def _l47_column(inversion_level=None, inversion_jump=3.0, ps=101325.0):
    """Build an L47 column: 288 K surface, 6.5 K/km troposphere, 210 K above.

    ``inversion_level`` (0-based) is made ``inversion_jump`` K colder than the
    level above it, so its ``zdtdz`` is positive: an inversion on top of it.
    """
    ph, pf = _l47_pressures(ps)
    z_est = 7000.0 * np.log(ps / pf)
    t = np.maximum(288.0 - 6.5e-3 * z_est, 210.0)
    if inversion_level is not None:
        t[inversion_level] = t[inversion_level - 1] - inversion_jump
        t[inversion_level + 1:] = (
            t[inversion_level]
            + 6.5e-3 * (z_est[inversion_level] - z_est[inversion_level + 1:]))
    return t, ph, pf, _geopotential(t, ph, pf)


def _es_numpy(t, ice):
    """Evaluate the selected formula, written out (``echam_saturation``)."""
    if es.SATURATION_FORMULA == "sonntag":
        def fit(tt, a):
            return np.exp(a[0] / tt + a[1] + a[2] * 0.01 * tt
                          + a[3] * tt * tt * 1e-5 + a[4] * np.log(tt))
        return np.where(ice, fit(t, es.ICE_COEFFICIENTS),
                        fit(t, es.WATER_COEFFICIENTS))
    def tetens(tt, ab):
        return 610.78 * np.exp(ab[0] * (tt - c.tmelt) / (tt - c.tmelt + ab[1]))
    return np.where(ice, tetens(t, es.TETENS_ICE), tetens(t, es.TETENS_WATER))


def _qs_numpy(t, qi, p, csecfrl=5e-6, cthomi=238.15):
    ice = (t < cthomi) | ((t < c.tmelt) & (qi > csecfrl))
    e = _es_numpy(t, ice)
    x = np.minimum(e * c.rd / c.rv / p, 0.5)
    return x / (1.0 - (c.rv / c.rd - 1.0) * x)


def _echam_cover_reference(t, q, qi, pf, ps, geo, row, jbmin, jbmax,
                           enhance=True):
    """``mo_cover.f90`` l.164-252 for one column, as the Fortran loops run.

    Indices are 0-based and top-first; ECHAM's "no level found" value
    ``knvb = 1`` is represented by ``-1``.
    """
    g, cpd = c.grav, c.cpd
    nlev = t.shape[0]
    dtmin = -row["cinv"] * g / cpd
    knvb = -1
    if enhance:
        for jk in range(nlev - 1, jbmin - 1, -1):
            dtdz = min(0.0, (t[jk - 1] - t[jk]) * g / (geo[jk - 1] - geo[jk]))
            if dtmin - dtdz < 0.0:        # FSEL(dtmin - dtdz, keep, new)
                knvb = jk
            dtmin = max(dtdz, dtmin)
    qs = _qs_numpy(t, qi, pf, csecfrl=row["csecfrl"])
    cover = np.zeros(nlev)
    for jk in range(nlev):
        rhc = row["crt"] + (row["crs"] - row["crt"]) * np.exp(
            1.0 - (ps / pf[jk]) ** row["nex"])
        zsat = 1.0
        jb = knvb
        if jbmin <= jb <= jbmax and jk in (jb, jb + row["nadd"]):
            dtdz = (t[jb - 1] - t[jb]) * g / (geo[jb - 1] - geo[jb])
            zsat = min(1.0, row["csatsc"] + max(0.0, -dtdz * cpd / g))
        b0 = (q[jk] / (qs[jk] * zsat) - rhc) / (1.0 - rhc)
        cover[jk] = 1.0 - np.sqrt(1.0 - min(max(b0, 0.0), 1.0))
    return cover, knvb


def _row(params):
    return dict(crt=float(params.crt), crs=float(params.crs),
                nex=float(params.nex), csatsc=float(params.csatsc),
                cinv=float(params.cinv), csecfrl=float(params.csecfrl),
                nadd=int(params.nadd))


def _jcm_cover(t, q, qi, pf, ps, geo, params, enhance=True,
               inversion_range=L47_RANGE):
    cf, rh = calculate_cloud_fraction(
        jnp.asarray(t), jnp.asarray(q), jnp.asarray(qi), jnp.asarray(pf),
        jnp.asarray(ps), jnp.asarray(geo), params, inversion_range,
        jnp.asarray(enhance))
    return np.asarray(cf), np.asarray(rh)


@pytest.fixture(autouse=True)
def _x64():
    """Value comparisons against the float64 transcription run in x64."""
    previous = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", previous)


@pytest.fixture(params=es.SATURATION_FORMULAS)
def formula(request, monkeypatch):
    """Run a value test under ECHAM's formula and the Tetens test utility."""
    monkeypatch.setattr(es, "SATURATION_FORMULA", request.param)
    return request.param


# ---------------------------------------------------------------------------
# Closure values
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("formula")
class TestClosure:
    """``rhc`` and ``cover = 1 - sqrt(1 - clip(b0, 0, 1))`` (l.233, 248-251)."""

    def test_critical_rh_profile(self):
        params = CloudParameters.default()
        p = jnp.array([100000.0, 95000.0, 70000.0, 50000.0, 20000.0])
        rhc = critical_relative_humidity(p, jnp.asarray(100000.0), params)
        expected = 0.75 + (0.975 - 0.75) * np.exp(
            1.0 - (100000.0 / np.asarray(p)) ** 2)
        np.testing.assert_allclose(np.asarray(rhc), expected, rtol=1e-6)
        assert float(rhc[0]) == pytest.approx(0.975)

    def test_rh_below_critical_is_exactly_clear(self):
        """``b0 < 0`` gives a cover of exactly 0, not a small positive one."""
        params = CloudParameters.default()
        t, ph, pf, geo = _l47_column()
        qi = np.zeros_like(t)
        qs = _qs_numpy(t, qi, pf)
        rhc = np.asarray(critical_relative_humidity(
            jnp.asarray(pf), jnp.asarray(ph[-1]), params))
        cf, _ = _jcm_cover(t, (rhc - 0.01) * qs, qi, pf, ph[-1], geo,
                           params, enhance=False)
        assert np.all(cf == 0.0), cf

    def test_rh_at_critical(self):
        """``b0 = 0``: exactly 0; ``q = rhc·qs`` lands within rounding of it."""
        assert float(cover_from_b0(jnp.array(0.0), 0.02)) == 0.0
        params = CloudParameters.default()
        t, ph, pf, geo = _l47_column()
        qi = np.zeros_like(t)
        rhc = np.asarray(critical_relative_humidity(
            jnp.asarray(pf), jnp.asarray(ph[-1]), params))
        cf, _ = _jcm_cover(t, rhc * _qs_numpy(t, qi, pf), qi, pf, ph[-1],
                           geo, params, enhance=False)
        assert np.all(cf < 1e-13), cf

    def test_rh_above_critical_follows_the_closure(self):
        params = CloudParameters.default()
        t, ph, pf, geo = _l47_column()
        qi = np.zeros_like(t)
        q = 0.9 * _qs_numpy(t, qi, pf)
        cf, _ = _jcm_cover(t, q, qi, pf, ph[-1], geo, params, enhance=False)
        ref, _ = _echam_cover_reference(t, q, qi, pf, ph[-1], geo,
                                        _row(params), *L47_RANGE,
                                        enhance=False)
        np.testing.assert_allclose(cf, ref, rtol=1e-10, atol=1e-12)
        # rhc exceeds 0.9 in the lowest levels (crs = 0.975 at the surface)
        assert np.all(cf[:35] > 0.0) and np.all(cf[42:] == 0.0)
        assert np.all(cf < 1.0)

    @pytest.mark.parametrize("rh", [1.0 + 1e-9, 1.3])
    def test_saturation_is_exactly_overcast(self, rh):
        """``b0 >= 1`` gives exactly 1 (``b0 = 1`` itself: next test)."""
        params = CloudParameters.default()
        t, ph, pf, geo = _l47_column()
        qi = np.zeros_like(t)
        cf, _ = _jcm_cover(t, rh * _qs_numpy(t, qi, pf), qi, pf, ph[-1], geo,
                           params, enhance=False)
        assert np.all(cf == 1.0), cf

    def test_exact_saturation_point(self):
        """``b0 = 1`` exactly: value 1 and a finite surrogate slope."""
        x = jnp.array([1.0])
        assert float(cover_from_b0(x, 0.02)[0]) == 1.0
        slope = jax.grad(lambda v: cover_from_b0(v, 0.02).sum())(x)
        assert np.isfinite(float(slope[0])) and float(slope[0]) > 0.0

    def test_no_stratospheric_cutoff(self):
        """ECHAM computes the cover at every level (ktdia = 1).

        A supersaturated layer at 5 hPa is overcast, as in ECHAM; nothing
        forces the cover to zero above a pressure.
        """
        params = CloudParameters.default()
        t = np.array([200.0, 205.0, 250.0, 280.0])
        pf = np.array([500.0, 5000.0, 50000.0, 90000.0])
        geo = np.array([3.5e5, 2.0e5, 5.5e4, 9.0e3])
        qi = np.zeros(4)
        q = 1.2 * _qs_numpy(t, qi, pf)
        cf, _ = _jcm_cover(t, q, qi, pf, 100000.0, geo, params,
                           enhance=False, inversion_range=(1, 2))
        assert np.all(cf == 1.0)


@pytest.mark.usefixtures("formula")
class TestPhase:
    """The cover's ``lo2`` saturation (l.215-224)."""

    def _cover_and_rh(self, t, qi, rh_water, params=None):
        """Return the cover's ``q/qs`` for ``q = rh_water·qs_water``."""
        params = params or CloudParameters.default()
        pf = np.full(3, 50000.0)
        geo = np.array([2e4, 1e4, 0.0])
        q = rh_water * np.asarray(es.qsat_from_es(
            es.es_water(jnp.full(3, t)), jnp.asarray(pf)))
        return _jcm_cover(np.full(3, t), q, np.full(3, qi), pf, 100000.0,
                          geo, params, enhance=False,
                          inversion_range=(1, 2))

    @staticmethod
    def _ratio(t):
        p = jnp.asarray(50000.0)
        return float(es.qsat_from_es(es.es_water(t), p)
                     / es.qsat_from_es(es.es_ice(t), p))

    def test_cold_cell_uses_ice(self):
        """Below cthomi the ice table applies with or without ice."""
        _, rh = self._cover_and_rh(230.0, 0.0, 0.8)
        assert rh[0] == pytest.approx(0.8 * self._ratio(230.0), rel=1e-9)

    def test_mixed_phase_without_ice_uses_water(self):
        _, rh = self._cover_and_rh(255.0, 0.0, 0.8)
        assert rh[0] == pytest.approx(0.8, rel=1e-9)

    def test_mixed_phase_ice_threshold_is_strict(self):
        """``xi > csecfrl`` is strict: at the threshold, water."""
        _, rh_at = self._cover_and_rh(255.0, 5.0e-6, 0.8)
        _, rh_above = self._cover_and_rh(255.0, 5.1e-6, 0.8)
        assert rh_at[0] == pytest.approx(0.8, rel=1e-9)
        assert rh_above[0] == pytest.approx(0.8 * self._ratio(255.0),
                                            rel=1e-9)

    def test_csecfrl_comes_from_the_resolution_defaults(self):
        """At T127 ECHAM's csecfrl is 1e-5, so 6e-6 of ice stays water."""
        _, rh63 = self._cover_and_rh(255.0, 6.0e-6, 0.8)
        _, rh127 = self._cover_and_rh(
            255.0, 6.0e-6, 0.8, CloudParameters.default(truncation=127))
        assert rh63[0] == pytest.approx(0.8 * self._ratio(255.0), rel=1e-9)
        assert rh127[0] == pytest.approx(0.8, rel=1e-9)

    def test_melting_point_uses_water(self):
        _, rh = self._cover_and_rh(c.tmelt, 1.0e-3, 0.8)
        assert rh[0] == pytest.approx(0.8, rel=1e-9)


# ---------------------------------------------------------------------------
# Stratocumulus enhancement
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("formula")
class TestInversion:
    """ECHAM's inversion search and ``zsat`` (l.179-207, 234-247)."""

    def _case(self, t, ph, pf, geo, params=None, enhance=True, rh=0.8):
        params = params or CloudParameters.default()
        qi = np.zeros_like(t)
        q = rh * _qs_numpy(t, qi, pf)
        cf, _ = _jcm_cover(t, q, qi, pf, ph[-1], geo, params, enhance=enhance)
        ref, knvb = _echam_cover_reference(t, q, qi, pf, ph[-1], geo,
                                           _row(params), *L47_RANGE,
                                           enhance=enhance)
        cf_plain, _ = _jcm_cover(t, q, qi, pf, ph[-1], geo, params,
                                 enhance=False)
        np.testing.assert_allclose(cf, ref, rtol=1e-10, atol=1e-12)
        return cf, cf_plain, knvb

    def test_inversion_inside_range_enhances_that_level_only(self):
        t, ph, pf, geo = _l47_column(inversion_level=42)
        cf, cf_plain, knvb = self._case(t, ph, pf, geo)
        assert knvb == 42
        changed = np.nonzero(cf != cf_plain)[0]
        assert list(changed) == [42]
        zsat = np.asarray(stratocumulus_saturation_factor(
            jnp.asarray(t), jnp.asarray(geo), CloudParameters.default(),
            L47_RANGE))
        assert zsat[42] == pytest.approx(0.7)          # zgam = 0 at an inversion
        assert np.all(np.delete(zsat, 42) == 1.0)

    def test_most_stable_level_below_jbmax_blocks_enhancement(self):
        """A surface-based inversion is chosen and suppresses enhancement.

        ECHAM searches down to the lowest level, so a near-surface inversion
        wins the search and, lying below ``jbmax``, gives no enhancement at
        all, even though a weaker inversion exists inside the range.
        """
        t, ph, pf, geo = _l47_column(inversion_level=41, inversion_jump=1.0)
        t[46] = t[45] - 4.0                   # a stronger inversion at the ground
        geo = _geopotential(t, ph, pf)
        cf, cf_plain, knvb = self._case(t, ph, pf, geo)
        assert knvb in (41, 46)
        # both inversions clip to 0: the tie goes to the lowest level
        assert knvb == 46
        np.testing.assert_array_equal(cf, cf_plain)

    def test_no_level_stable_enough_gives_no_enhancement(self):
        """A dry-adiabatic boundary layer is more unstable than -cinv·g/cpd."""
        t, ph, pf, geo = _l47_column()
        z = geo / c.grav
        bl = z < 2500.0
        t[bl] = 300.0 - 9.8e-3 * z[bl]
        geo = _geopotential(t, ph, pf)
        cf, cf_plain, knvb = self._case(t, ph, pf, geo)
        assert knvb == -1
        np.testing.assert_array_equal(cf, cf_plain)

    def test_weakly_stable_level_uses_zgam(self):
        """A stable but non-inverted layer gets ``csatsc + zgam``."""
        t, ph, pf, geo = _l47_column()
        # make the 42/41 lapse -1 K/km (more stable than the -6.5 elsewhere)
        dz = (geo[41] - geo[42]) / c.grav
        t[:42] = t[:42] - (t[41] - (t[42] - 1.0e-3 * dz))
        geo = _geopotential(t, ph, pf)
        cf, cf_plain, knvb = self._case(t, ph, pf, geo)
        lapse = (t[41] - t[42]) * c.grav / (geo[41] - geo[42])
        zsat = np.asarray(stratocumulus_saturation_factor(
            jnp.asarray(t), jnp.asarray(geo), CloudParameters.default(),
            L47_RANGE))
        assert knvb == 42
        assert zsat[42] == pytest.approx(0.7 - lapse * c.cpd / c.grav,
                                         rel=1e-9)

    def test_nadd_enhances_the_level_below(self):
        """T31's ``nadd = 1`` enhances the chosen level and the one below."""
        params = CloudParameters.default(truncation=31)
        t, ph, pf, geo = _l47_column(inversion_level=42)
        zsat = np.asarray(stratocumulus_saturation_factor(
            jnp.asarray(t), jnp.asarray(geo), params, L47_RANGE))
        assert list(np.nonzero(zsat < 1.0)[0]) == [42, 43]
        assert zsat[43] == zsat[42] == pytest.approx(float(params.csatsc))
        self._case(t, ph, pf, geo, params=params)

    def test_gate_off_gives_no_enhancement(self):
        t, ph, pf, geo = _l47_column(inversion_level=42)
        cf, cf_plain, _ = self._case(t, ph, pf, geo, enhance=False)
        np.testing.assert_array_equal(cf, cf_plain)


# ---------------------------------------------------------------------------
# The term: surface types, convective types, layouts
# ---------------------------------------------------------------------------

class _Forcing:
    def __init__(self, sice):
        self.sice_am = sice


class _Terrain:
    def __init__(self, fmask):
        self.fmask = fmask


def _term_inputs(ncols, inversion=True):
    from jcm.physics_interface import PhysicsState
    t, ph, pf, geo = _l47_column(inversion_level=42 if inversion else None)
    qi = np.zeros_like(t)
    q = 0.8 * _qs_numpy(t, qi, pf)
    tile = lambda a: jnp.asarray(np.repeat(a[:, None], ncols, axis=1))  # noqa: E731
    state = PhysicsState(
        u_wind=tile(np.zeros(47)), v_wind=tile(np.zeros(47)),
        temperature=tile(t), specific_humidity=tile(q),
        geopotential=tile(geo),
        normalized_surface_pressure=jnp.full((ncols,), ph[-1] / c.p0),
        tracers={"qc": tile(np.zeros(47)), "qi": tile(qi)})
    diags = {"pressure_full": tile(pf),
             "surface_pressure": jnp.full((ncols,), ph[-1])}
    return state, diags


def _cached_term(params=None, **kw):
    from jcm.utils import get_coords
    term = SundqvistCloudFraction(params, **kw)
    term.cache_coords(get_coords(get_echam_levels(47), spectral_truncation=63))
    return term


class TestTerm:

    def test_requires_cache_coords(self):
        state, diags = _term_inputs(1)
        with pytest.raises(RuntimeError, match="cache_coords"):
            SundqvistCloudFraction()(state, diags, _Forcing(None),
                                     _Terrain(jnp.zeros(1)))

    def test_inversion_levels_cached(self):
        assert _cached_term()._inversion_range == L47_RANGE

    def test_surface_and_convective_gate(self):
        """Enhanced only over ice-free ocean without convection (l.181)."""
        state, diags = _term_inputs(5)
        #          ocean  land  sea-ice  ocean+ktype1  ocean(seaice 1e-13)
        fmask = jnp.array([0.0, 1.0, 0.0, 0.0, 0.0])
        sice = jnp.array([0.0, 0.0, 0.2, 0.0, 1e-13])
        # the previous step's convection carry: only ``ktype`` is read
        conv = type("Convection", (), {"ktype": jnp.array([0, 0, 0, 1, 0])})()
        diags = {**diags, "convection": conv}
        _, out = _cached_term()(state, diags, _Forcing(sice), _Terrain(fmask))
        cf = np.asarray(out["clouds"].cloud_fraction)
        enhanced = cf[42] > cf[42, 1]
        assert list(enhanced) == [True, False, False, False, True]
        assert np.all(cf[:42] == cf[:42, :1]) and np.all(cf[43:] == cf[43:, :1])

    def test_column_and_block_agree(self):
        """Broadcasting-native: (nlev,) and (nlev, ncols) give the same cover."""
        t, ph, pf, geo = _l47_column(inversion_level=42)
        qi = np.zeros_like(t)
        q = 0.8 * _qs_numpy(t, qi, pf)
        params = CloudParameters.default()
        col, _ = _jcm_cover(t, q, qi, pf, ph[-1], geo, params)
        scale = np.array([1.0, 0.95, 1.05])
        block_args = [np.stack([a * s for s in scale], axis=1)
                      for a in (t, q, qi, pf, geo)]
        cf, _ = calculate_cloud_fraction(
            *(jnp.asarray(a) for a in block_args[:4]),
            jnp.asarray(ph[-1] * scale), jnp.asarray(block_args[4]),
            params, L47_RANGE, jnp.ones(3, bool))
        for k in range(3):
            one, _ = _jcm_cover(*(a[:, k] for a in block_args[:4]),
                                ph[-1] * scale[k], block_args[4][:, k], params)
            np.testing.assert_array_equal(np.asarray(cf)[:, k], one)
        np.testing.assert_array_equal(np.asarray(cf)[:, 0], col)
        grid = [a.reshape(47, 1, 3) for a in block_args]
        cf3, _ = calculate_cloud_fraction(
            *(jnp.asarray(a) for a in grid[:4]),
            jnp.asarray((ph[-1] * scale).reshape(1, 3)), jnp.asarray(grid[4]),
            params, L47_RANGE, jnp.ones((1, 3), bool))
        np.testing.assert_array_equal(np.asarray(cf3)[:, 0, :],
                                      np.asarray(cf))


# ---------------------------------------------------------------------------
# Surrogates
# ---------------------------------------------------------------------------

class TestCoverSurrogate:
    """The b0 clip and square root (``_cover_surrogate``, width smooth_b0)."""

    WIDTH = 0.02
    POINTS = jnp.array([-0.4, -0.05, 0.0, 0.1, 0.5, 0.9, 0.99, 1.0, 1.05, 1.6])

    def _wrapped(self):
        return with_surrogate_gradient(
            _cover_exact, lambda x: _cover_surrogate(x, self.WIDTH))

    def test_value_exact_and_derivative_is_the_surrogates(self):
        check_surrogate_gradient(
            self._wrapped(), _cover_exact,
            lambda x: _cover_surrogate(x, self.WIDTH), (self.POINTS,))
        np.testing.assert_array_equal(
            np.asarray(cover_from_b0(self.POINTS, self.WIDTH)),
            np.asarray(_cover_exact(self.POINTS)))

    def test_surrogate_is_smooth(self):
        check_gradients(lambda x: _cover_surrogate(x, self.WIDTH),
                        (jnp.linspace(-0.3, 1.4, 23),), rtol=1e-4)

    def test_distance_bound(self):
        """``|surrogate - exact| <= sqrt(w ln 2)``, the value at ``b0 = 1``."""
        x = jnp.linspace(-1.0, 3.0, 4001)
        gap = np.abs(np.asarray(_cover_surrogate(x, self.WIDTH)
                                - _cover_exact(x)))
        bound = np.sqrt(self.WIDTH * np.log(2.0))
        assert gap.max() <= bound * (1 + 1e-6)
        assert gap.max() >= 0.9 * bound
        far = (np.asarray(x) < -0.2) | (np.asarray(x) > 1.2)
        assert gap[far].max() < 1e-3

    def test_slope_is_bounded(self):
        """Surrogate slope peaks near ``1/(2 sqrt(w ln 2))``; finite everywhere."""
        x = jnp.linspace(-1.0, 5.0, 6001)
        slope = np.asarray(jax.vmap(jax.grad(
            lambda v: cover_from_b0(v, self.WIDTH)))(x))
        assert np.all(np.isfinite(slope)) and np.all(slope >= 0.0)
        assert slope.max() < 1.0 / (2.0 * np.sqrt(self.WIDTH * np.log(2.0)))

    def test_zero_width_selects_the_reference_derivative(self):
        x = jnp.array([-0.5, 0.5, 1.5])
        slope = np.asarray(jax.vmap(jax.grad(lambda v: cover_from_b0(v, 0.0)))(x))
        np.testing.assert_allclose(slope, [0.0, 0.5 / np.sqrt(0.5), 0.0])


class TestInversionSurrogate:
    """The stability test of the inversion search (``_zsat_surrogate``)."""

    WIDTH = 2.0e-4
    STATIC = dict(jbmin=39, jbmax=44, nadd=0)

    def _args(self, bump=0.0):
        t, ph, pf, geo = _l47_column(inversion_level=None)
        dz = (geo[41] - geo[42]) / c.grav
        # level 42 at -2.4e-3 K/m (+ bump): just above the -2.44e-3 threshold
        t[:42] -= t[41] - (t[42] + (-2.4e-3 + bump) * dz)
        geo = _geopotential(t, ph, pf)
        lapse = _inversion_lapse(jnp.asarray(t), jnp.asarray(geo))
        return (lapse, jnp.asarray(0.7), jnp.asarray(0.25), jnp.asarray(1.0))

    def _exact(self, *a):
        return _zsat_exact(*a, **self.STATIC)

    def _surrogate(self, *a):
        return _zsat_surrogate(*a, **self.STATIC, width=self.WIDTH)

    def test_value_exact_and_derivative_is_the_surrogates(self):
        f = with_surrogate_gradient(self._exact, self._surrogate)
        check_surrogate_gradient(f, self._exact, self._surrogate, self._args())

    def test_surrogate_is_smooth(self):
        check_gradients(self._surrogate, self._args(), rtol=1e-3)

    def test_cinv_carries_a_gradient_near_the_threshold(self):
        f = with_surrogate_gradient(self._exact, self._surrogate)
        lapse, csatsc, cinv, enhance = self._args()
        g = jax.grad(lambda ci: f(lapse, csatsc, ci, enhance).sum())(cinv)
        assert np.isfinite(float(g)) and float(g) < 0.0

    def test_distance_bound(self):
        """``|surrogate - exact| <= (1 - csatsc)·max|H - σ|``.

        Half the enhancement at the threshold itself, and below
        ``0.3·exp(-5)`` once the chosen level is 5 widths from it.
        """
        for bump, bound in ((0.0, 0.3 * 0.5), (5 * self.WIDTH, 0.3 * np.exp(-5)),
                            (-5 * self.WIDTH, 0.3 * np.exp(-5))):
            args = self._args(bump)
            gap = np.abs(np.asarray(self._surrogate(*args) - self._exact(*args)))
            assert gap.max() <= bound * 1.01, (bump, gap.max())


class TestJaxTransformations:

    def test_jit_and_grad_through_the_cover(self):
        params = CloudParameters.default()
        t, ph, pf, geo = _l47_column(inversion_level=42)
        qi = np.zeros_like(t)
        q = jnp.asarray(0.95 * _qs_numpy(t, qi, pf))

        def total(temp):
            return calculate_cloud_fraction(
                temp, q, jnp.asarray(qi), jnp.asarray(pf),
                jnp.asarray(ph[-1]), jnp.asarray(geo), params,
                L47_RANGE)[0].sum()

        g = jax.jit(jax.grad(total))(jnp.asarray(t))
        assert np.all(np.isfinite(np.asarray(g)))
        assert np.any(np.asarray(g) != 0.0)

    def test_parameter_gradients_are_live(self):
        """crt, crs, csatsc carry gradients through the surrogate."""
        t, ph, pf, geo = _l47_column(inversion_level=42)
        qi = np.zeros_like(t)
        # 0.672 = 0.96·csatsc: inside the ramp at the enhanced level 42
        q = jnp.asarray(0.672 * _qs_numpy(t, qi, pf))
        base = CloudParameters.default()

        def total(params):
            return calculate_cloud_fraction(
                jnp.asarray(t), q, jnp.asarray(qi), jnp.asarray(pf),
                jnp.asarray(ph[-1]), jnp.asarray(geo), params,
                L47_RANGE)[0].sum()

        g = jax.grad(total)(base)
        for name in ("crt", "crs", "csatsc", "nex"):
            value = float(getattr(g, name))
            assert np.isfinite(value) and value != 0.0, name


# ---------------------------------------------------------------------------
# The mixed-phase helper kept for other callers
# ---------------------------------------------------------------------------

class TestMixedPhaseHelper:

    def test_blends_the_sonntag_fits(self):
        """The water fit at and above ``tmelt``, ice at and below 238.15 K.

        Linear in temperature between, with ``qs`` in ECHAM's form, both
        from ``thermodynamics``.
        """
        p = jnp.array(60000.0)
        mixed = (255.0 - 238.15) / (c.tmelt - 238.15)
        for temp, weight in ((c.tmelt + 5.0, 1.0), (c.tmelt, 1.0),
                             (255.0, mixed), (238.15, 0.0), (230.0, 0.0)):
            t = jnp.array(temp)
            vapour = (weight * thermodynamics.es_water(t)
                      + (1.0 - weight) * thermodynamics.es_ice(t))
            np.testing.assert_allclose(
                float(saturation_specific_humidity(p, t)),
                float(thermodynamics.qsat_from_es(vapour, p)), rtol=1e-6)
        qs = saturation_specific_humidity(jnp.array(101325.0), jnp.array(288.15))
        assert 0.008 < float(qs) < 0.012


def test_cover_qs_is_echams_form():
    """``x/(1 - vtmpc1·x)`` with ``x = min(es·rd/rv/p, 0.5)`` (l.221-223)."""
    params = CloudParameters.default()
    t = jnp.array([230.0, 260.0, 300.0])
    p = jnp.array([30000.0, 60000.0, 100000.0])
    qs = cover_saturation_specific_humidity(t, jnp.zeros(3), p, params)
    np.testing.assert_allclose(np.asarray(qs),
                               _qs_numpy(np.asarray(t), np.zeros(3),
                                         np.asarray(p)), rtol=1e-12)


# ---------------------------------------------------------------------------
# The formulation against the Fortran
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("prec", ["float64", "float32"])
@pytest.mark.parametrize("formula_variant,nn", [
    ("tetens", 63), ("sonntag", 31), ("sonntag", 63), ("sonntag", 127),
    ("sonntag", 255)])
def test_formulation_matches_echam(monkeypatch, formula_variant, nn, prec):
    """Every Fortran cover column, under ECHAM's formula and the test Tetens.

    ``sonntag`` (the cover's formula) against the primary reference at ECHAM's
    four truncations, each with its parameter row (the same comparison
    ``echam_fortran_reference_test`` makes); ``tetens`` (the test utility)
    against the reference's ``tetens`` variant, ECHAM's routine with that
    pair inside, which confirms the formulation independently of the vapour
    pressure. Both at the reference module's own tolerances.
    """
    from jcm.physics.clouds import echam_fortran_reference_test as ref

    if prec == "float32" and nn != 63:
        pytest.skip("the resolution reference is float64 only")
    monkeypatch.setattr(es, "SATURATION_FORMULA", formula_variant)
    which = None if nn == 63 else nn
    inp = ref.echam_inputs("cover", formula_variant, which)
    want = ref.echam_outputs("cover", formula_variant, which)["paclc"]
    with ref.echam_constants(), ref.precision(prec):
        got = ref.run_jcm_cover(inp, nn=nn)["paclc"]
    rtol = ref.RTOL_F64 if prec == "float64" else ref.RTOL_F32
    atol = (ref.ATOL if prec == "float64" else ref.ATOL_F32)["paclc"]
    d = ref.B0_ERROR[prec]
    scale = np.max(np.abs(want), axis=0)
    tol = (atol + rtol * scale[None]
           + np.minimum(np.sqrt(d), d / (2.0 * np.maximum(1.0 - want, 1e-300))))
    bad = np.abs(got - want) > tol
    names = ref.column_names("cover")
    assert not bad.any(), [names[j] for j in np.nonzero(bad.any(axis=0))[0]]
