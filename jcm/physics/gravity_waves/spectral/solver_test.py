"""Tests for the spectral GW solver against a NumPy loop reference.

The reference functions below are direct loop-by-loop transliterations of
CAM ``gw_common.F90`` (``gw_prof`` / ``gw_drag_prof``), kept in NumPy with
explicit ``k``/``l`` loops as the repo's established Fortran-port
validation pattern: the vectorized JAX solver must agree with the scalar
loop code on randomized, physically-plausible profiles.
"""

import math
import unittest

import jax
import jax.numpy as jnp
import numpy as np

import jcm.constants as c
from jcm.physics.gravity_waves.spectral.frontal import (
    gaussian_spectrum,
    gw_cm_src,
)
from jcm.physics.gravity_waves.spectral.solver import (
    DBACK,
    GWUT_TINY,
    N2MIN,
    TAUMIN,
    UBMC2MN,
    GWBand,
    gw_drag_prof,
    gw_prof,
    newtonian_cooling_profile,
)

# ---------------------------------------------------------------------------
# NumPy loop reference (transliterated from gw_common.F90, 0-based top-first)
# ---------------------------------------------------------------------------


def ref_gw_prof(t, p_ifc, p_mid):
    """Loop transliteration of gw_common.F90::gw_prof (single column)."""
    nlev = t.size
    ti = np.empty(nlev + 1)
    rhoi = np.empty(nlev + 1)
    ni = np.empty(nlev + 1)
    ti[0] = t[0]
    rhoi[0] = p_ifc[0] / (c.rd * ti[0])
    ni[0] = math.sqrt(c.grav * c.grav / (c.cpd * ti[0]))
    for k in range(1, nlev):
        ti[k] = 0.5 * (t[k - 1] + t[k])
        rhoi[k] = p_ifc[k] / (c.rd * ti[k])
        dtdp = (t[k] - t[k - 1]) / (p_mid[k] - p_mid[k - 1])
        n2 = c.grav * c.grav / ti[k] * (1.0 / c.cpd - rhoi[k] * dtdp)
        ni[k] = math.sqrt(max(N2MIN, n2))
    ti[nlev] = t[nlev - 1]
    rhoi[nlev] = p_ifc[nlev] / (c.rd * ti[nlev])
    ni[nlev] = ni[nlev - 1]
    nm = 0.5 * (ni[:-1] + ni[1:])
    return rhoi, nm, ni


def ref_gw_drag_prof(ngwv, kwv, effkwv, ksrc, dt, t, p_ifc, piln, rhoi, ni,
                     ubm, ubi, xv, yv, effgw, cc, tau_src, alpha,
                     tndmax, umcfac, satfac, limit_sum=False):
    """Loop transliteration of gw_common.F90::gw_drag_prof (single column).

    lapply_effgw=.true. path, no vertical diffusion, kvtt=0,
    tend_level == src_level. Returns the pre-down-scan stress profile too
    (for the saturation-monotonicity check). ``limit_sum=False`` is the
    exact CAM limiter (cap the net tendency); ``limit_sum=True`` is the
    port's production default (cap the absolute sum, which also bounds
    the frictional heating — solver deviation 6).
    """
    nlev = t.size
    nspec = 2 * ngwv + 1
    rog = c.rd / c.grav
    tau = np.zeros((nlev + 1, nspec))
    tau[ksrc + 1] = tau_src

    # Up-scan: do k = kbot_src, ktop, -1
    for k in range(ksrc, -1, -1):
        d = DBACK  # + kvtt(k), zero here
        for l_ in range(nspec):
            ubmc = ubi[k] - cc[l_]
            tausat = 0.0
            if (ubmc > 0.0) == (ubi[k + 1] > cc[l_]):
                tausat = abs(effkwv * rhoi[k] * ubmc**3 / (satfac * ni[k]))
            ubmc2 = max(ubmc**2, UBMC2MN)
            mi = ni[k] / (2.0 * kwv * ubmc2) * (alpha[k] + ni[k]**2 / ubmc2 * d)
            wrk = -2.0 * mi * rog * t[k] * (piln[k + 1] - piln[k])
            taudmp = tau[k + 1, l_] * math.exp(wrk)
            if tausat <= TAUMIN:
                tausat = 0.0
            if taudmp <= TAUMIN:
                taudmp = 0.0
            tau[k, l_] = min(taudmp, tausat)

    # Apply efficiency to the completed stress profile
    # (interfaces ktop..tend_level+1).
    tau[:ksrc + 2] *= effgw
    tau_up = tau.copy()

    # Down-scan: do k = ktop, kbot_tend
    utgw = np.zeros(nlev)
    vtgw = np.zeros(nlev)
    gwut = np.zeros((nlev, nspec))
    dttke = np.zeros(nlev)
    for k in range(0, ksrc + 1):
        ubt = 0.0
        for l_ in range(nspec):
            ubtl = c.grav * (tau[k + 1, l_] - tau[k, l_]) / (p_ifc[k + 1] - p_ifc[k])
            ubtl = min(ubtl, umcfac * abs(cc[l_] - ubm[k]) / dt)
            # sign(ubtl, c - ubm)
            gwut[k, l_] = abs(ubtl) if cc[l_] - ubm[k] >= 0.0 else -abs(ubtl)
            ubt += gwut[k, l_]
        lim = max(abs(ubt), np.abs(gwut[k]).sum()) if limit_sum else abs(ubt)
        if lim > tndmax:
            ratio = tndmax / lim
            ubt *= ratio
        else:
            ratio = 1.0
        for l_ in range(nspec):
            gwut[k, l_] *= ratio
            if abs(gwut[k, l_]) < GWUT_TINY:
                gwut[k, l_] = 0.0
            tau[k + 1, l_] = tau[k, l_] + abs(gwut[k, l_]) * (p_ifc[k + 1] - p_ifc[k]) / c.grav
        utgw[k] = ubt * xv
        vtgw[k] = ubt * yv
    for l_ in range(nspec):
        for k in range(0, ksrc + 1):
            dttke[k] -= (ubm[k] - cc[l_]) * gwut[k, l_]
    return utgw, vtgw, dttke, gwut, tau, tau_up


def ref_gaussian_src_tau(ngwv, dc, height, width):
    """Bin-averaged Gaussian spectrum via math.erfc (gw_front.F90)."""
    cref = dc * np.arange(-ngwv, ngwv + 1)
    bounds = np.concatenate([cref - 0.5 * dc, [cref[-1] + 0.5 * dc]])
    integ = np.array([math.erfc(b / width) for b in bounds])
    integ *= height * width * math.sqrt(math.pi) / 2.0
    src = (integ[:-1] - integ[1:]) / dc
    src[ngwv] = 0.0
    return src


def make_profiles(rng, nlev=40):
    """Randomized but physically-plausible column: jet + stratification."""
    # Interface pressures: CAM-like, top at 2 Pa, surface ~1e5 Pa.
    frac = np.linspace(0.0, 1.0, nlev + 1) ** 2.2
    p_ifc = 2.0 + (1.0e5 - 2.0) * frac
    p_mid = 0.5 * (p_ifc[:-1] + p_ifc[1:])
    # Temperature: tropospheric lapse to a 205 K tropopause, warming above.
    t = np.where(
        p_mid > 2.0e4,
        288.0 - 45.0 * np.log(1.0e5 / np.maximum(p_mid, 1.0)) / np.log(5.0),
        205.0 + 12.0 * np.log(2.0e4 / np.maximum(p_mid, 1.0)),
    )
    t = np.clip(t + rng.normal(0.0, 1.5, nlev), 180.0, 310.0)
    # Winds: mid-latitude jet near 250 hPa plus noise.
    u = 35.0 * np.exp(-((np.log(p_mid) - np.log(2.5e4)) / 0.8) ** 2)
    u = u + rng.normal(0.0, 2.0, nlev)
    v = 5.0 * np.sin(np.linspace(0, np.pi, nlev)) + rng.normal(0.0, 1.0, nlev)
    return p_ifc, p_mid, t, u, v


def project_source(u, v, ksrc):
    """Source wind, unit vector and projections (gw_cm_src wind part)."""
    usrc = 0.5 * (u[ksrc + 1] + u[ksrc])
    vsrc = 0.5 * (v[ksrc + 1] + v[ksrc])
    mag = math.hypot(usrc, vsrc)
    xv, yv = (usrc / mag, vsrc / mag) if mag > 0 else (0.0, 0.0)
    ubm = u * xv + v * yv
    ubi = np.concatenate([[ubm[0]], 0.5 * (ubm[:-1] + ubm[1:]), [ubm[-1]]])
    ubi[ksrc + 1] = mag
    return ubm, ubi, xv, yv, mag


NGWV = 8
DC = 2.5
WAVELENGTH = 1.0e5
FCRIT2 = 1.0
KSRC = 27           # midpoint index near 500 hPa for the 40-level test grid
DT = 1800.0
TNDMAX = 400.0 / 86400.0
UMCFAC = 0.5
SATFAC = 2.0
TAUBGND = 1.5e-3
WIDTH = 30.0


class GwDragProfReferenceTest(unittest.TestCase):
    """JAX solver vs the NumPy loop reference, float64, rtol 1e-5."""

    def test_matches_loop_reference(self):
        rng = np.random.default_rng(42)
        with jax.enable_x64():
            band = GWBand(dc=DC, fcrit2=FCRIT2, wavelength=WAVELENGTH,
                          ngwv=NGWV)
            kwv = 2.0 * np.pi / WAVELENGTH
            effkwv = FCRIT2 * kwv
            src_tau = ref_gaussian_src_tau(NGWV, DC, TAUBGND, WIDTH)
            effgw = 1.0

            for col in range(6):
                p_ifc, p_mid, t, u, v = make_profiles(rng)
                nlev = t.size
                piln = np.log(p_ifc)
                alpha = newtonian_cooling_profile(p_ifc)
                ubm, ubi, xv, yv, mag = project_source(u, v, KSRC)
                cc = DC * np.arange(-NGWV, NGWV + 1) + mag

                rhoi_r, nm_r, ni_r = ref_gw_prof(t, p_ifc, p_mid)
                rhoi_j, nm_j, ni_j = gw_prof(
                    jnp.asarray(t), jnp.asarray(p_ifc), jnp.asarray(p_mid))
                np.testing.assert_allclose(rhoi_j, rhoi_r, rtol=1e-10)
                np.testing.assert_allclose(nm_j, nm_r, rtol=1e-10)
                np.testing.assert_allclose(ni_j, ni_r, rtol=1e-10)

                # Validate both limiter modes: exact CAM (net cap,
                # limit_sum=False) and the production heating-bounded
                # variant (absolute-sum cap, limit_sum=True).
                for limit_sum in (False, True):
                    utgw_r, vtgw_r, dttke_r, gwut_r, tau_r, tau_up_r = (
                        ref_gw_drag_prof(
                            NGWV, kwv, effkwv, KSRC, DT, t, p_ifc, piln,
                            rhoi_r, ni_r, ubm, ubi, xv, yv, effgw, cc,
                            src_tau, alpha, TNDMAX, UMCFAC, SATFAC,
                            limit_sum=limit_sum,
                        )
                    )

                    result = gw_drag_prof(
                        band, KSRC, DT,
                        jnp.asarray(t), jnp.asarray(p_ifc), jnp.asarray(piln),
                        rhoi_j, ni_j,
                        jnp.asarray(ubm), jnp.asarray(ubi),
                        jnp.asarray(xv), jnp.asarray(yv),
                        effgw, jnp.asarray(cc), jnp.asarray(src_tau),
                        jnp.asarray(alpha),
                        tndmax=TNDMAX, umcfac=UMCFAC, satfac=SATFAC,
                        limit_tendency_sum=limit_sum,
                    )
                    msg = f"column {col}, limit_sum={limit_sum}"
                    np.testing.assert_allclose(
                        result.utgw, utgw_r, rtol=1e-5, atol=1e-14, err_msg=msg)
                    np.testing.assert_allclose(
                        result.vtgw, vtgw_r, rtol=1e-5, atol=1e-14, err_msg=msg)
                    np.testing.assert_allclose(
                        result.ttgw, dttke_r, rtol=1e-5, atol=1e-14, err_msg=msg)
                    np.testing.assert_allclose(
                        result.gwut, gwut_r, rtol=1e-5, atol=1e-14, err_msg=msg)
                    np.testing.assert_allclose(
                        result.tau, tau_r, rtol=1e-5, atol=1e-14, err_msg=msg)

                # Saturation monotonicity of the pre-adjustment stress:
                # above the source, tau[k] = min(damped tau[k+1], tausat)
                # can never exceed tau[k+1].
                for k in range(0, KSRC + 1):
                    self.assertTrue(
                        np.all(tau_up_r[k] <= tau_up_r[k + 1] + 1e-18), msg)

                # nlev sanity for the fixture
                self.assertEqual(nlev, 40)


class GwDragPropertiesTest(unittest.TestCase):
    """Physical bookkeeping properties of the JAX solver output."""

    def _run(self, seed=7, tau_0_ubc=False):
        rng = np.random.default_rng(seed)
        p_ifc, p_mid, t, u, v = make_profiles(rng)
        piln = np.log(p_ifc)
        alpha = newtonian_cooling_profile(p_ifc)
        ubm, ubi, xv, yv, mag = project_source(u, v, KSRC)
        cc = DC * np.arange(-NGWV, NGWV + 1) + mag
        src_tau = ref_gaussian_src_tau(NGWV, DC, TAUBGND, WIDTH)
        band = GWBand(dc=DC, fcrit2=FCRIT2, wavelength=WAVELENGTH, ngwv=NGWV)
        rhoi, _, ni = gw_prof(jnp.asarray(t), jnp.asarray(p_ifc),
                              jnp.asarray(p_mid))
        result = gw_drag_prof(
            band, KSRC, DT,
            jnp.asarray(t), jnp.asarray(p_ifc), jnp.asarray(piln),
            rhoi, ni, jnp.asarray(ubm), jnp.asarray(ubi),
            jnp.asarray(xv), jnp.asarray(yv),
            1.0, jnp.asarray(cc), jnp.asarray(src_tau), jnp.asarray(alpha),
            tndmax=TNDMAX, umcfac=UMCFAC, satfac=SATFAC, tau_0_ubc=tau_0_ubc,
        )
        return result, p_ifc, xv, yv

    def test_momentum_bookkeeping_per_wave(self):
        # The adjusted stress divergence must equal the deposited tendency
        # exactly: |gwut[k,l]| * dp[k] / g == tau[k+1,l] - tau[k,l].
        result, p_ifc, _, _ = self._run()
        dp = (p_ifc[1:] - p_ifc[:-1])[:, None]
        deposited = np.abs(np.asarray(result.gwut)) * dp / c.grav
        dtau = np.asarray(result.tau[1:]) - np.asarray(result.tau[:-1])
        np.testing.assert_allclose(
            deposited[:KSRC + 1], dtau[:KSRC + 1], rtol=1e-5, atol=1e-12)

    def test_tendency_never_exceeds_tndmax(self):
        result, _, xv, yv = self._run()
        # utgw = ubt * xv, vtgw = ubt * yv with xv^2 + yv^2 = 1, and the
        # limiter caps |ubt| at tndmax.
        ubt = np.hypot(np.asarray(result.utgw), np.asarray(result.vtgw))
        self.assertLessEqual(float(ubt.max()), TNDMAX * (1.0 + 1e-5))

    def test_tau_non_increasing_above_source(self):
        result, _, _, _ = self._run()
        tau = np.asarray(result.tau)
        # Non-increasing upward through the propagation region.
        self.assertTrue(np.all(tau[:KSRC + 1] <= tau[1:KSRC + 2] + 1e-15))

    def test_tau_0_ubc_zeroes_top(self):
        result, _, _, _ = self._run(tau_0_ubc=True)
        np.testing.assert_array_equal(np.asarray(result.tau[0]), 0.0)


class BroadcastingTest(unittest.TestCase):
    """Column (kx,) vs batched (kx, ncols) equality per CLAUDE.md."""

    def test_column_vs_batch(self):
        rng = np.random.default_rng(3)
        ncols = 4
        cols = [make_profiles(rng) for _ in range(ncols)]
        band = GWBand(dc=DC, fcrit2=FCRIT2, wavelength=WAVELENGTH, ngwv=NGWV)
        src_tau = jnp.asarray(ref_gaussian_src_tau(NGWV, DC, TAUBGND, WIDTH))

        def solve(p_ifc, p_mid, t, u, v):
            piln = jnp.log(p_ifc)
            alpha = jnp.asarray(newtonian_cooling_profile(
                np.asarray(cols[0][0])))  # same reference alpha for all
            rhoi, _, ni = gw_prof(t, p_ifc, p_mid)
            src = gw_cm_src(band, KSRC, u, v,
                            jnp.full(t.shape[1:], 1.0e-14), 3.0e-15, src_tau)
            return gw_drag_prof(
                band, KSRC, DT, t, p_ifc, piln, rhoi, ni,
                src.ubm, src.ubi, src.xv, src.yv, 1.0, src.c, src.tau_src,
                alpha, tndmax=TNDMAX, umcfac=UMCFAC, satfac=SATFAC)

        stacked = [jnp.stack([jnp.asarray(col[i]) for col in cols], axis=-1)
                   for i in range(5)]
        batch = solve(*stacked)

        for i in range(ncols):
            single = solve(*(jnp.asarray(cols[i][j]) for j in range(5)))
            np.testing.assert_allclose(
                np.asarray(batch.utgw[..., i]), np.asarray(single.utgw),
                rtol=2e-5, atol=1e-9)
            np.testing.assert_allclose(
                np.asarray(batch.ttgw[..., i]), np.asarray(single.ttgw),
                rtol=2e-5, atol=1e-9)
            np.testing.assert_allclose(
                np.asarray(batch.tau[..., i]), np.asarray(single.tau),
                rtol=2e-5, atol=1e-12)


class GradientTest(unittest.TestCase):
    """No NaN gradients, including through columns where nothing launches."""

    def _loss_fn(self, frontgf_value):
        rng = np.random.default_rng(11)
        p_ifc, p_mid, t, u, v = make_profiles(rng)
        band = GWBand(dc=DC, fcrit2=FCRIT2, wavelength=WAVELENGTH, ngwv=NGWV)
        alpha = jnp.asarray(newtonian_cooling_profile(p_ifc))
        p_ifc_j = jnp.asarray(p_ifc)
        p_mid_j = jnp.asarray(p_mid)
        t_j = jnp.asarray(t)
        frontgf = jnp.asarray(frontgf_value)

        def loss(u_prof, v_prof, taubgnd):
            src_tau = gaussian_spectrum(band, taubgnd, WIDTH)
            piln = jnp.log(p_ifc_j)
            rhoi, _, ni = gw_prof(t_j, p_ifc_j, p_mid_j)
            src = gw_cm_src(band, KSRC, u_prof, v_prof, frontgf,
                            3.0e-15, src_tau)
            result = gw_drag_prof(
                band, KSRC, DT, t_j, p_ifc_j, piln, rhoi, ni,
                src.ubm, src.ubi, src.xv, src.yv, 1.0, src.c, src.tau_src,
                alpha, tndmax=TNDMAX, umcfac=UMCFAC, satfac=SATFAC)
            return jnp.sum(result.utgw ** 2) + jnp.sum(result.ttgw ** 2)

        return loss, jnp.asarray(u), jnp.asarray(v)

    def test_grad_finite_when_launching(self):
        loss, u, v = self._loss_fn(1.0e-14)   # above frontgfc -> launches
        gu, gv, gtau = jax.grad(loss, argnums=(0, 1, 2))(u, v, 1.5e-3)
        self.assertTrue(bool(jnp.all(jnp.isfinite(gu))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(gv))))
        self.assertTrue(bool(jnp.isfinite(gtau)))
        # Waves launched, so something must actually depend on the inputs.
        self.assertGreater(float(jnp.abs(gu).max()), 0.0)

    def test_grad_finite_when_masked(self):
        # frontgf below threshold: NO wave launches — the fully-masked
        # branch is where where-mask 0*inf poisons hide.
        loss, u, v = self._loss_fn(0.0)
        self.assertEqual(float(loss(u, v, 1.5e-3)), 0.0)
        gu, gv, gtau = jax.grad(loss, argnums=(0, 1, 2))(u, v, 1.5e-3)
        self.assertTrue(bool(jnp.all(jnp.isfinite(gu))))
        self.assertTrue(bool(jnp.all(jnp.isfinite(gv))))
        self.assertTrue(bool(jnp.isfinite(gtau)))

    def test_grad_finite_zero_wind_column(self):
        # Zero source wind: get_unit_vector's masked sqrt(0)/division.
        loss, _, _ = self._loss_fn(1.0e-14)
        u0 = jnp.zeros(40)
        v0 = jnp.zeros(40)
        gu = jax.grad(loss, argnums=0)(u0, v0, 1.5e-3)
        self.assertTrue(bool(jnp.all(jnp.isfinite(gu))))


if __name__ == "__main__":
    unittest.main()
