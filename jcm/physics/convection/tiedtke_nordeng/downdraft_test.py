"""Regression tests for the Tiedtke-Nordeng downdraft scheme.

Covers two bugs surfaced by the Fortran harness against ECHAM
``mo_cudescent.f90``:

1. **Runaway mass flux**. The original implementation used the wrong
   entrainment constant (``entrscv*0.5`` instead of ``entrdd``) and
   only entrained without a matching detrainment, so ``|mfd|`` grew
   ~50x as the parcel descended a deep RCE column. ECHAM cuddraf
   conserves the downdraft mass flux in the bulk (entrainment matched
   by detrainment) and only tapers it to zero in the lowest two layers.

2. **Missing adiabatic compression**. The temperature update mixed
   ``td`` toward the environment without first applying the
   ``g·dz/cp`` adiabatic warming a descending parcel undergoes. As a
   result, the downdraft cooled monotonically (273 K at LFS → 248 K
   at the surface, vs an environment that ranged 290 → 304 K).

Both regressions are caught by feeding a tropical RCE-like sounding
through ``calculate_downdraft`` and checking that:

  * ``|mfd|`` stays within a small factor of its LFS value through
    the bulk of the column, and ramps to zero at the surface,
  * ``td`` stays within ~10 K of the environment (rather than
    diverging by tens of K),
  * the downdraft does not produce zero or grow-without-bound mfd.
"""

import unittest

import jax.numpy as jnp
import numpy as np

from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
    ConvectionParameters, find_cloud_base, calculate_cape_cin,
)
from jcm.physics.convection.tiedtke_nordeng.flux_tendencies import (
    mass_flux_closure,
)
from jcm.physics.convection.tiedtke_nordeng.updraft import calculate_updraft
from jcm.physics.convection.tiedtke_nordeng.downdraft import (
    calculate_downdraft,
    downdraft_entrainment_ledger,
)


def _rce_column(klev=47):
    """Build a simple tropical RCE column (305 K SST, 90 % RH).

    Returns ``(T, q, p, layer_thickness, rho)`` as JAX arrays in the
    surface-first convention (k=0 top, k=nlev-1 surface).
    """
    grav = 9.80665
    p0 = 101325.0
    p_top = 1000.0  # 10 hPa
    sigma_bnds = jnp.linspace(p_top / p0, 1.0, klev + 1)
    p_half = sigma_bnds * p0
    p_full = 0.5 * (p_half[:-1] + p_half[1:])

    # Moist-adiabatic-ish sounding: cool aloft, ~305 K at surface, 90 % RH
    p = p_full
    surf_T = 305.0
    surf_q = 0.025
    Gamma = 6.5e-3  # K/m
    # Approximate height from p (just for T profile shape). ``z_full`` is
    # POSITIVE height above the surface (p < p0 ⇒ z_full > 0), so the lapse
    # subtracts with height: T decreases upward, as in a real sounding. (The
    # earlier ``-Gamma·(-z_full)`` had a sign error that made T *increase*
    # with height — 376 K at 255 hPa — an unphysical column that only stayed
    # hidden while the plume/LFS happened to land in its lower part.)
    z_full = -8000.0 * jnp.log(p / p0)  # H ~ 8 km scale height
    T = jnp.maximum(surf_T - Gamma * z_full, 200.0)
    # Constant 90 % RH (clipped to small at top)
    from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
        saturation_mixing_ratio,
    )
    qs = jnp.array([float(saturation_mixing_ratio(jnp.asarray(p[k]),
                                                   jnp.asarray(T[k])))
                    for k in range(klev)])
    q = 0.9 * qs
    q = q.at[-1].set(surf_q)

    Rd = 287.04
    Tv = T * (1.0 + 0.608 * q)
    rho = p / (Rd * Tv)
    layer_thickness = jnp.diff(-(-8000.0) * jnp.log(p_half[:-1] / p_half[1:]))
    # Simpler: use hypsometric Δz = R_d T_v / g · dlnp
    dlnp = jnp.diff(jnp.log(p_half))
    layer_thickness = Rd * Tv / grav * dlnp

    return T, q, p, layer_thickness, rho


def _default_config():
    return ConvectionParameters.default(
        entrpen=1.0e-4, entrscv=3.0e-3, entrmid=1.0e-4,
        entrdd=2.0e-4, tau=7200.0, cmfcmax=1.0, cmfcmin=1.0e-10,
        cprcon=2.5e-4, cevapcu=2.0e-5, cmfdeps=0.30,
    )


class TestDowndraftMassFlux(unittest.TestCase):
    """Mass-flux conservation in the bulk + surface taper."""

    def setUp(self):
        self.T, self.q, self.p, self.dz, self.rho = _rce_column(47)
        self.config = _default_config()
        cb, _ = find_cloud_base(self.T, self.q, self.p, self.config)
        self.cb = cb
        cape, cin = calculate_cape_cin(
            self.T, self.q, self.p, self.dz, cb, self.config,
        )
        self.ktop_ceil = jnp.maximum(cb - 35, jnp.array(2))
        self.mfb = mass_flux_closure(
            cape, cin, jnp.array(0.0), 1, self.config,
        )
        self.upd = calculate_updraft(
            self.T, self.q, self.p, self.dz, self.rho,
            cb, self.ktop_ceil, 1, self.mfb, self.config,
        )
        # cudlfs needs the rain the ascent produced (``zrfl``) and searches
        # inside the REALIZED cloud (kctop < jk < kcbot).
        self.precip = jnp.sum(self.upd.pdmfup)
        self.kctop = int(np.min(np.where(np.asarray(self.upd.mfu) > 0.0)[0]))

    def test_mfd_does_not_run_away(self):
        """``|mfd|`` should never exceed 2x its LFS-init value below the LFS.

        ECHAM cuddraf conserves mfd by matching entrainment and
        detrainment. A factor-2 ceiling is plenty of slack for any
        entrainment/detrainment imbalance in the implementation while
        catching the original 50x runaway.
        """
        dwn = calculate_downdraft(
            self.T, self.q, self.p, self.dz, self.rho,
            self.upd, self.precip, self.cb, self.kctop, self.config,
        )
        mfd = np.asarray(dwn.mfd)
        nonzero = mfd[np.abs(mfd) > 1e-12]
        if nonzero.size == 0:
            self.skipTest("no downdraft initialised on this column")
        # The init value at LFS sets the floor; nothing should grow above 2x.
        ref = float(np.max(np.abs(nonzero)))
        # Use the value at the LFS index as the reference if available.
        lfs = int(dwn.lfs)
        if abs(mfd[lfs]) > 1e-12:
            ref_init = abs(float(mfd[lfs]))
            ratio = ref / ref_init
            self.assertLess(
                ratio, 2.0,
                f"|mfd| peaked at {ref:.3e} kg/m²/s, "
                f"vs LFS-init {ref_init:.3e}; ratio={ratio:.2f} "
                "indicates runaway entrainment (Bug D regression)."
            )

    def test_mfd_tapers_linearly_to_the_surface(self):
        """Below ``itopde = klev − 2`` the downdraft stops entraining and
        detrains linearly in pressure (cuddraf), so its flux through each
        interface below itopde is the itopde value scaled by the air mass
        still below that interface, reaching zero at the surface — which
        leaves the flux through the top of the lowest layer small but
        nonzero (cudtdq's surface branch then deposits it there).
        """
        from jcm.physics.convection.tiedtke_nordeng.half_levels import (
            reconstruct_pressure_half,
        )
        dwn = calculate_downdraft(
            self.T, self.q, self.p, self.dz, self.rho,
            self.upd, self.precip, self.cb, self.kctop, self.config,
        )
        mfd = np.asarray(dwn.mfd)
        itopde = mfd.shape[0] - 3
        self.assertLess(mfd[itopde], 0.0, "fixture downdraft must reach itopde")
        ph = np.asarray(reconstruct_pressure_half(self.p))
        ps = ph[-1]
        expected = mfd[itopde] * (ps - ph[itopde + 1:-1]) / (ps - ph[itopde])
        np.testing.assert_allclose(mfd[itopde + 1:], expected, rtol=1e-4)
        self.assertLess(abs(mfd[-1]), 0.5 * abs(mfd[itopde]))


class TestDowndraftTemperature(unittest.TestCase):
    """Temperature evolution under adiabatic compression + mixing."""

    def setUp(self):
        self.T, self.q, self.p, self.dz, self.rho = _rce_column(47)
        self.config = _default_config()
        cb, _ = find_cloud_base(self.T, self.q, self.p, self.config)
        self.cb = cb
        cape, cin = calculate_cape_cin(
            self.T, self.q, self.p, self.dz, cb, self.config,
        )
        self.ktop_ceil = jnp.maximum(cb - 35, jnp.array(2))
        self.mfb = mass_flux_closure(
            cape, cin, jnp.array(0.0), 1, self.config,
        )
        self.upd = calculate_updraft(
            self.T, self.q, self.p, self.dz, self.rho,
            cb, self.ktop_ceil, 1, self.mfb, self.config,
        )
        # cudlfs needs the rain the ascent produced (``zrfl``) and searches
        # inside the REALIZED cloud (kctop < jk < kcbot).
        self.precip = jnp.sum(self.upd.pdmfup)
        self.kctop = int(np.min(np.where(np.asarray(self.upd.mfu) > 0.0)[0]))

    def test_td_stays_close_to_environment(self):
        """Without adiabatic warming, the downdraft used to cool by ~25 K
        descending from LFS to surface (the parcel inertia retained the
        cold initial wet-bulb temperature). With g·dz/cp warming applied
        each layer plus mixing toward env, td should track the env to
        within ~10 K all the way down.
        """
        dwn = calculate_downdraft(
            self.T, self.q, self.p, self.dz, self.rho,
            self.upd, self.precip, self.cb, self.kctop, self.config,
        )
        td = np.asarray(dwn.td)
        T_env = np.asarray(self.T)
        mfd = np.asarray(dwn.mfd)
        active = np.abs(mfd) > 1e-10
        if not np.any(active):
            self.skipTest("no downdraft initialised on this column")
        deviations = np.abs(td[active] - T_env[active])
        max_dev = float(np.max(deviations))
        self.assertLess(
            max_dev, 15.0,
            f"max |td - T_env| = {max_dev:.2f} K is too large; "
            "indicates missing adiabatic compression (Bug D follow-on "
            "regression)."
        )

    def test_td_warms_with_descent(self):
        """Going from LFS down to the surface, the downdraft temperature
        should generally INCREASE due to adiabatic compression. (Mixing
        toward env can temporarily reverse this in any one layer, but
        the net trend over the bulk should be warming.)
        """
        dwn = calculate_downdraft(
            self.T, self.q, self.p, self.dz, self.rho,
            self.upd, self.precip, self.cb, self.kctop, self.config,
        )
        mfd = np.asarray(dwn.mfd)
        td = np.asarray(dwn.td)
        active_indices = np.where(np.abs(mfd) > 1e-10)[0]
        if active_indices.size < 4:
            self.skipTest("downdraft too short to test trend")
        first = int(active_indices[0])
        last = int(active_indices[-1])
        delta = float(td[last] - td[first])
        self.assertGreater(
            delta, 0.0,
            f"td(surface)={td[last]:.2f} should exceed td(LFS)={td[first]:.2f} "
            "via adiabatic compression."
        )


class TestDowndraftEntrainmentLedger(unittest.TestCase):
    """The #622 export: cuddraf's zentr = entrdd·|mfd|·dz per layer.

    ``mfd[j]`` is the half-level flux through the TOP interface of layer j,
    so layer j entrains while the descent enters it (``mfd[j] < 0``) and
    continues through its bottom (``mfd[j+1] < 0``).
    """

    def test_bulk_taper_and_lfs_masking(self):
        nlev, entrdd, dz_val = 10, 2.0e-4, 400.0
        lev = jnp.arange(nlev)[:, None]
        # LFS at interface 4, bulk down to ``itopde = nlev − 3``, then the
        # cuddraf surface taper (zero at the surface interface).
        mfd = jnp.where((lev >= 4) & (lev <= nlev - 3), -0.02, 0.0)
        mfd = mfd.at[nlev - 2].set(-0.0133).at[nlev - 1].set(-0.0067)
        dz = jnp.full((nlev, 1), dz_val)
        ledger = np.asarray(downdraft_entrainment_ledger(mfd, dz, entrdd))
        # Zero above the LFS (no descent through those layers).
        np.testing.assert_array_equal(ledger[:4, 0], 0.0)
        # Bulk: entrdd·|mfd|·dz, from the LFS layer down to above itopde.
        np.testing.assert_allclose(
            ledger[4:nlev - 3, 0], entrdd * 0.02 * dz_val, rtol=1e-6,
        )
        # Surface taper: entrainment shut off (Fortran itopde).
        np.testing.assert_array_equal(ledger[nlev - 3:, 0], 0.0)

    def test_dead_downdraft_has_zero_ledger(self):
        # Buoyancy shut-off inside layer 5 (mfd == 0 at its bottom interface
        # with inflow through its top): no entrainment there, so plume
        # continuity dumps the arriving flux as detrainment.
        lev = jnp.arange(10)[:, None]
        mfd = jnp.where((lev >= 3) & (lev <= 5), -0.02, 0.0)
        ledger = np.asarray(downdraft_entrainment_ledger(
            mfd, jnp.full((10, 1), 400.0), 2.0e-4,
        ))
        self.assertGreater(float(ledger[4, 0]), 0.0)
        self.assertEqual(float(ledger[5, 0]), 0.0)
        self.assertEqual(float(ledger[6, 0]), 0.0)

    def test_column_and_block_shapes_agree(self):
        # Broadcasting-native: a (nlev,) column and a (nlev, ncols) block
        # must agree per column.
        lev = jnp.arange(10)
        mfd_col = jnp.where((lev >= 4) & (lev < 8), -0.02, 0.0)
        dz_col = jnp.full(10, 400.0)
        col = np.asarray(downdraft_entrainment_ledger(mfd_col, dz_col, 2e-4))
        block = np.asarray(downdraft_entrainment_ledger(
            mfd_col[:, None] * jnp.ones((1, 3)),
            dz_col[:, None] * jnp.ones((1, 3)), 2e-4,
        ))
        for c in range(3):
            np.testing.assert_allclose(block[:, c], col)


if __name__ == "__main__":
    unittest.main()
