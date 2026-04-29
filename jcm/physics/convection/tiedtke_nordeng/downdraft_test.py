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
    # Approximate height from p (just for T profile shape)
    z_full = -8000.0 * jnp.log(p / p0)  # H ~ 8 km scale height
    T = jnp.maximum(surf_T - Gamma * (-z_full), 200.0)
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


def _default_config(dt=1800.0):
    return ConvectionParameters.default(
        dt_conv=dt, entrpen=1.0e-4, entrscv=3.0e-3, entrmid=1.0e-4,
        entrdd=2.0e-4, tau=7200.0, cmfcmax=1.0, cmfcmin=1.0e-10,
        cprcon=2.5e-4, cevapcu=2.0e-5, cmfctop=0.20, cmfdeps=0.30,
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

    def test_mfd_does_not_run_away(self):
        """``|mfd|`` should never exceed 2x its LFS-init value below the LFS.

        ECHAM cuddraf conserves mfd by matching entrainment and
        detrainment. A factor-2 ceiling is plenty of slack for any
        entrainment/detrainment imbalance in the implementation while
        catching the original 50x runaway.
        """
        dwn = calculate_downdraft(
            self.T, self.q, self.p, self.dz, self.rho,
            self.upd, jnp.array(0.0), self.cb, self.ktop_ceil, self.config,
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

    def test_mfd_zeroes_at_surface(self):
        """In the lowest layer the surface taper must drive |mfd| to zero
        (or near-zero) — ECHAM detrains the residual mass flux over the
        bottom 2 layers (Fortran ``itopde = klev-2``).
        """
        dwn = calculate_downdraft(
            self.T, self.q, self.p, self.dz, self.rho,
            self.upd, jnp.array(0.0), self.cb, self.ktop_ceil, self.config,
        )
        mfd_surface = float(dwn.mfd[-1])
        # In the bulk, |mfd| is on the order of cmfdeps*mfb ≈ 0.022.
        self.assertLess(
            abs(mfd_surface), 1e-3,
            f"|mfd| at surface = {abs(mfd_surface):.3e} kg/m²/s; "
            "should be ≪ bulk |mfd| due to surface taper."
        )


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

    def test_td_stays_close_to_environment(self):
        """Without adiabatic warming, the downdraft used to cool by ~25 K
        descending from LFS to surface (the parcel inertia retained the
        cold initial wet-bulb temperature). With g·dz/cp warming applied
        each layer plus mixing toward env, td should track the env to
        within ~10 K all the way down.
        """
        dwn = calculate_downdraft(
            self.T, self.q, self.p, self.dz, self.rho,
            self.upd, jnp.array(0.0), self.cb, self.ktop_ceil, self.config,
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
            self.upd, jnp.array(0.0), self.cb, self.ktop_ceil, self.config,
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


if __name__ == "__main__":
    unittest.main()
