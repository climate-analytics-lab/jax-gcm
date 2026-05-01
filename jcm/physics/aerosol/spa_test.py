"""Tests for the SPA-style cloud-droplet activation floor (#374)."""

import jax.numpy as jnp
import jax
import numpy as np
import unittest

from jcm.physics.aerosol.spa import (
    SPA_EXPONENT,
    SPA_PREFACTOR,
    spa_activated_cdnc,
)


class TestSpaActivatedCdnc(unittest.TestCase):

    def test_zero_cloud_fraction_gives_zero_floor(self):
        """No cloud → no droplets to activate."""
        Nccn = jnp.array([100.0, 500.0, 1000.0])
        cf = jnp.zeros_like(Nccn)
        nc = spa_activated_cdnc(Nccn, cf)
        self.assertTrue(jnp.allclose(nc, 0.0))

    def test_zero_ccn_gives_zero_floor(self):
        """No CCN → no droplets, regardless of cloud fraction."""
        Nccn = jnp.zeros(3)
        cf = jnp.array([0.1, 0.5, 1.0])
        nc = spa_activated_cdnc(Nccn, cf)
        self.assertTrue(jnp.allclose(nc, 0.0))

    def test_units_returned_in_per_m3(self):
        """The function takes Nccn in cm^-3 and returns Nc in m^-3.

        Pin a single point against the bare formula to catch a units
        regression.
        """
        Nccn_cm3 = jnp.array(500.0)        # 500 CCN per cc
        cf = jnp.array(1.0)                # full cloud
        nc = float(spa_activated_cdnc(Nccn_cm3, cf))
        expected_cm3 = SPA_PREFACTOR * (500.0) ** SPA_EXPONENT
        expected_m3 = expected_cm3 * 1.0e6
        # float32 precision: ~1e-7 relative, so check ratio not absolute.
        self.assertAlmostEqual(nc / expected_m3, 1.0, places=4)

    def test_sublinear_in_ccn(self):
        """A 4× increase in CCN should give substantially less than a 4×
        increase in activated droplets — that's the whole point of the
        sublinear fit. With exponent 0.55 the ratio is 4^0.55 ≈ 2.18.
        """
        cf = jnp.array(1.0)
        nc1 = float(spa_activated_cdnc(jnp.array(100.0), cf))
        nc4 = float(spa_activated_cdnc(jnp.array(400.0), cf))
        ratio = nc4 / nc1
        self.assertGreater(ratio, 2.0)
        self.assertLess(ratio, 2.4)
        self.assertAlmostEqual(ratio, 4.0 ** SPA_EXPONENT, places=4)

    def test_observational_slope_band(self):
        """The d(ln Nc) / d(ln Nccn) slope should land in the 0.3 — 0.8 band
        cited by Lin et al. (2025) as observationally constrained.
        """
        # The slope is exactly the SPA exponent; this is a contract test
        # against the constant.
        self.assertGreaterEqual(SPA_EXPONENT, 0.3)
        self.assertLessEqual(SPA_EXPONENT, 0.8)

    def test_jit_compatible(self):
        """SPA helper must be JAX-traceable."""
        f = jax.jit(spa_activated_cdnc)
        nc = f(jnp.ones(4) * 200.0, jnp.ones(4) * 0.5)
        self.assertTrue(np.all(np.isfinite(nc)))
        self.assertTrue(np.all(np.asarray(nc) > 0.0))

    def test_broadcasts_to_per_level(self):
        """Column Nccn `(ncols,)` broadcast against per-level cloud
        fraction `(nlev, ncols)` should give a per-level Nc floor.
        """
        ncols, nlev = 5, 8
        Nccn = jnp.ones(ncols) * 300.0
        cf = jnp.ones((nlev, ncols)) * 0.4
        nc = spa_activated_cdnc(Nccn[jnp.newaxis, :], cf)
        self.assertEqual(nc.shape, (nlev, ncols))
        self.assertTrue(jnp.all(nc > 0.0))


if __name__ == "__main__":
    unittest.main()
