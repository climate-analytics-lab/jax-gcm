"""Tests for the SPA-style cloud-droplet activation floor (#374)."""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.macv2_sp_params import AerosolParameters
from jcm.physics.aerosol.spa import spa_activated_cdnc


# Lin et al. (2025) reference values — used as the AerosolParameters
# defaults; pinned here so the unit tests catch any unintended drift.
LIN2025_PREFACTOR = 2000.0
LIN2025_EXPONENT = 0.55


def _fit(prefactor=LIN2025_PREFACTOR, exponent=LIN2025_EXPONENT):
    return jnp.array(prefactor), jnp.array(exponent)


class TestSpaActivatedCdnc(unittest.TestCase):

    def test_zero_cloud_fraction_gives_zero_floor(self):
        """No cloud → no droplets to activate."""
        prefactor, exponent = _fit()
        Nccn = jnp.array([100.0, 500.0, 1000.0])
        cf = jnp.zeros_like(Nccn)
        nc = spa_activated_cdnc(Nccn, cf, prefactor, exponent)
        self.assertTrue(jnp.allclose(nc, 0.0))

    def test_zero_ccn_gives_zero_floor(self):
        """No CCN → no droplets, regardless of cloud fraction."""
        prefactor, exponent = _fit()
        Nccn = jnp.zeros(3)
        cf = jnp.array([0.1, 0.5, 1.0])
        nc = spa_activated_cdnc(Nccn, cf, prefactor, exponent)
        self.assertTrue(jnp.allclose(nc, 0.0))

    def test_units_returned_in_per_m3(self):
        """The function takes Nccn in cm^-3 and returns Nc in m^-3.

        Pin a single point against the bare formula to catch a units
        regression.
        """
        prefactor, exponent = _fit()
        Nccn_cm3 = jnp.array(500.0)        # 500 CCN per cc
        cf = jnp.array(1.0)                # full cloud
        nc = float(spa_activated_cdnc(Nccn_cm3, cf, prefactor, exponent))
        expected_cm3 = LIN2025_PREFACTOR * (500.0) ** LIN2025_EXPONENT
        expected_m3 = expected_cm3 * 1.0e6
        # float32 precision: ~1e-7 relative, so check ratio not absolute.
        self.assertAlmostEqual(nc / expected_m3, 1.0, places=4)

    def test_sublinear_in_ccn(self):
        """A 4× increase in CCN should give substantially less than a 4×
        increase in activated droplets — that's the whole point of the
        sublinear fit. With exponent 0.55 the ratio is 4^0.55 ≈ 2.18.
        """
        prefactor, exponent = _fit()
        cf = jnp.array(1.0)
        nc1 = float(spa_activated_cdnc(jnp.array(100.0), cf, prefactor, exponent))
        nc4 = float(spa_activated_cdnc(jnp.array(400.0), cf, prefactor, exponent))
        ratio = nc4 / nc1
        self.assertGreater(ratio, 2.0)
        self.assertLess(ratio, 2.4)
        self.assertAlmostEqual(ratio, 4.0 ** LIN2025_EXPONENT, places=4)

    def test_default_params_are_lin2025(self):
        """`AerosolParameters.default()` should ship the Lin (2025) fit
        values; downstream uses pull them from `parameters.aerosol.spa_*`.
        """
        params = AerosolParameters.default()
        self.assertAlmostEqual(float(params.spa_prefactor), LIN2025_PREFACTOR, places=4)
        self.assertAlmostEqual(float(params.spa_exponent), LIN2025_EXPONENT, places=4)
        # Observational slope band (Lin 2025).
        self.assertGreaterEqual(float(params.spa_exponent), 0.3)
        self.assertLessEqual(float(params.spa_exponent), 0.8)

    def test_jit_compatible(self):
        """SPA helper must be JAX-traceable."""
        prefactor, exponent = _fit()
        f = jax.jit(spa_activated_cdnc)
        nc = f(jnp.ones(4) * 200.0, jnp.ones(4) * 0.5, prefactor, exponent)
        self.assertTrue(np.all(np.isfinite(nc)))
        self.assertTrue(np.all(np.asarray(nc) > 0.0))

    def test_broadcasts_to_per_level(self):
        """Column Nccn `(ncols,)` broadcast against per-level cloud
        fraction `(nlev, ncols)` should give a per-level Nc floor.
        """
        prefactor, exponent = _fit()
        ncols, nlev = 5, 8
        Nccn = jnp.ones(ncols) * 300.0
        cf = jnp.ones((nlev, ncols)) * 0.4
        nc = spa_activated_cdnc(Nccn[jnp.newaxis, :], cf, prefactor, exponent)
        self.assertEqual(nc.shape, (nlev, ncols))
        self.assertTrue(jnp.all(nc > 0.0))

    def test_differentiable_through_prefactor_and_exponent(self):
        """Both fit parameters need a non-zero gradient when fed through
        `spa_activated_cdnc`. Pin the differentiability so a future
        refactor can't accidentally make either parameter constant.
        """
        Nccn = jnp.array(500.0)
        cf = jnp.array(0.7)

        def loss(prefactor, exponent):
            return jnp.sum(spa_activated_cdnc(Nccn, cf, prefactor, exponent))

        grad_pre, grad_exp = jax.grad(loss, argnums=(0, 1))(
            jnp.array(LIN2025_PREFACTOR), jnp.array(LIN2025_EXPONENT),
        )
        self.assertTrue(jnp.isfinite(grad_pre) & (grad_pre != 0.0))
        self.assertTrue(jnp.isfinite(grad_exp) & (grad_exp != 0.0))


if __name__ == "__main__":
    unittest.main()
