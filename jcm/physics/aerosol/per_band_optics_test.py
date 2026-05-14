"""Tests for the wavelength-dependent aerosol optical-property scaling.

Locks in the ``per_band_optical_properties`` formulas from Stevens 2017
(``mo_bc_aeropt_splumes.f90:378-397``) — verifying both identity at the
550 nm reference and the expected directional changes at other bands.
"""

import unittest

import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.macv2_sp import per_band_optical_properties

# Band-center wavelengths (nm) of the standard 14-band RRTMGP SW
# gas-optics file, used here to smoke-test the wavelength scaling at a
# realistic band set. The production code reads these from the live
# ``RadiationBandConfig`` owned by ``ComposablePhysics`` — this constant
# is test-local so the formula tests don't depend on the runtime config.
RRTMGP_SW_BAND_CENTERS_NM = (
    5714.3, 3372.7, 2758.6, 2312.1, 2040.8, 1769.9, 1444.0, 1269.8,
     956.9,  693.2,  517.5,  387.2,  298.5,  227.3,
)


class TestPerBandOptics(unittest.TestCase):

    def setUp(self):
        # Single column / single level reference values.
        self.aod550 = jnp.array(0.2)
        self.ssa550 = jnp.array(0.95)
        self.asy550 = jnp.array(0.65)
        self.angstrom = jnp.array(1.5)

    def test_identity_at_550nm(self):
        bands = jnp.array([550.0])
        aod, ssa, asy = per_band_optical_properties(
            self.aod550, self.ssa550, self.asy550, self.angstrom, bands,
        )
        np.testing.assert_allclose(np.asarray(aod).ravel(), [0.20], atol=1e-6)
        np.testing.assert_allclose(np.asarray(ssa).ravel(), [0.95], atol=1e-3)
        np.testing.assert_allclose(np.asarray(asy).ravel(), [0.65], atol=1e-3)

    def test_aod_decreases_with_wavelength_at_positive_angstrom(self):
        """``aod(λ) = aod550·exp(−angstrom·ln(λ/550))`` ⇒ red-shift drops AOD."""
        bands = jnp.array([350.0, 550.0, 1000.0])
        aod, _, _ = per_band_optical_properties(
            self.aod550, self.ssa550, self.asy550, self.angstrom, bands,
        )
        a = np.asarray(aod).ravel()
        self.assertGreater(a[0], a[1])   # 350 nm > 550 nm
        self.assertGreater(a[1], a[2])   # 550 nm > 1000 nm

    def test_ssa_lfactor_saturates_at_short_wavelengths(self):
        """``lfactor = min(1, 700/λ)`` ⇒ ssa unchanged below 700 nm."""
        bands = jnp.array([350.0, 550.0, 700.0])
        _, ssa, _ = per_band_optical_properties(
            self.aod550, self.ssa550, self.asy550, self.angstrom, bands,
        )
        s = np.asarray(ssa).ravel()
        np.testing.assert_allclose(s[0], s[1], atol=1e-6)
        np.testing.assert_allclose(s[1], s[2], atol=1e-6)

    def test_ssa_drops_at_long_wavelengths(self):
        """ssa(1500 nm) < ssa550 (more absorption relative to scattering)."""
        bands = jnp.array([550.0, 1500.0])
        _, ssa, _ = per_band_optical_properties(
            self.aod550, self.ssa550, self.asy550, self.angstrom, bands,
        )
        s = np.asarray(ssa).ravel()
        self.assertLess(s[1], s[0])

    def test_asy_scaling_matches_closed_form_at_1um(self):
        """``asy(λ) = asy550·√lfactor`` ⇒ at 1000 nm, asy = 0.65·√0.7."""
        bands = jnp.array([1000.0])
        _, _, asy = per_band_optical_properties(
            self.aod550, self.ssa550, self.asy550, self.angstrom, bands,
        )
        expected = float(self.asy550) * np.sqrt(0.7)
        np.testing.assert_allclose(np.asarray(asy).ravel()[0], expected, atol=1e-3)

    def test_full_rrtmgp_sw_band_set(self):
        """Smoke-test against the actual 14 RRTMGP SW band centers."""
        bands = jnp.asarray(RRTMGP_SW_BAND_CENTERS_NM)
        aod, ssa, asy = per_band_optical_properties(
            self.aod550, self.ssa550, self.asy550, self.angstrom, bands,
        )
        a, s, g = (np.asarray(x).ravel() for x in (aod, ssa, asy))
        self.assertEqual(len(a), 14)
        # All values must be physical:
        self.assertTrue(np.all(a > 0))
        self.assertTrue(np.all((0 < s) & (s <= 1)))
        self.assertTrue(np.all((0 < g) & (g < 1)))

    def test_broadcasts_to_3d_input(self):
        """``aod550`` shaped ``(nlev, ncols)`` ⇒ output ``(n_bnd, nlev, ncols)``."""
        nlev, ncols = 5, 4
        aod550 = jnp.full((nlev, ncols), 0.2)
        ssa550 = jnp.full((nlev, ncols), 0.95)
        asy550 = jnp.full((nlev, ncols), 0.65)
        ang    = jnp.full((1, ncols), 1.5)
        bands  = jnp.asarray(RRTMGP_SW_BAND_CENTERS_NM)
        aod, ssa, asy = per_band_optical_properties(
            aod550, ssa550, asy550, ang, bands,
        )
        self.assertEqual(aod.shape, (14, nlev, ncols))
        self.assertEqual(ssa.shape, (14, nlev, ncols))
        self.assertEqual(asy.shape, (14, nlev, ncols))


if __name__ == "__main__":
    unittest.main()
