"""The aerosol scavenging interface the Tiedtke scheme publishes (HAMMOZ).

HAMMOZ's convective wet deposition reads two ratios from the ascent and the
precipitation budget: the precipitation efficiency ``peff =
pmrateprecip/pmwc`` of each plume level (cuasc's condensate before and
after conversion, split by phase in ``prep_wetdep_hydro``) and the
evaporation fraction ``prevap`` of the falling precipitation. The scheme
publishes both in ``ConvectionData``; these tests pin ``peff``.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

import jcm.constants as c
from jcm.physics.convection.tiedtke_nordeng.half_level_ledger_test import (
    _l47_tropical_column,
    _run,
)
from jcm.physics.convection.tiedtke_nordeng.updraft import (
    ham_precip_efficiency,
)


class TestHamPrecipEfficiency(unittest.TestCase):
    def test_warm_level_is_the_converted_fraction(self):
        peff = ham_precip_efficiency(jnp.asarray(2.0e-3), jnp.asarray(8.0e-4),
                                     jnp.asarray(c.tmelt + 5.0))
        np.testing.assert_allclose(float(peff), 0.6, rtol=1e-6)

    def test_mixed_phase_level_with_both_phases_is_the_ratio(self):
        # Both phases above zmin: peffwat = peffice = (plu − zlnew)/plu and
        # the phase weights sum to one.
        peff = ham_precip_efficiency(jnp.asarray(2.0e-3), jnp.asarray(8.0e-4),
                                     jnp.asarray(c.tmelt - 15.0))
        np.testing.assert_allclose(float(peff), 0.6, rtol=1e-6)

    def test_phase_below_zmin_converts_nothing(self):
        # At −1 °C cuflx's zalpha leaves (1 − zalpha) ≈ 3e-3 of the
        # condensate as ice; with plu = 2e-8 that ice is below HAMMOZ's
        # zmin = 1e-10, so only the liquid share scavenges.
        lu, tu = 2.0e-8, c.tmelt - 1.0
        zalpha = 0.0059 + (1 - 0.0059) * np.exp(-0.003102)
        peff = ham_precip_efficiency(jnp.asarray(lu), jnp.asarray(lu * 0.4),
                                     jnp.asarray(tu))
        np.testing.assert_allclose(float(peff), 0.6 * zalpha, rtol=1e-5)

    def test_no_condensate_is_zero_with_finite_gradient(self):
        f = lambda lu: ham_precip_efficiency(lu, 0.5 * lu, jnp.asarray(280.0))
        self.assertEqual(float(f(jnp.asarray(0.0))), 0.0)
        self.assertTrue(np.isfinite(float(jax.grad(f)(jnp.asarray(0.0)))))


class TestPublishedPrecipEfficiency(unittest.TestCase):
    """The published ``peff`` equals ``pdmfup/(pdmfup + pmfu·plu)``.

    On the half levels (#886) ``pdmfup`` of layer k is ``(plu − zlnew)``
    times the precipitating (continuing) flux at its top interface, and the
    published ``plu`` there is the flux-weighted condensate of the
    continuing plume (after conversion) and the overshoot (before), so
    ``pdmfup + pmfu·plu = pmfu·plu_before`` and the reconstruction is the
    flux-weighted efficiency exactly wherever both phases exceed zmin.
    """

    def test_matches_the_ledger_reconstruction_on_the_l47_deep_column(self):
        T, q, p, p_half, dz, rho = _l47_tropical_column()
        tend, state = _run(T, q, p, p_half, dz, rho, deep=True)
        self.assertEqual(int(state.ktype), 1)
        peff = np.asarray(tend.precip_efficiency, dtype=np.float64)
        pf = np.asarray(tend.precip_formation, dtype=np.float64)
        lu = np.asarray(tend.qc_conv + tend.qi_conv, dtype=np.float64)
        mfu = np.asarray(state.mfu, dtype=np.float64)
        sel = (lu > 1.0e-6) & (mfu > 0.0)
        self.assertGreater(int(sel.sum()), 5)
        recon = pf[sel] / (pf[sel] + mfu[sel] * lu[sel])
        np.testing.assert_allclose(peff[sel], recon, rtol=2e-4, atol=1e-6)
        self.assertGreater(float(peff.max()), 0.3)
        self.assertTrue(np.all((peff >= 0.0) & (peff <= 1.0)))

    def test_prevap_published_and_bounded(self):
        T, q, p, p_half, dz, rho = _l47_tropical_column()
        tend, _ = _run(T, q, p, p_half, dz, rho, deep=True)
        evap = np.asarray(tend.precip_evap_fraction)
        self.assertEqual(evap.shape, np.asarray(T).shape)
        self.assertTrue(np.all((evap >= 0.0) & (evap <= 1.0)))


if __name__ == "__main__":
    unittest.main()
