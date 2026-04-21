"""Tests for the updraft saturation adjustment.

Regression tests covering the iterative Newton-Raphson saturation adjustment
that matches ECHAM/ICON `cuadjtq` (see `../../../../atm_phy_echam/mo_cuadjust.f90`).

The original JAX implementation was a single-pass saturation step which left
updraft parcels supersaturated between iterations, under-releasing latent
heating and giving unrealistic RCE temperature profiles.
"""

import unittest
import jax.numpy as jnp
import numpy as np

from jcm.physics.icon.convection.updraft import saturation_adjustment
from jcm.physics.icon.convection.tiedtke_nordeng import saturation_mixing_ratio


class TestSaturationAdjustmentNewton(unittest.TestCase):
    """cuadjtq-style Newton-Raphson saturation adjustment."""

    def _run(self, T, q, p):
        """Wrapper — returns (T_adj, vapor, liquid)."""
        return saturation_adjustment(
            jnp.asarray(T, dtype=jnp.float32),
            jnp.asarray(q, dtype=jnp.float32),
            jnp.asarray(p, dtype=jnp.float32),
        )

    def test_unsaturated_passes_through(self):
        """If q < qs(T), output should equal input (no condensation)."""
        T, p = 288.0, 101325.0
        q = 0.001  # ~1 g/kg, well below qsat ≈ 10 g/kg at 288K
        T_adj, vapor, liquid = self._run(T, q, p)
        self.assertAlmostEqual(float(T_adj), T, places=3)
        self.assertAlmostEqual(float(vapor), q, places=6)
        self.assertAlmostEqual(float(liquid), 0.0, places=6)

    def test_saturation_enforced(self):
        """After adjustment, vapor mixing ratio should equal qsat(T_adj)
        to within tight tolerance (the essence of proper `cuadjtq`)."""
        T, p = 288.0, 101325.0
        qsat_initial = float(saturation_mixing_ratio(
            jnp.asarray(p), jnp.asarray(T)
        ))
        total_q = 1.5 * qsat_initial  # 50% supersaturated
        T_adj, vapor, liquid = self._run(T, total_q, p)
        qsat_final = float(saturation_mixing_ratio(
            jnp.asarray(p), T_adj
        ))
        rel_error = abs(float(vapor) - qsat_final) / qsat_final
        self.assertLess(
            rel_error, 0.005,
            f"After adjustment, vapor={float(vapor):.6f} should equal "
            f"qsat(T_adj)={qsat_final:.6f} within 0.5%; "
            f"relative error = {rel_error:.3%}"
        )

    def test_mass_conservation(self):
        """Total water (vapor + liquid) must equal input total water."""
        T, p = 280.0, 80000.0
        total_q = 0.02  # 20 g/kg — strong supersaturation
        T_adj, vapor, liquid = self._run(T, total_q, p)
        total_out = float(vapor) + float(liquid)
        self.assertAlmostEqual(total_out, total_q, places=5)

    def test_latent_heating(self):
        """Condensation must warm the parcel; the latent heat released
        should match the energy budget cp*dT = L*d_condensed."""
        from jcm.physics.icon.constants.physical_constants import cp, alhc
        T, p = 290.0, 90000.0
        qsat_T = float(saturation_mixing_ratio(
            jnp.asarray(p), jnp.asarray(T)
        ))
        total_q = 2.0 * qsat_T  # Heavily supersaturated
        T_adj, vapor, liquid = self._run(T, total_q, p)
        dT = float(T_adj) - T
        expected_dT = alhc * float(liquid) / cp
        # Allow 2% tolerance for the Newton iteration's residual
        self.assertAlmostEqual(dT / expected_dT, 1.0, delta=0.02)
        self.assertGreater(dT, 0.0, "Condensation must warm the parcel")

    def test_convergence_strong_supersaturation(self):
        """Under very strong supersaturation, the iterative Newton scheme
        must still converge — the single-pass version under-converged here."""
        T, p = 300.0, 95000.0
        qsat_T = float(saturation_mixing_ratio(
            jnp.asarray(p), jnp.asarray(T)
        ))
        total_q = 5.0 * qsat_T  # 5x saturated — truly unphysical but exercises
                                 # the iteration's robustness
        T_adj, vapor, liquid = self._run(T, total_q, p)
        qsat_final = float(saturation_mixing_ratio(
            jnp.asarray(p), T_adj
        ))
        rel_error = abs(float(vapor) - qsat_final) / qsat_final
        self.assertLess(rel_error, 0.01,
                        f"High-supersaturation case: vapor={float(vapor):.6f} "
                        f"vs qsat(T_adj)={qsat_final:.6f}, "
                        f"relative error = {rel_error:.3%}")

    def test_batched(self):
        """The adjustment should vectorise correctly across multiple parcels."""
        T = jnp.asarray([285.0, 290.0, 295.0, 300.0])
        p = jnp.asarray([100000.0, 90000.0, 80000.0, 70000.0])
        qsat = saturation_mixing_ratio(p, T)
        q = 1.3 * qsat
        T_adj, vapor, liquid = saturation_adjustment(T, q, p)
        # Each parcel should satisfy vapor ≈ qsat(T_adj)
        qsat_final = saturation_mixing_ratio(p, T_adj)
        rel_err = jnp.abs(vapor - qsat_final) / qsat_final
        self.assertTrue(jnp.all(rel_err < 0.01),
                        f"Max rel error across batch: {float(rel_err.max()):.3%}")
        # Each parcel should warm
        self.assertTrue(jnp.all(T_adj > T))
        # Each parcel should have positive liquid
        self.assertTrue(jnp.all(liquid >= 0))


if __name__ == "__main__":
    unittest.main()
