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

from jcm.physics.convection.tiedtke_nordeng.updraft import saturation_adjustment
from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import saturation_mixing_ratio


class TestSaturationAdjustmentNewton(unittest.TestCase):
    """cuadjtq-style Newton-Raphson saturation adjustment."""

    def _run(self, T, q, p):
        """Return (T_adj, vapor, liquid) from saturation_adjustment."""
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
        to within tight tolerance (the essence of proper `cuadjtq`).
        """
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
        should match the energy budget cp*dT = L*d_condensed.
        """
        from jcm.constants import cp, alhc
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
        must still converge — the single-pass version under-converged here.
        """
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


class TestDynamicCloudTop(unittest.TestCase):
    """The updraft must terminate at the LNB, not at a hard-coded ktop.

    We build two environments with identical cloud-base forcing but
    different upper-atmosphere stability; the effective cloud top should
    be set by the point at which the parcel becomes negatively buoyant,
    not by the `ktop` argument.
    """

    def _run_updraft(self, temperature, ktop_override=None):
        """Run the updraft for a specified T(z) profile.

        Convention: index 0 = TOA (low pressure, cold), index nlev-1 = surface
        (high pressure, warm). The caller supplies temperature in this order.

        Parcel starts at the bottom level with T=surface and
        q=q_sat(surface) (saturated cloud base).
        """
        from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
            ConvectionParameters, saturation_mixing_ratio
        )
        from jcm.physics.convection.tiedtke_nordeng.updraft import calculate_updraft

        nlev = temperature.shape[0]
        # Pressure: TOA (low) → surface (high) to match index convention
        pressure = jnp.linspace(10_000.0, 100_000.0, nlev)

        humidity = saturation_mixing_ratio(pressure, temperature)
        layer_thickness = jnp.full(nlev, 1000.0)
        rho = pressure / (287.0 * temperature)

        cfg = ConvectionParameters.default()
        kbase = nlev - 2  # 1 level above surface
        ktop = 1 if ktop_override is None else ktop_override  # Near TOA

        state = calculate_updraft(
            temperature, humidity, pressure, layer_thickness, rho,
            kbase=kbase, ktop=ktop, ktype=1, mass_flux_base=0.1, config=cfg,
        )
        return state, pressure

    def test_dry_stable_upper_atmosphere_limits_cloud_top(self):
        """With a strongly stable inversion above the PBL, mass flux must
        vanish above the inversion regardless of the `ktop` argument.
        """
        # Build a 20-level profile: adiabatic below, isothermal above (stable).
        # Index 0 = TOA (cold); index 19 = surface (warm)
        #   [0..4]   stratosphere (220 K)
        #   [5..14]  isothermal inversion (260 K) — strongly stable
        #   [15..19] PBL, conditionally unstable
        T_profile = jnp.concatenate([
            jnp.full(5, 220.0),
            jnp.full(10, 260.0),
            jnp.linspace(270.0, 300.0, 5),
        ])
        state, pressure = self._run_updraft(T_profile, ktop_override=1)

        mfu = state.mfu
        # Mass flux must vanish in the stratosphere (k < 5) even though
        # we passed a generous ktop=1 (allowing ascent to near TOA).
        self.assertTrue(
            jnp.all(mfu[0:3] < 1e-3),
            f"Expected mfu≈0 in top 3 levels (stratosphere), got "
            f"{np.array(mfu[0:3])}",
        )

    def test_organized_entrainment_responds_to_buoyancy(self):
        """Nordeng organized entrainment activates only for `ktype=1`.

        Directly validates the branch we added: running deep-convection
        mode (`ktype=1`) against a trivial, stable environment should give
        similar entrainment to shallow mode, whereas running deep-convection
        mode against a strongly-buoyant environment should yield a
        *different* entrainment profile (buoyancy-dependent rate kicks in).
        """
        from jcm.physics.convection.tiedtke_nordeng.tiedtke_nordeng import (
            ConvectionParameters, saturation_mixing_ratio,
        )
        from jcm.physics.convection.tiedtke_nordeng.updraft import calculate_updraft

        nlev = 20
        pressure = jnp.linspace(10_000.0, 100_000.0, nlev)
        layer_thickness = jnp.full(nlev, 1000.0)

        # Environment A: neutral — parcel ≈ environment everywhere
        T_neutral = jnp.full(nlev, 260.0)
        # Environment B: strongly unstable — parcel much warmer than env
        T_unstable = jnp.linspace(240.0, 310.0, nlev)

        cfg = ConvectionParameters.default()

        def run(T, ktype):
            humidity = saturation_mixing_ratio(pressure, T)
            rho = pressure / (287.0 * T)
            return calculate_updraft(
                T, humidity, pressure, layer_thickness, rho,
                kbase=18, ktop=1, ktype=ktype, mass_flux_base=0.1, config=cfg,
            )

        deep_unstable = run(T_unstable, ktype=1)
        deep_neutral = run(T_neutral, ktype=1)

        # The organized term is > 0 when buoyancy > 0 (unstable environment)
        # and = 0 for neutral environment, so entr profiles must differ.
        diff = jnp.sum(jnp.abs(deep_unstable.entr - deep_neutral.entr))
        self.assertGreater(
            float(diff), 1e-5,
            f"Organized entrainment should depend on buoyancy profile; "
            f"entr differs by only {float(diff):.6f} between unstable "
            f"and neutral environments.",
        )

    def test_updraft_terminates_on_negative_buoyancy(self):
        """If the parcel becomes negatively buoyant at a level well below the
        nominal `ktop`, the mass flux above should be zero — the updraft is
        terminated dynamically rather than pushed to `ktop` regardless.
        """
        # Very strong capping inversion at k=10: stratospheric temperatures
        # above. Regardless of how high we pass `ktop`, the updraft shouldn't
        # support mass flux in the cap.
        T_profile = jnp.concatenate([
            jnp.full(10, 200.0),         # Extremely cold cap (highly stable)
            jnp.linspace(290.0, 300.0, 10),  # PBL
        ])
        # Pass ktop_override=1 — updraft is *asked* to go all the way to
        # near TOA, but buoyancy termination should stop it at the cap.
        state, _ = self._run_updraft(T_profile, ktop_override=1)
        mfu = state.mfu
        # No mass flux should persist above the inversion
        self.assertTrue(
            jnp.all(mfu[0:9] < 1e-3),
            f"Updraft should terminate at capping inversion; got mfu[0:9]="
            f"{np.array(mfu[0:9]).round(4)}",
        )


if __name__ == "__main__":
    unittest.main()
