"""Phase 7: end-to-end JAM harness integration on a T21 aquaplanet.

Runs the full ECHAM stack with ``aerosol_module="jam"`` + the 2-moment cloud
scheme for a few steps and checks the model stays finite, the prognostic
aerosol tracers are seeded and remain physical, and ARG activation feeds the
2M scheme. Marked slow.
"""

import unittest

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.slow
class JamIntegrationTest(unittest.TestCase):
    def _run(self, **physics_kwargs):
        from jcm.model import Model
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.terrain import TerrainData
        from jcm.utils import get_coords

        sigma_boundaries = np.linspace(0, 1, 21)  # 20 layers
        coords = get_coords(sigma_boundaries, spectral_truncation=21)
        terrain = TerrainData.aquaplanet(coords)
        model = Model(
            coords=coords,
            time_step=30,
            terrain=terrain,
            physics=echam_physics(
                aerosol_module="jam", cloud_scheme="2m", **physics_kwargs
            ),
        )
        # ~1.5 h (3 steps) — enough to exercise tracer transport + coupling.
        return model, model.run(save_interval=0.0625, total_time=0.0625)

    def test_runs_finite_with_ham_aerosol(self):
        from jcm.physics.aerosol.jam import MAM4_SPEC, mass_name, number_name

        model, predictions = self._run()
        dyn = predictions.dynamics

        # Core dynamics stay finite and physical.
        self.assertFalse(bool(jnp.any(jnp.isnan(dyn.temperature))))
        self.assertFalse(bool(jnp.any(jnp.isnan(dyn.specific_humidity))))
        self.assertTrue(bool(jnp.all(dyn.temperature > 150.0)))
        self.assertTrue(bool(jnp.all(dyn.temperature < 360.0)))

        # JAM aerosol tracers were seeded and stay finite and bounded. Small
        # negatives are expected (spectral-transport Gibbs ringing around the
        # sharp surface emission source on a near-zero field); the meaningful
        # checks are finiteness and no blow-up, not strict non-negativity.
        tracers = dyn.tracers
        mname = mass_name(MAM4_SPEC.modes[0].species[0], MAM4_SPEC.modes[0].short)
        nname = number_name(MAM4_SPEC.modes[0].short)
        for key, bound in ((mname, 1.0), (nname, 1.0e15)):
            self.assertIn(key, tracers)
            arr = np.asarray(tracers[key])
            self.assertTrue(np.all(np.isfinite(arr)))
            self.assertLess(float(np.max(np.abs(arr))), bound)

        # The docstring's actual claim — activation feeds the 2M scheme —
        # must hold: the activated-CDNC diagnostic the 2M term consumes is
        # present and positive somewhere (ARG, or the ECHAM-HAM cdnc_min floor
        # the JAM path falls back to where ARG is empty, #640), and the 2M
        # droplet-number tracer has been populated in response. A run where the
        # coupling silently no-ops passes every finiteness check above but
        # fails here (measured on a healthy cold-start run: activated_cdnc max
        # ~66, qnc max ~3e-5 after 3 steps).
        physics = predictions.physics
        self.assertIn("activated_cdnc", physics)
        self.assertGreater(
            float(np.max(np.asarray(physics["activated_cdnc"]))), 0.0,
            "activation never produced droplets (ARG + cdnc_min floor zero)",
        )
        # The shared aerosol slot is present and finite. MACv2-SP was removed
        # from the JAM path (#640), so JAM carries no prescribed-plume Nccn —
        # it stays zero (the cdnc_min fallback, not Nccn, drives activation).
        self.assertIn("aerosol", physics)
        self.assertTrue(
            bool(np.all(np.isfinite(np.asarray(physics["aerosol"].Nccn))))
        )
        self.assertGreater(
            float(np.max(np.asarray(tracers["qnc"]))), 0.0,
            "2M droplet-number tracer untouched — activation not coupled",
        )

    def test_ghosh_variant_also_runs(self):
        _, predictions = self._run(jam_arg_variant="ghosh2025")
        self.assertFalse(
            bool(jnp.any(jnp.isnan(predictions.dynamics.temperature)))
        )

    def test_carry_stored_cloud_borne_runs_and_cycles(self):
        # The default (and only) storage (#602 item 3): cloud-borne lives
        # in the physics carry, never the dycore tracer set, and the model
        # runs finite with the carry fields reaching saved output.
        _, predictions = self._run()
        tracers = predictions.dynamics.tracers
        self.assertFalse(
            any(k.startswith(("mc_", "nc_")) for k in tracers),
            "carry mode must not declare mirror tracers",
        )
        self.assertFalse(
            bool(jnp.any(jnp.isnan(predictions.dynamics.temperature)))
        )
        carry = predictions.physics.get("_jam_cloud_borne")
        self.assertIsNotNone(carry, "carry store missing from diagnostics")
        self.assertTrue(carry, "carry store empty")
        for k, v in carry.items():
            self.assertTrue(np.all(np.isfinite(np.asarray(v))), k)

    def test_cloud_borne_off_drops_mirrors_and_runs(self):
        # The A/B switch (#602): with jam_cloud_borne=False the mirror
        # tracers are not declared at all — the transported set halves —
        # and the model still runs finite with the implicit scavenging.
        _, predictions = self._run(jam_cloud_borne=False)
        tracers = predictions.dynamics.tracers
        self.assertFalse(
            any(k.startswith(("mc_", "nc_")) for k in tracers),
            "cloud-borne mirrors must not be transported when off",
        )
        self.assertFalse(
            bool(jnp.any(jnp.isnan(predictions.dynamics.temperature)))
        )


if __name__ == "__main__":
    unittest.main()
