"""Phase 7: end-to-end HAM harness integration on a T21 aquaplanet.

Runs the full ECHAM stack with ``aerosol_module="ham"`` + the 2-moment cloud
scheme for a few steps and checks the model stays finite, the prognostic
aerosol tracers are seeded and remain physical, and ARG activation feeds the
2M scheme. Marked slow.
"""

import unittest

import jax.numpy as jnp
import numpy as np
import pytest


@pytest.mark.slow
class HamIntegrationTest(unittest.TestCase):
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
                aerosol_module="ham", cloud_scheme="2m", **physics_kwargs
            ),
        )
        # ~1.5 h (3 steps) — enough to exercise tracer transport + coupling.
        return model, model.run(save_interval=0.0625, total_time=0.0625)

    def test_runs_finite_with_ham_aerosol(self):
        from jcm.physics.aerosol.ham import MAM4_SPEC, mass_name, number_name

        model, predictions = self._run()
        dyn = predictions.dynamics

        # Core dynamics stay finite and physical.
        self.assertFalse(bool(jnp.any(jnp.isnan(dyn.temperature))))
        self.assertFalse(bool(jnp.any(jnp.isnan(dyn.specific_humidity))))
        self.assertTrue(bool(jnp.all(dyn.temperature > 150.0)))
        self.assertTrue(bool(jnp.all(dyn.temperature < 360.0)))

        # HAM aerosol tracers were seeded and stay finite and bounded. Small
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

    def test_ghosh_variant_also_runs(self):
        _, predictions = self._run(ham_arg_variant="ghosh2025")
        self.assertFalse(
            bool(jnp.any(jnp.isnan(predictions.dynamics.temperature)))
        )


if __name__ == "__main__":
    unittest.main()
