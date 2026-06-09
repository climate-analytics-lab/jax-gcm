"""End-to-end: JAM online aerosol optics feeding RRTMGP (#495)."""

import unittest

import jax.numpy as jnp
import numpy as np
import pytest


class OpticsWiringTest(unittest.TestCase):
    def test_rrtmgp_jam_composition_builds(self):
        # Composition + ordering validation (JamOpticsTerm requires the
        # MACv2-SP ``aerosol`` diagnostic and ``_jam_state``).
        from jcm.physics.echam.echam_terms import echam_physics

        phys = echam_physics(
            aerosol_module="jam", cloud_scheme="2m", radiation_scheme="rrtmgp",
        )
        cats = [t.category for t in phys.terms]
        self.assertIn("aerosol_optics", cats)
        # optics must precede radiation (it writes the aerosol optics it reads).
        self.assertLess(cats.index("aerosol_optics"), cats.index("radiation"))


@pytest.mark.slow
class OpticsIntegrationTest(unittest.TestCase):
    def test_jam_optics_rrtmgp_runs_finite(self):
        from jcm.model import Model
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.terrain import TerrainData
        from jcm.utils import get_coords

        coords = get_coords(np.linspace(0, 1, 21), spectral_truncation=21)
        model = Model(
            coords=coords, time_step=30,
            terrain=TerrainData.aquaplanet(coords),
            physics=echam_physics(
                aerosol_module="jam", cloud_scheme="2m",
                radiation_scheme="rrtmgp",
            ),
        )
        predictions = model.run(save_interval=0.0625, total_time=0.0625)
        dyn = predictions.dynamics
        self.assertFalse(bool(jnp.any(jnp.isnan(dyn.temperature))))
        self.assertTrue(bool(jnp.all(dyn.temperature > 150.0)))
        self.assertTrue(bool(jnp.all(dyn.temperature < 360.0)))


if __name__ == "__main__":
    unittest.main()
