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
        from jcm.physics.radiation.radiation_types import RadiationParameters
        from jcm.terrain import TerrainData
        from jcm.utils import get_coords

        coords = get_coords(np.linspace(0, 1, 21), spectral_truncation=21)
        model = Model(
            coords=coords, time_step=30,
            terrain=TerrainData.aquaplanet(coords),
            # radiation_interval=0: per-step radiation AND per-step band
            # optics. With the default 2 h gate this 3-step test saves a
            # frame that replays the step-0 (cold-start, zero-aerosol)
            # optics cache, so the AOD assertion below reads an exact zero
            # regardless of wiring. The gate's replay semantics have their
            # own unit test (optics_term_test); this test wants the wiring.
            physics=echam_physics(
                aerosol_module="jam", cloud_scheme="2m",
                radiation_scheme="rrtmgp",
                radiation=RadiationParameters.default(radiation_interval=0),
            ),
        )
        predictions = model.run(save_interval=0.0625, total_time=0.0625)
        dyn = predictions.dynamics
        self.assertFalse(bool(jnp.any(jnp.isnan(dyn.temperature))))
        self.assertTrue(bool(jnp.all(dyn.temperature > 150.0)))
        self.assertTrue(bool(jnp.all(dyn.temperature < 360.0)))

        # The column AOD (~550 nm) diagnostic surfaces in the physics output,
        # finite, non-negative, and STRICTLY positive somewhere: three steps
        # of surface emissions must reach the optics chain (measured healthy
        # cold-start max ≈ 1.5e-8 — tiny, but an exact zero means the
        # emissions→microphysics→optics wiring silently dropped out, which
        # the old ``aod >= 0`` bound could not distinguish from healthy).
        # A stronger on/off flux-difference test needs a seeded aerosol
        # burden (cold-start AOD is radiatively invisible) — deferred.
        aod = predictions.physics["aerosol_optical_depth"]
        aod = np.asarray(aod)
        self.assertTrue(np.all(np.isfinite(aod)))
        self.assertTrue(np.all(aod >= 0.0))
        self.assertGreater(float(aod.max()), 0.0,
                           "AOD identically zero — optics chain unwired")

        # And RRTMGP genuinely ran with the optics in the loop: daytime
        # columns received shortwave at the surface.
        rad = predictions.physics["radiation"]
        self.assertGreater(float(np.max(np.asarray(rad.surface_sw_down))), 0.0)


if __name__ == "__main__":
    unittest.main()
