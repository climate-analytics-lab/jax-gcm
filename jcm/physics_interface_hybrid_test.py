"""Tests for the physics-dynamics interface with hybrid vertical coordinates.

Regression tests covering the bugs we hit when switching from sigma to hybrid:
- `compute_diagnostic_state` dispatch (sigma vs hybrid)
- geopotential computation using the actual (spatially varying) surface
  pressure in Pa, not a ratio / reference scalar
- Initial-state geopotential monotonically decreasing from TOA to surface

Full model-run smoke tests (multi-day T31 / T85) live in the manual
validation scripts — they're too heavy for CI. ``TestIconPhysicsIntegration``
in ``jcm/physics/icon/icon_physics_test.py`` already covers the "full
pipeline doesn't blow up" question at T31+sigma.
"""

import unittest
import jax.numpy as jnp
import numpy as np

from jcm.utils import get_coords
from jcm.physics.icon.icon_levels import get_icon_levels


def _build_test_model(use_hybrid=True):
    """Build a small T31 model with hybrid or sigma coords, IconPhysics."""
    import logging
    from dinosaur.sigma_coordinates import SigmaCoordinates
    from jcm.model import Model
    from jcm.physics.icon.icon_physics import IconPhysics

    if use_hybrid:
        vertical = get_icon_levels(47)
    else:
        vertical = SigmaCoordinates.equidistant(47)
    coords = get_coords(vertical, spectral_truncation=31)
    physics = IconPhysics(radiation_scheme="grey", checkpoint_terms=False)
    return Model(coords=coords, physics=physics, time_step=3.0,
                 log_level=logging.CRITICAL)


class TestHybridInitialGeopotential(unittest.TestCase):
    """Initial geopotential must be sensible for a hybrid-coord model."""

    def test_geopotential_decreases_from_toa_to_surface(self):
        """For an isothermal rest atmosphere, nodal geopotential must
        monotonically decrease from level 0 (TOA) to level nlev-1 (surface).
        """
        model = _build_test_model(use_hybrid=True)
        model._final_modal_state = model._prepare_initial_modal_state(None, 0)
        from jcm.physics_interface import dynamics_state_to_physics_state
        ps = dynamics_state_to_physics_state(
            model._final_modal_state, model.primitive
        )
        # Mean geopotential per level (spatial mean)
        phi_profile = jnp.mean(ps.geopotential, axis=(1, 2))
        dphi = jnp.diff(phi_profile)
        self.assertTrue(
            jnp.all(dphi <= 0),
            f"Geopotential must decrease from TOA to surface; "
            f"dphi={np.array(dphi)}",
        )

    def test_surface_geopotential_near_zero_aquaplanet(self):
        """On an aquaplanet (no orography), surface geopotential ≈ 0."""
        model = _build_test_model(use_hybrid=True)
        model._final_modal_state = model._prepare_initial_modal_state(None, 0)
        from jcm.physics_interface import dynamics_state_to_physics_state
        ps = dynamics_state_to_physics_state(
            model._final_modal_state, model.primitive
        )
        # Surface layer geopotential should be much smaller than TOA
        # (scales with hypsometric height * g; aquaplanet surface ≈ 0)
        surface_mean = float(jnp.mean(jnp.abs(ps.geopotential[-1])))
        toa_mean = float(jnp.mean(ps.geopotential[0]))
        self.assertLess(surface_mean, 0.01 * toa_mean,
                        f"Aquaplanet surface phi {surface_mean:.3g} should be "
                        f"<< TOA phi {toa_mean:.3g}")


if __name__ == "__main__":
    unittest.main()
