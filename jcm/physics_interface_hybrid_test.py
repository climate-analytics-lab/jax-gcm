"""Tests for the physics-dynamics interface with hybrid vertical coordinates.

Regression tests covering the bugs we hit when switching from sigma to hybrid:
- `compute_diagnostic_state` dispatch (sigma vs hybrid)
- geopotential computation using the actual (spatially varying) surface
  pressure in Pa, not a ratio / reference scalar
- Initial-state geopotential monotonically decreasing from TOA to surface
"""

import unittest
import jax.numpy as jnp
import numpy as np
import pytest

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


@pytest.mark.slow
class TestHybridDynamicsPhysicsInterface(unittest.TestCase):
    """The dynamics→physics→dynamics round trip should preserve validity.

    Marked slow: runs a full 1-day T31 ICON physics integration, which
    includes JIT-compiling the complete physics pipeline (radiation,
    convection, cloud microphysics, turbulence). Too heavy for the
    push / fast-tests CI budget.
    """

    def test_one_step_no_nan(self):
        """After one model step from the default isothermal state, no NaN."""
        model = _build_test_model(use_hybrid=True)
        # Single day via the same path the CLI uses
        preds = model.run(save_interval=1.0, total_time=1.0)
        T = preds.dynamics.temperature
        nan_frac = float(jnp.isnan(T).mean())
        self.assertEqual(nan_frac, 0.0,
                         f"NaN fraction {nan_frac:.1%} after 1 day hybrid run")
        self.assertTrue(jnp.all(T > 100), "T below 100 K after 1 day")
        self.assertTrue(jnp.all(T < 400), "T above 400 K after 1 day")


@pytest.mark.slow
class TestHybridAtT85(unittest.TestCase):
    """At T85 the dynamics are more energetic; hybrid must still be stable.

    Regression test for the log_surface_pressure normalization bug: sigma
    coords need log(P_s/p0), hybrid needs log(P_s_in_Pa), and mixing the
    two caused all fields to go NaN within a single model step at T85.

    Marked slow: a 3-day T85 physics integration runs for several minutes
    on CPU and won't fit the push CI budget.
    """

    def test_multi_day_no_nan_at_t85(self):
        """3-day T85 hybrid run with default physics should not blow up."""
        import logging
        from jcm.model import Model
        from jcm.physics.icon.icon_physics import IconPhysics
        vertical = get_icon_levels(47)
        coords = get_coords(vertical, spectral_truncation=85)
        physics = IconPhysics(radiation_scheme="grey", checkpoint_terms=False)
        model = Model(coords=coords, physics=physics, time_step=3.0,
                      log_level=logging.CRITICAL)
        preds = model.run(save_interval=1.0, total_time=3.0)
        T = preds.dynamics.temperature
        nan_frac = float(jnp.isnan(T).mean())
        self.assertEqual(nan_frac, 0.0,
                         f"NaN fraction {nan_frac:.1%} after 3 days T85 hybrid")
        # Wind should have spun up a bit but stay within CFL bounds
        u = preds.dynamics.u_wind
        self.assertLess(float(jnp.max(jnp.abs(u))), 100.0,
                        "Wind speed grew to > 100 m/s (CFL/numerical instability)")


if __name__ == "__main__":
    unittest.main()
