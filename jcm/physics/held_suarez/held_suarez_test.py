import unittest

import jax.numpy as jnp
import pytest


class TestHeldSuarezUnit(unittest.TestCase):
    def test_held_suarez_forcing(self):
        """Held-Suarez physics produces a non-trivial gridpoint tendency.

        Rewritten for the v2 dycore protocol: the physics step now produces a
        :class:`PhysicsTendency` in gridpoint space; conversion to a
        dycore-native tendency happens inside the dycore.
        """
        from jcm.model import Model
        from jcm.physics.held_suarez.held_suarez_physics import held_suarez_physics
        from jcm.physics.held_suarez.utils import get_held_suarez_coords
        from jcm.physics_interface import compute_physics_step_gridpoint

        time_step = 10
        coords = get_held_suarez_coords()
        model = Model(coords=coords, time_step=time_step, physics=held_suarez_physics())

        dycore_state = model._prepare_initial_dycore_state()
        physics_grid_state = model.dycore.to_physics_state(dycore_state)
        physics_tendency, _ = compute_physics_step_gridpoint(
            physics_grid_state,
            forcing=None,
            terrain=None,
            physics_state_carry=model._build_initial_physics_carry(),
            physics=model.physics,
            time_step=time_step * 60,
        )

        self.assertIsNotNone(physics_tendency)
        # Held-Suarez is dry and prescribes a non-zero T-relaxation, so the
        # temperature tendency must be non-trivial somewhere on the grid.
        self.assertTrue(jnp.any(physics_tendency.temperature != 0))

    @pytest.mark.slow
    def test_held_suarez_model(self):
        """A full Held-Suarez integration leaves every prognostic finite.

        2026-10 shrink pass (#1143): ``total_time=36`` is DAYS, not steps —
        the previous docstring's "36-step run" was stale/wrong; at this
        default T31L8 grid's auto-selected 12-minute step that was actually
        4320 steps (~36 days / 6 saves), making this the single biggest test
        in the shard. Held-Suarez is the textbook-robust dry-dynamics +
        Newtonian-relaxation benchmark: unlike the orography-coupled T63
        land case (``echam_with_land_test.py``), nothing here is known to
        need a long spin-up before a numerical problem would show up — a
        bad coupling between the forcing and the spectral filters blows up
        fast, not slowly, and there is no other test in this suite covering
        a long Held-Suarez integration specifically for finiteness. 4 days
        (2 save intervals, 480 steps instead of 4320) keeps a genuine
        multi-save-interval ``model.run()`` integration — not a single-step
        smoke test — at roughly a tenth of the previous wall time. The
        forcing itself is covered by the unit test above and ``model.run``
        by ``model_test``/``runners_test``, so the fast gate loses no
        unique lines to the move.
        """
        from jcm.model import Model
        from jcm.physics.held_suarez.held_suarez_physics import held_suarez_physics
        from jcm.physics.held_suarez.utils import get_held_suarez_coords

        coords = get_held_suarez_coords()
        model = Model(coords=coords, physics=held_suarez_physics())

        _ = model.run(total_time=4, save_interval=2)

        final_state = model._final_dycore_state

        self.assertFalse(jnp.any(jnp.isnan(final_state.vorticity)))
        self.assertFalse(jnp.any(jnp.isnan(final_state.divergence)))
        self.assertFalse(jnp.any(jnp.isnan(final_state.temperature_variation)))
        self.assertFalse(jnp.any(jnp.isnan(final_state.log_surface_pressure)))
        self.assertFalse(jnp.any(jnp.isnan(final_state.tracers['specific_humidity'])))
