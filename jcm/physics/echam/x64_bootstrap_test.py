"""An ECHAM-composed model bootstraps under ``jax_enable_x64`` (#927).

Importing ``veros`` flips ``jax_enable_x64`` process-wide, so an ECHAM
atmosphere coupled to a Veros ocean is built and bootstrapped with x64 on
and float32 physics. Bootstrap traces the full composed physics once
(``get_empty_data`` / ``initial_physics_carry``), which is where Tiedtke-
Nordeng's activation ``lax.cond`` used to reject its int32/int64 index
mismatch. The test scopes x64 to a context; the root ``conftest.py`` also
restores the session default after every test.
"""

import unittest

import jax
import jax.numpy as jnp
import jax.tree_util as tu
import numpy as np


class TestEchamBootstrapUnderX64(unittest.TestCase):

    def test_bootstrap_state_and_integer_carry_dtypes(self):
        from jcm.model import Model
        from jcm.physics.echam.echam_terms import echam_physics
        from jcm.terrain import TerrainData
        from jcm.utils import get_coords

        with jax.enable_x64():
            coords = get_coords(np.linspace(0, 1, 9), spectral_truncation=21)
            model = Model(
                coords=coords, time_step=30,
                terrain=TerrainData.aquaplanet(coords),
                physics=echam_physics(),
            )
            _dycore_state, carry = model.bootstrap_state()

            int_leaves = {
                tu.keystr(path): leaf.dtype
                for path, leaf in tu.tree_flatten_with_path(carry)[0]
                if hasattr(leaf, "dtype")
                and jnp.issubdtype(leaf.dtype, jnp.integer)
            }
        # A vacuous pass would hide the carry losing its index leaves.
        self.assertGreaterEqual(len(int_leaves), 3)
        # Every index/type-code leaf sits at the codebase's int32 width, so
        # the cross-step scan carry has one integer dtype whatever the
        # process-wide x64 setting.
        for name, dtype in int_leaves.items():
            self.assertEqual(dtype, jnp.int32, name)


if __name__ == "__main__":
    unittest.main()
