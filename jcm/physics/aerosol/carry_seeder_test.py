"""Tests for ``AerosolCarrySeeder`` (#640)."""

import unittest

import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.carry_seeder import AerosolCarrySeeder
from jcm.physics.radiation.band_config import RadiationBandConfig
from jcm.physics_interface import PhysicsState


class AerosolCarrySeederTest(unittest.TestCase):
    def test_provides_and_withholds_the_aerosol_slot(self):
        term = AerosolCarrySeeder()
        self.assertEqual(term.provides, ("aerosol",))
        self.assertIn("aerosol", term.carry_slots)
        # The whole internal struct is withheld from output (plumbing).
        withheld = term.withheld_output_keys()
        self.assertIn("aerosol.aod_total", withheld)
        self.assertIn("aerosol.Nccn", withheld)
        self.assertIn("aerosol.aod_profile", withheld)

    def test_call_resets_to_zero_base_at_band_shape(self):
        term = AerosolCarrySeeder()
        band = RadiationBandConfig(
            sw_band_centers_nm=(400.0, 550.0, 900.0),
            lw_band_centers_nm=(8000.0, 12000.0),
        )
        term.cache_band_config(band)
        nlev, ncols = 4, 3
        state = PhysicsState.zeros((nlev, ncols)).copy(
            temperature=jnp.full((nlev, ncols), 285.0),
        )
        # A stale non-zero slot in the carry must be reset to zeros.
        from jcm.physics.aerosol.aerosol_types import AerosolData
        stale = AerosolData.zeros((ncols,), nlev, n_bnd_sw=3, n_bnd_lw=2).copy(
            aod_total=jnp.ones(ncols),
        )
        tend, out = term(
            state, {"aerosol": stale, "_band_config": band}, None, None,
        )
        a = out["aerosol"]
        self.assertEqual(a.aod_sw_per_band.shape, (3, nlev, ncols))
        self.assertEqual(a.aod_lw_per_band.shape, (2, nlev, ncols))
        self.assertEqual(float(jnp.sum(a.aod_total)), 0.0)
        self.assertEqual(float(jnp.max(jnp.abs(a.aod_sw_per_band))), 0.0)
        # Zero atmospheric tendency.
        self.assertEqual(float(jnp.max(jnp.abs(tend.temperature))), 0.0)

    def test_call_falls_back_to_broadband_without_band_config(self):
        term = AerosolCarrySeeder()
        nlev, ncols = 2, 2
        state = PhysicsState.zeros((nlev, ncols)).copy(
            temperature=jnp.full((nlev, ncols), 285.0),
        )
        _, out = term(state, {}, None, None)
        a = out["aerosol"]
        bb = RadiationBandConfig.broadband()
        self.assertEqual(a.aod_sw_per_band.shape[0],
                         len(bb.sw_band_centers_nm))

    def test_initial_carry_state_zero_filled(self):
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.utils import get_coords

        term = AerosolCarrySeeder()
        term.cache_band_config(RadiationBandConfig(
            sw_band_centers_nm=(550.0,), lw_band_centers_nm=(10000.0,),
        ))
        coords = get_coords(get_echam_levels(47), spectral_truncation=21)
        slot = term.initial_carry_state(coords)["aerosol"]
        self.assertTrue(bool(np.all(np.asarray(slot.aod_sw_per_band) == 0.0)))


if __name__ == "__main__":
    unittest.main()
