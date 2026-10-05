"""M7 population chain test (jax-gcm#1017): the JAM harness runs end-to-end.

Companion to ``jam_integration_test.py``'s MAM4 coverage. The fast test below
composes the JAM term list on the M7 population and checks every term
built; the slow class runs a T21/20-layer aquaplanet with
``jam_microphysics="m7_placeholder"`` and ``jam_cloud_borne=False`` (M7 has
no explicit cloud-borne phase — HAM scavenges interstitial aerosol by its
activated fraction) for a few steps and checks the model stays finite, the
transported tracer set is exactly M7's own interstitial tracers plus the
gas precursors, HAM's mixed-phase freezing inputs are published and
finite, and sea salt/dust emission lands in the modes HAM itself uses.
"""

import unittest

import jax.numpy as jnp
import numpy as np
import pytest

from jcm.physics.aerosol.jam.emissions.dust import DustEmissions
from jcm.physics.aerosol.jam.emissions.seasalt import SeaSaltEmissions
from jcm.physics.aerosol.jam.jam_terms import jam_aerosol_physics
from jcm.physics.aerosol.jam.microphysics.m7_data import M7_SPEC
from jcm.physics.physics_term import PhysicsTerm


class M7TermListTest(unittest.TestCase):
    """Fast: ``jam_aerosol_physics`` composes a full term list on M7_SPEC."""

    def test_every_term_built_on_the_m7_population(self):
        terms = jam_aerosol_physics(microphysics="m7_placeholder", cloud_borne=False)
        self.assertGreater(len(terms), 0)
        for term in terms:
            self.assertIsInstance(term, PhysicsTerm)
            self.assertIsNotNone(term)

        # No cloud-borne-cycling terms: M7's own population has no explicit
        # cloud-borne phase, so neither the carry store nor the exchange
        # term should be composed.
        names = {type(t).__name__ for t in terms}
        self.assertNotIn("CloudBorneCarryStore", names)
        self.assertNotIn("CloudBorneExchange", names)

    def test_dust_emission_targets_insoluble_accum_coarse(self):
        dust = next(
            t for t in jam_aerosol_physics(
                microphysics="m7_placeholder", cloud_borne=False)
            if isinstance(t, DustEmissions)
        )
        self.assertEqual(dust._accum.short, "ai")
        self.assertEqual(dust._coarse.short, "ci")
        self.assertIsNotNone(dust._fixed_number_factor)

    def test_seasalt_emission_targets_soluble_accum_coarse(self):
        seasalt = next(
            t for t in jam_aerosol_physics(
                microphysics="m7_placeholder", cloud_borne=False)
            if isinstance(t, SeaSaltEmissions)
        )
        self.assertEqual(
            {m.short for m in seasalt._spec.classes_for("ss")}, {"as", "cs"})


@pytest.mark.slow
class M7ChainTest(unittest.TestCase):
    def _build(self, **physics_kwargs):
        from jcm.model import Model
        from jcm.physics.echam.testing import idealized_echam_physics
        from jcm.terrain import TerrainData
        from jcm.utils import get_coords

        sigma_boundaries = np.linspace(0, 1, 21)  # 20 layers
        coords = get_coords(sigma_boundaries, spectral_truncation=21)
        terrain = TerrainData.aquaplanet(coords)
        return Model(
            coords=coords,
            time_step=30,
            terrain=terrain,
            physics=idealized_echam_physics(
                aerosol_module="jam", cloud_scheme="2m",
                jam_microphysics="m7_placeholder", jam_cloud_borne=False,
                **physics_kwargs,
            ),
        )

    def _run(self, **physics_kwargs):
        model = self._build(**physics_kwargs)
        # Six steps, matching jam_integration_test.py's MAM4 companion: a
        # cold start with no seeded aerosol needs that long for the
        # emission -> transport -> core -> activation chain to produce
        # anything.
        return model, model.run(save_interval=0.125, total_time=0.125)

    def test_runs_finite_and_tracer_set_is_m7s_own(self):
        from jcm.physics.aerosol.jam.gas_species import SULFUR_GASES
        from jcm.physics.aerosol.jam.tracer_layout import (
            gas_name,
            mass_name,
            number_name,
        )

        _, predictions = self._run()
        dyn = predictions.dynamics

        self.assertFalse(bool(jnp.any(jnp.isnan(dyn.temperature))))
        self.assertFalse(bool(jnp.any(jnp.isnan(dyn.specific_humidity))))
        self.assertTrue(bool(jnp.all(dyn.temperature > 150.0)))
        self.assertTrue(bool(jnp.all(dyn.temperature < 360.0)))

        aerosol_expected = {
            mass_name(sp, mode.short)
            for mode in M7_SPEC.modes for sp in mode.species
        } | {number_name(mode.short) for mode in M7_SPEC.modes}
        # 18 mass tracers (so4 x4, bc x4, oc x4, ss x2, du x4) + 7 number.
        self.assertEqual(len(aerosol_expected), 25)
        expected = aerosol_expected | {gas_name(g) for g in SULFUR_GASES}

        present = {
            k for k in dyn.tracers
            if k.startswith(("m_", "mc_", "n_", "nc_", "g_"))
        }
        self.assertEqual(present, expected)
        for key in expected:
            arr = np.asarray(dyn.tracers[key])
            self.assertTrue(np.all(np.isfinite(arr)), key)

    def test_freezing_aerosol_published_and_dust_contact_inputs_finite(self):
        _, predictions = self._run()
        physics = predictions.physics
        self.assertIn("freezing_aerosol", physics)
        freezing = physics["freezing_aerosol"]
        for field in ("dust_insoluble_accumulation", "dust_insoluble_coarse"):
            arr = np.asarray(getattr(freezing, field))
            self.assertTrue(np.all(np.isfinite(arr)), field)

    def test_seasalt_mass_appears_only_in_as_cs(self):
        from jcm.physics.aerosol.jam.tracer_layout import mass_name

        _, predictions = self._run()
        tracers = predictions.dynamics.tracers
        for short in ("as", "cs"):
            self.assertIn(mass_name("ss", short), tracers)
        for short in ("ns", "ks", "ki", "ai", "ci"):
            self.assertNotIn(mass_name("ss", short), tracers)


if __name__ == "__main__":
    unittest.main()
