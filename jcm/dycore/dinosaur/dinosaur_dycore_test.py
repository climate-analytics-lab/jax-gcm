"""Tests for the dinosaur dycore backend's option handling."""

import unittest

import numpy as np


class SemiLagrangianAvailabilityTest(unittest.TestCase):
    """``advection='semi_lagrangian'`` needs an SL-capable dinosaur."""

    def test_raises_clear_error_when_unavailable(self):
        from unittest import mock

        from jcm.dycore.dinosaur import dycore as dycore_mod
        from jcm.terrain import TerrainData
        from jcm.utils import get_coords

        coords = get_coords(np.linspace(0, 1, 4), spectral_truncation=21)
        with mock.patch.object(dycore_mod, "semi_lagrangian_available",
                               return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                dycore_mod.DinosaurDycore(
                    coords=coords, terrain=TerrainData.aquaplanet(coords),
                    dt_seconds=1800.0, advection="semi_lagrangian",
                )
        self.assertIn("semi-Lagrangian", str(ctx.exception))

    def test_eulerian_default_builds_without_sl(self):
        from unittest import mock

        from jcm.dycore.dinosaur import dycore as dycore_mod
        from jcm.terrain import TerrainData
        from jcm.utils import get_coords

        coords = get_coords(np.linspace(0, 1, 4), spectral_truncation=21)
        with mock.patch.object(dycore_mod, "semi_lagrangian_available",
                               return_value=False):
            dyc = dycore_mod.DinosaurDycore(
                coords=coords, terrain=TerrainData.aquaplanet(coords),
                dt_seconds=1800.0,
            )
        self.assertEqual(dyc.advection, "eulerian")

    def test_availability_probe_checks_both_modules(self):
        # The operator classes live on primitive_equations, the integrator
        # on time_integration; probing one module reports unavailable on a
        # capable build and blocks every SL run.
        from unittest import mock

        from dinosaur import primitive_equations, time_integration

        from jcm.dycore.dinosaur.dycore import semi_lagrangian_available

        with mock.patch.object(primitive_equations,
                               "SemiLagrangianPrimitiveEquations",
                               object, create=True), \
             mock.patch.object(primitive_equations,
                               "SemiLagrangianPrimitiveEquationsHybrid",
                               object, create=True), \
             mock.patch.object(time_integration,
                               "semi_lagrangian_crank_nicolson_rk2",
                               object, create=True):
            self.assertTrue(semi_lagrangian_available())
