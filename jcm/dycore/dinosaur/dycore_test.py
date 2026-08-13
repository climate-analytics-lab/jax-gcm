"""Tests for the dinosaur dycore backend's transport contract."""

import unittest


class SemiLagrangianRequiredTest(unittest.TestCase):
    """The dinosaur backend is semi-Lagrangian only — no Eulerian fallback.

    The Eulerian spectral transport was removed, not merely deselected: it
    rang negative on sharp emission sources and NaN'd the aerosol
    microphysics (#521), and while it remained the silent default whole
    investigations were run on it by accident.
    """

    def test_no_advection_knob_is_exposed(self):
        import inspect

        from jcm.dycore.dinosaur.dycore import DinosaurDycore

        params = inspect.signature(DinosaurDycore.__init__).parameters
        self.assertNotIn(
            "advection", params,
            "an advection selector is back; there is no supported Eulerian "
            "configuration, so it must not be selectable",
        )

    def test_missing_sl_core_fails_with_an_actionable_message(self):
        from unittest import mock

        from jcm.dycore.dinosaur import dycore as dycore_mod

        with mock.patch.object(dycore_mod, "semi_lagrangian_available",
                               return_value=False):
            with self.assertRaises(RuntimeError) as ctx:
                dycore_mod._require_semi_lagrangian()
        msg = str(ctx.exception)
        # The message has to say what to install, not just what is wrong.
        self.assertIn("semi-Lagrangian", msg)
        self.assertIn("pip install", msg)

    def test_available_probe_matches_the_installed_dinosaur(self):
        from dinosaur import primitive_equations

        from jcm.dycore.dinosaur.dycore import semi_lagrangian_available

        expected = all(
            hasattr(primitive_equations, n)
            for n in ("SemiLagrangianPrimitiveEquations",
                      "SemiLagrangianPrimitiveEquationsHybrid")
        )
        self.assertEqual(semi_lagrangian_available(), expected)
