"""Tests for the idealized ECHAM-stack test helper."""

import unittest

from jcm.physics.echam.testing import idealized_echam_physics
from jcm.physics.radiation.band_config import RadiationBandConfig
from jcm.physics.radiation.grey_two_stream import GreyTwoStreamRadiation
from jcm.physics.radiation.radiation_types import RadiationParameters


def _radiation(physics):
    (rad,) = (t for t in physics.terms if t.category == "radiation")
    return rad


class TestIdealizedEchamPhysics(unittest.TestCase):
    """The helper is the ECHAM stack with an explicit grey radiation term."""

    def test_composes_grey_with_broadband_bands(self):
        physics = idealized_echam_physics(checkpoint_terms=False)
        self.assertIsInstance(_radiation(physics), GreyTwoStreamRadiation)
        self.assertEqual(physics.band_config, RadiationBandConfig.broadband())

    def test_takes_the_factory_radiation_defaults(self):
        # Same defaults as echam_physics() gives its own radiation term:
        # ECHAM-HAM's zinhomi = 0.7 with JAM, ECHAM6's 0.8 otherwise.
        def zinhomi(physics):
            p = _radiation(physics).params.get_value()
            return float(p.cloud_inhomogeneity_ice)

        self.assertAlmostEqual(
            zinhomi(idealized_echam_physics(checkpoint_terms=False)), 0.8,
            places=6)
        self.assertAlmostEqual(zinhomi(idealized_echam_physics(
            checkpoint_terms=False, aerosol_module="jam", cloud_scheme="2m",
            jam_microphysics="placeholder")), 0.7, places=6)

    def test_radiation_parameters_reach_the_grey_term(self):
        params = RadiationParameters.default(solar_constant=420.0)
        physics = idealized_echam_physics(radiation=params)
        held = _radiation(physics).params.get_value()
        self.assertAlmostEqual(float(held.solar_constant), 420.0, places=3)

    def test_other_kwargs_reach_the_factory(self):
        physics = idealized_echam_physics(cloud_scheme="2m")
        self.assertIn("lohmann_2m_microphysics",
                      [t.name for t in physics.terms])

    def test_radiation_scheme_cannot_be_passed(self):
        with self.assertRaises(TypeError):
            idealized_echam_physics(radiation_scheme="rrtmgp")


if __name__ == "__main__":
    unittest.main()
