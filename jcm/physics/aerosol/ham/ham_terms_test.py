"""Phase 6 tests: harness factory + echam_physics(aerosol_module="ham")."""

import unittest


class HamFactoryTest(unittest.TestCase):
    def test_term_order_and_categories(self):
        from jcm.physics.aerosol.ham import ham_aerosol_physics

        terms = ham_aerosol_physics()
        cats = [t.category for t in terms]
        self.assertEqual(
            cats,
            [
                "aerosol_emissions",
                "aerosol_microphysics",
                "aerosol_activation",
                "aerosol_sedimentation",
                "aerosol_drydep",
                "aerosol_wetdep",
            ],
        )

    def test_activation_precedes_deposition(self):
        # wetdep requires activated_fraction, so ARG must come first.
        from jcm.physics.aerosol.ham import ham_aerosol_physics

        cats = [t.category for t in ham_aerosol_physics()]
        self.assertLess(
            cats.index("aerosol_activation"), cats.index("aerosol_wetdep")
        )

    def test_ghosh_variant_threads_through(self):
        from jcm.physics.aerosol.ham import ham_aerosol_physics

        terms = ham_aerosol_physics(arg_variant="ghosh2025")
        arg = next(t for t in terms if t.category == "aerosol_activation")
        self.assertEqual(arg._variant, "ghosh2025")

    def test_unknown_microphysics_raises(self):
        from jcm.physics.aerosol.ham import ham_aerosol_physics

        with self.assertRaises(ValueError):
            ham_aerosol_physics(microphysics="m7")

    def test_harness_declares_aerosol_tracers(self):
        from jcm.physics.aerosol.ham import MAM4_SPEC, ham_aerosol_physics, tracer_specs

        names = set()
        for t in ham_aerosol_physics():
            names |= {s.name for s in t.required_tracers()}
        self.assertTrue(
            {s.name for s in tracer_specs(MAM4_SPEC)}.issubset(names)
        )


class EchamPhysicsWiringTest(unittest.TestCase):
    """Constructing the composition runs _validate_ordering across the stack."""

    def test_ham_module_builds_with_2m(self):
        from jcm.physics.echam.echam_terms import echam_physics

        phys = echam_physics(aerosol_module="ham", cloud_scheme="2m")
        cats = [t.category for t in phys.terms]
        # MACv2-SP retained for optics, HAM aerosol terms appended.
        self.assertIn("aerosol", cats)            # MACv2-SP provides "aerosol"
        self.assertIn("aerosol_activation", cats)  # HAM ARG present
        # ARG activation must precede the 2M cloud term that reads it.
        self.assertLess(
            cats.index("aerosol_activation"), cats.index("clouds")
        )

    def test_ham_module_builds_with_1m(self):
        from jcm.physics.echam.echam_terms import echam_physics

        phys = echam_physics(aerosol_module="ham", cloud_scheme="1m")
        self.assertIn("aerosol_microphysics", [t.category for t in phys.terms])

    def test_default_is_macv2sp_only(self):
        from jcm.physics.echam.echam_terms import echam_physics

        phys = echam_physics()
        cats = [t.category for t in phys.terms]
        self.assertNotIn("aerosol_microphysics", cats)

    def test_unknown_module_raises(self):
        from jcm.physics.echam.echam_terms import echam_physics

        with self.assertRaises(ValueError):
            echam_physics(aerosol_module="bogus")

    def test_ham_ghosh_variant_builds(self):
        from jcm.physics.echam.echam_terms import echam_physics

        phys = echam_physics(
            aerosol_module="ham", cloud_scheme="2m", ham_arg_variant="ghosh2025",
        )
        arg = next(t for t in phys.terms if t.category == "aerosol_activation")
        self.assertEqual(arg._variant, "ghosh2025")


if __name__ == "__main__":
    unittest.main()
