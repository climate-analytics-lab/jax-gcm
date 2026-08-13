"""Phase 6 tests: harness factory + echam_physics(aerosol_module="jam")."""

import unittest


class JamFactoryTest(unittest.TestCase):
    def test_term_order_and_categories(self):
        from jcm.physics.aerosol.jam import jam_aerosol_physics

        terms = jam_aerosol_physics()
        cats = [t.category for t in terms]
        names = [t.name for t in terms]
        # Default storage is CARRY (#602 item 3 A/B): the store term owns
        # the carry slot and runs first. Then the emi_* accumulator reset
        # (which must precede every emitter — the diagnostics dict is
        # threaded back from the previous step), the natural-emission
        # schemes, then the core + processes.
        self.assertEqual(names[0], "jam_cloud_borne_store")
        self.assertEqual(names[1], "reset_emission_fluxes")
        self.assertEqual(
            names[2:5],
            [
                "jam_seasalt_emissions",
                "jam_dms_emissions",
                "jam_dust_emissions",
            ],
        )
        self.assertEqual(
            cats[5:],
            [
                # Physics-side vertical transport (#602 item 2): turbulent
                # mixing of every JAM tracer, then convective mass-flux
                # transport of the interstitial + gas set.
                "tracer_transport",
                "tracer_transport",
                "aerosol_oxidants",
                "aerosol_gas_chemistry",
                "aerosol_microphysics",
                "aerosol_optics",
                "aerosol_activation",
                "aerosol_ice_nucleation",
                "aerosol_sedimentation",
                "aerosol_drydep",
                "aerosol_cloud_borne",
                "aerosol_aqueous_chemistry",
                "aerosol_wetdep",
            ],
        )
        # The reset shares the emitters' category — it is part of that
        # block, not a separate stage.
        self.assertTrue(all(c == "aerosol_emissions" for c in cats[1:5]))

    def test_activation_precedes_deposition(self):
        # wetdep requires activated_fraction, so ARG must come first.
        from jcm.physics.aerosol.jam import jam_aerosol_physics

        cats = [t.category for t in jam_aerosol_physics()]
        self.assertLess(
            cats.index("aerosol_activation"), cats.index("aerosol_wetdep")
        )

    def test_ghosh_variant_threads_through(self):
        from jcm.physics.aerosol.jam import jam_aerosol_physics

        terms = jam_aerosol_physics(arg_variant="ghosh2025")
        arg = next(t for t in terms if t.category == "aerosol_activation")
        self.assertEqual(arg._variant, "ghosh2025")

    def test_unknown_microphysics_raises(self):
        from jcm.physics.aerosol.jam import jam_aerosol_physics

        with self.assertRaises(ValueError):
            jam_aerosol_physics(microphysics="m7")

    def test_harness_declares_aerosol_tracers(self):
        from jcm.physics.aerosol.jam import MAM4_SPEC, jam_aerosol_physics, tracer_specs

        # Explicit tracers storage declares the full mirror set...
        names = set()
        for t in jam_aerosol_physics(cloud_borne_storage="tracers"):
            names |= {s.name for s in t.required_tracers()}
        self.assertTrue(
            {s.name for s in tracer_specs(MAM4_SPEC)}.issubset(names)
        )
        # ...while the carry default declares interstitial only.
        names = set()
        for t in jam_aerosol_physics():
            names |= {s.name for s in t.required_tracers()}
        self.assertFalse(any(n.startswith(("mc_", "nc_")) for n in names))


class EchamPhysicsWiringTest(unittest.TestCase):
    """Constructing the composition runs _validate_ordering across the stack."""

    def test_jam_module_builds_with_2m(self):
        from jcm.physics.echam.echam_terms import echam_physics

        phys = echam_physics(aerosol_module="jam", cloud_scheme="2m")
        cats = [t.category for t in phys.terms]
        # MACv2-SP retained for optics, JAM aerosol terms appended.
        self.assertIn("aerosol", cats)            # MACv2-SP provides "aerosol"
        self.assertIn("aerosol_activation", cats)  # JAM ARG present
        # ARG activation must precede the 2M cloud term that reads it.
        self.assertLess(
            cats.index("aerosol_activation"), cats.index("clouds")
        )

    def test_jam_module_builds_with_1m(self):
        from jcm.physics.echam.echam_terms import echam_physics

        phys = echam_physics(aerosol_module="jam", cloud_scheme="1m")
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

    def test_jam_ghosh_variant_builds(self):
        from jcm.physics.echam.echam_terms import echam_physics

        phys = echam_physics(
            aerosol_module="jam", cloud_scheme="2m", jam_arg_variant="ghosh2025",
        )
        arg = next(t for t in phys.terms if t.category == "aerosol_activation")
        self.assertEqual(arg._variant, "ghosh2025")


if __name__ == "__main__":
    unittest.main()
