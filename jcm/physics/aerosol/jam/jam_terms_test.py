"""Phase 6 tests: harness factory + echam_physics(aerosol_module="jam")."""

import unittest

import pytest


class JamFactoryTest(unittest.TestCase):
    def test_term_order_and_categories(self):
        from jcm.physics.aerosol.jam import jam_aerosol_physics

        terms = jam_aerosol_physics()
        cats = [t.category for t in terms]
        names = [t.name for t in terms]
        # The ``aerosol`` carry-slot seeder runs first (#640): radiation and
        # the 2M microphysics read that slot, so it must be well-formed before
        # any consumer. Default storage is CARRY (#602 item 3 A/B): the store
        # term owns the cloud-borne carry and runs next. Then the emi_*
        # accumulator reset (which must precede every emitter — the diagnostics
        # dict is threaded back from the previous step), the natural-emission
        # schemes, then the core + processes.
        self.assertEqual(names[0], "aerosol_carry_seeder")
        self.assertEqual(cats[0], "aerosol")
        self.assertEqual(names[1], "jam_cloud_borne_store")
        self.assertEqual(names[2], "reset_emission_fluxes")
        self.assertEqual(
            names[3:6],
            [
                "jam_seasalt_emissions",
                "jam_dms_emissions",
                "jam_dust_emissions",
            ],
        )
        self.assertEqual(
            cats[6:],
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
        self.assertTrue(all(c == "aerosol_emissions" for c in cats[2:6]))

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

        # The interstitial set is declared; cloud-borne names never are
        # (the phase lives in the physics carry, #602).
        names = set()
        for t in jam_aerosol_physics():
            names |= {s.name for s in t.required_tracers()}
        self.assertTrue(
            {s.name for s in tracer_specs(MAM4_SPEC)}.issubset(names)
        )
        self.assertFalse(any(n.startswith(("mc_", "nc_")) for n in names))


class EchamPhysicsWiringTest(unittest.TestCase):
    """Constructing the composition runs _validate_ordering across the stack."""

    def test_jam_module_builds_with_2m(self):
        from jcm.physics.echam.echam_terms import echam_physics

        phys = echam_physics(aerosol_module="jam", cloud_scheme="2m")
        cats = [t.category for t in phys.terms]
        names = [t.name for t in phys.terms]
        # No MACv2-SP in the JAM path (#640): the ``aerosol`` slot is owned by
        # JAM's own ``AerosolCarrySeeder`` (category "aerosol"), not MACv2-SP.
        self.assertNotIn("macv2_sp_aerosol", names)
        self.assertIn("aerosol_carry_seeder", names)
        self.assertIn("aerosol", cats)            # the seeder provides "aerosol"
        self.assertIn("aerosol_activation", cats)  # JAM ARG present
        # ARG activation must precede the 2M cloud term that reads it.
        self.assertLess(
            cats.index("aerosol_activation"), cats.index("clouds")
        )
        # The seeder must precede the radiation term that reads ``aerosol``.
        self.assertLess(
            names.index("aerosol_carry_seeder"),
            next(i for i, t in enumerate(phys.terms)
                 if t.category == "radiation"),
        )

    def test_jam_module_rejects_1m(self):
        # JAM's scavenging/resuspension terms read the process-time ledger
        # only the 2M scheme publishes; the combination must fail loudly
        # at compose time rather than silently scavenge nothing.
        from jcm.physics.echam.echam_terms import echam_physics

        with self.assertRaisesRegex(ValueError, "cloud_scheme='2m'"):
            echam_physics(aerosol_module="jam", cloud_scheme="1m")

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


# ---------------------------------------------------------------------------
# 64-bit mode (#770)
# ---------------------------------------------------------------------------

def _trace_float32_physics_under_x64(**physics_kwargs):
    """Trace one ECHAM+JAM physics step, float32 physics with x64 on.

    pySES runs float32 physics under ``jax_enable_x64``, and ``mam4_jax``
    turns the flag on process-wide when it is imported. The parameters are
    then float64 while the state is float32, and a float64 value scattered
    into a float32 operand (``.at[...].set``) is a JAX ``FutureWarning`` today
    and an error in later releases, so every ``FutureWarning`` is raised as an
    error here. Dtype promotion is decided when the step is traced, so
    tracing it with ``jax.eval_shape`` reaches every such site without
    running the physics. The flag is set and restored explicitly, in a
    ``finally``, rather than through ``jax.enable_x64``: the adapter
    ``mam4_jax`` builds sets the process-wide flag itself, which a context
    manager would not undo.
    """
    import warnings

    import jax
    import jax.numpy as jnp
    import numpy as np

    from jcm.forcing import ForcingData
    from jcm.physics_interface import PhysicsState
    from jcm.terrain import TerrainData
    from jcm.utils import get_coords

    previous = bool(jax.config.read("jax_enable_x64"))
    try:
        jax.config.update("jax_enable_x64", True)
        from jcm.model import Model
        from jcm.physics.echam.echam_terms import echam_physics

        coords = get_coords(np.linspace(0, 1, 9), spectral_truncation=21)
        model = Model(coords=coords, time_step=30,
                      terrain=TerrainData.aquaplanet(coords),
                      physics=echam_physics(aerosol_module="jam",
                                            cloud_scheme="2m",
                                            **physics_kwargs))
        physics = model.physics
        nodal = coords.horizontal.nodal_shape
        shape_3d = (coords.nodal_shape[0],) + nodal
        state = PhysicsState.zeros(shape_3d).copy(
            temperature=jnp.full(shape_3d, 288.0),
            normalized_surface_pressure=jnp.ones(nodal),
            tracers={spec.name: jnp.zeros(shape_3d)
                     for spec in physics.required_tracers()},
        )
        forcing = ForcingData.zeros(nodal)
        for term in physics.terms:
            forcing = term.augment_probe_forcing(forcing)

        def cast(tree):
            return jax.tree.map(
                lambda x: x.astype(jnp.float32)
                if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating)
                else x, tree)

        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            jax.eval_shape(physics.compute_tendencies, cast(state),
                           cast(forcing), cast(model.terrain))
    finally:
        jax.config.update("jax_enable_x64", previous)


def test_jam_package_traces_under_x64_without_mixed_dtype_scatters():
    """The ECHAM+JAM package, placeholder aerosol core, in 64-bit mode."""
    _trace_float32_physics_under_x64()


@pytest.mark.requires_extra("mam4")
def test_mam4_jam_package_traces_under_x64_without_mixed_dtype_scatters():
    """The same with the MAM4-JAX aerosol core, whose import sets x64 itself."""
    import jax

    before = bool(jax.config.read("jax_enable_x64"))
    _trace_float32_physics_under_x64(jam_microphysics="mam4_jax")
    assert bool(jax.config.read("jax_enable_x64")) == before


if __name__ == "__main__":
    unittest.main()
