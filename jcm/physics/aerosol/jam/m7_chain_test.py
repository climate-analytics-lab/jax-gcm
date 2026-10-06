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

    def test_seeded_m7_dust_mass_gives_nonzero_contact_freezing_fractions(self):
        """jax-gcm#1017 task 2 part 1: the composed-physics WIRING, not just
        the bare :func:`ham_freezing_aerosol` call, produces non-zero contact
        freezing fractions from a real M7 aerosol state.

        The test just above only confirms *finite* on this model's cold
        start, because that start's own dust is genuinely ~0 (an aquaplanet
        has no land, so the Tegen source -- #802/#808 -- never fires in a
        6-step run): it cannot exercise the non-zero branch. Emission is
        validated end to end on its own path (``dust_integration_test.py``);
        what is NOT otherwise covered is that ``PhysicsState`` tracers for
        M7's insoluble accumulation/coarse dust (``ai``/``ci``) survive
        through the scan-based ``ComposablePhysics`` loop into
        :class:`IceNucleation`'s own harvesting of mass/number/wet-radius
        diagnostics and out through :func:`ham_freezing_aerosol` -- the
        exact chain ``m7_freezing_chain_reference_test.py`` verifies to
        1e-8/1e-4 (float64/float32) against compiled HAM, but by calling the
        two functions directly rather than through the live model.

        Seeds a uniform (every level, every column) dust mass/number mixing
        ratio directly into ``ai``/``ci`` at ``t=0`` -- bypassing emission
        entirely, since emission is covered elsewhere -- sized from the
        mode's own equilibrium geometry (:func:`particle_mean_mass`) so the
        seed is an internally-consistent aerosol population, not an
        arbitrary number. The absolute magnitude is not meant to be a
        realistic atmospheric dust burden; it only needs to be unambiguously
        non-zero after 6 steps of transport/microphysics.
        """
        from jcm.physics.aerosol.jam.emissions.distributors import (
            particle_mean_mass)
        from jcm.physics.aerosol.jam.tracer_layout import mass_name, number_name

        model = self._build()
        # Build the default (zeroed-aerosol) dycore-native state, then
        # convert it to the GRIDPOINT ``PhysicsState`` -- the dycore-native
        # tracer representation has its own spectral packing, and hand-built
        # ``to_modal`` arrays do not reproduce it (confirmed: doing so here
        # reshapes into the wrong size downstream). ``model.run`` accepts a
        # gridpoint ``PhysicsState`` directly and re-projects it through the
        # dycore's own, already-tested ``initial_state`` path.
        dycore_state = model.initial_state(physics_state=None, random_seed=0)
        nodal_state = model.dycore.to_physics_state(dycore_state)

        density = M7_SPEC.species_props("du").density
        mass_mixing_ratio = {"ai": 1.0e-9, "ci": 1.0e-9}  # kg/kg, synthetic

        tracers = dict(nodal_state.tracers)
        for short, mmr in mass_mixing_ratio.items():
            mean_mass = particle_mean_mass(M7_SPEC.mode(short), density)
            number_mixing_ratio = mmr / mean_mass  # 1/kg, consistent geometry
            tracers[mass_name("du", short)] = jnp.full_like(
                tracers[mass_name("du", short)], mmr)
            tracers[number_name(short)] = jnp.full_like(
                tracers[number_name(short)], number_mixing_ratio)
        nodal_state = nodal_state.copy(tracers=tracers)

        predictions = model.run(
            initial_state=nodal_state, save_interval=0.125, total_time=0.125)
        dyn = predictions.dynamics
        self.assertFalse(bool(jnp.any(jnp.isnan(dyn.temperature))))

        freezing = predictions.physics["freezing_aerosol"]
        for field in ("dust_insoluble_accumulation", "dust_insoluble_coarse"):
            arr = np.asarray(getattr(freezing, field))
            self.assertTrue(np.all(np.isfinite(arr)), field)
            self.assertGreater(float(np.max(arr)), 0.0, field)

    def test_seasalt_mass_appears_only_in_as_cs(self):
        from jcm.physics.aerosol.jam.tracer_layout import mass_name

        _, predictions = self._run()
        tracers = predictions.dynamics.tracers
        for short in ("as", "cs"):
            self.assertIn(mass_name("ss", short), tracers)
        for short in ("ns", "ks", "ki", "ai", "ci"):
            self.assertNotIn(mass_name("ss", short), tracers)

    def test_sector_emissions_on_all_sectors_and_subsets_land_in_ham_modes(self):
        """End-to-end smoke test of jax-gcm#1017 task 2 on the full model.

        Synthetic SO2/BC/OC emissions on all four super-sectors plus both
        subset channels (residential, energy) exercise every branch of
        ``AnthropogenicEmissions``'s HAM-sector-policy path inside a real
        dynamical-core integration. The EXACT per-sector mass-fraction/
        number-vs-zm2n arithmetic is checked term-level, in isolation, in
        ``emissions/anthropogenic_test.py``'s ``M7SectorEmissionTest`` (the
        right place to verify closed-form numbers, since after transport and
        deposition have acted the simple flux*fraction algebra no longer
        holds). This test instead confirms the chain runs finite end-to-end
        and that the classes HAM's policy actually targets (KI for BC, KI/KS
        for OC, KS/AS/CS for primary SO4) pick up finite, non-trivial mass. It does
        NOT assert on classes a species structurally CAN carry but HAM's
        primary emission never targets (e.g. BC in AS/CS, which M7 carries
        for aged/internally-mixed mass via HAM's own chemistry, not
        emission) — those are a near-zero (not exactly-zero) signal here
        since the placeholder core does no ageing, and the fast term-level
        test above already isolates exactly what the emission term itself
        produces.
        """
        from jcm.forcing import default_forcing
        from jcm.physics.aerosol.jam.tracer_layout import mass_name

        model = self._build(jam_anthropogenic=True)
        shape = model.coords.horizontal.nodal_shape
        channels = {}
        for sector, (so2, bc, oc) in {
            "surface_combustion": (2.0e-9, 1.0e-9, 3.0e-9),
            "elevated_industrial": (3.0e-9, 0.5e-9, 1.0e-9),
            "shipping": (1.0e-9, 0.2e-9, 0.3e-9),
            "biomass_burning": (0.5e-9, 2.0e-9, 4.0e-9),
        }.items():
            channels[f"emis_{sector}_so2"] = jnp.full(shape, so2)
            channels[f"emis_{sector}_bc"] = jnp.full(shape, bc)
            channels[f"emis_{sector}_oc"] = jnp.full(shape, oc)
        # Subsets: a fraction of their parent super-sector's own flux.
        for subset, parent, fraction in (
            ("residential", "surface_combustion", 0.3),
            ("energy", "elevated_industrial", 0.4),
        ):
            for sp in ("so2", "bc", "oc"):
                channels[f"emis_{subset}_{sp}"] = (
                    fraction * channels[f"emis_{parent}_{sp}"])
        forcing = default_forcing(model.coords.horizontal).copy(
            anthropogenic_emissions=channels)

        predictions = model.run(
            forcing=forcing, save_interval=0.125, total_time=0.125)
        dyn = predictions.dynamics
        self.assertFalse(bool(jnp.any(jnp.isnan(dyn.temperature))))

        tracers = dyn.tracers
        # BC's primary emission targets ONLY KI, in every HAM class
        # (fossil/energy_ships/biomass_like all give it mass_fraction=1.0
        # at KI — see ham_sectors.m7_sector_policy); it never splits to KS.
        arr = np.asarray(tracers[mass_name("bc", "ki")])
        self.assertTrue(np.all(np.isfinite(arr)))
        self.assertGreater(float(np.max(np.abs(arr))), 0.0)
        # OC targets KI in every class, and ALSO KS for biomass_like (the
        # biomass_burning sector, and the residential subset of
        # surface_combustion routed to biomass_like here).
        for short in ("ki", "ks"):
            arr = np.asarray(tracers[mass_name("oc", short)])
            self.assertTrue(np.all(np.isfinite(arr)), short)
            self.assertGreater(float(np.max(np.abs(arr))), 0.0, short)
        # Primary SO4 targets only KS/AS (fossil/biomass_like) and AS/CS
        # (energy_ships) — never the pure-nucleation NS.
        for short in ("ks", "as", "cs"):
            arr = np.asarray(tracers[mass_name("so4", short)])
            self.assertTrue(np.all(np.isfinite(arr)), short)
            self.assertGreater(float(np.max(np.abs(arr))), 0.0, short)

    def test_ham_variants_compose_and_run_finite(self):
        """Every HAM harness variant together on M7, through ``Model.run``.

        HAM activation over the 20-bin updraft PDF, the HAM Mie-table optics
        (jcm-built tables when ``HAM_INPUT_DIR`` is unset), Long sea salt and
        the full ``nwetdep = 3`` wet deposition: the integration path the
        ``echam-ham-m7`` preset runs, minus the extra-gated M7-JAX core.
        """
        model, predictions = self._run(
            jam_activation_scheme="ham_arg", jam_nactivpdf=1,
            jam_optics_backend="ham_lut", jam_seasalt_scheme="long",
            jam_wetdep_scheme="ham", jam_anthropogenic=True)
        dyn = predictions.dynamics
        self.assertTrue(bool(jnp.all(jnp.isfinite(dyn.temperature))))
        for key, arr in dyn.tracers.items():
            self.assertTrue(np.all(np.isfinite(np.asarray(arr))), key)
        names = {t.name for t in model.physics.terms}
        self.assertIn("jam_wet_deposition", names)
        wet = next(t for t in model.physics.terms
                   if t.name == "jam_wet_deposition")
        self.assertEqual(wet.scheme, "ham")


if __name__ == "__main__":
    unittest.main()
