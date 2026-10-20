"""The in-tree modal core's driver: scaffold, seam and process order.

AER-01 established that the core with no active process is identical to
:class:`PlaceholderMicrophysics`. AER-03 made the first process real, so
that criterion is now asserted on the scaffold alone (``processes=()``)
and, for the full core, on a population with nothing to exchange — no gas
and no semi-volatile aerosol. With SOA present and no SOAG, the full core
correctly *evaporates* the SOA, so it no longer matches a placeholder that
cannot.

The identity test is DELIBERATELY temporary and says so: each card must
delete its process from the list. A process that is filled in without its
test changing is a process nobody checked.
"""

import dataclasses
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.gas_species import SULFUR_GASES
from jcm.physics.aerosol.jam.microphysics import moments
from jcm.physics.aerosol.jam.microphysics.koehler import (
    equilibrium_modal_state,
    saturation_ratio,
)
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.microphysics.m7_data import M7_SPEC
from jcm.physics.aerosol.jam.microphysics.modal import (
    PROCESS_ORDER,
    JamModalMicrophysics,
    MicrophysicsEnvironment,
)
from jcm.physics.aerosol.jam.microphysics.placeholder import (
    PlaceholderMicrophysics,
)
from jcm.physics.aerosol.jam.tracer_layout import (
    gas_name,
    mass_name,
    number_name,
    tracer_specs,
)
from jcm.physics_interface import PhysicsState

NLEV, NCOLS = 4, 3
SHAPE = (NLEV, NCOLS)
TEMPERATURE, HUMIDITY, PRESSURE, DT = 285.0, 0.005, 9.0e4, 1800.0


def _population(nlev=NLEV, ncols=NCOLS, seed=0, *, soa=True):
    """Build a non-uniform population, so an identity test cannot pass trivially.

    Uniform fields would let a process that (say) overwrote a mode with its
    column mean still compare equal. ``soa=False`` zeroes the semi-volatile
    species, which is the only way a population can be inert under the
    full core when no gas is present.
    """
    rng = np.random.default_rng(seed)
    masses = {
        mass_name(sp, mode.short): jnp.asarray(
            rng.uniform(1.0e-12, 1.0e-9, (nlev, ncols)))
        for mode in MAM4_SPEC.modes for sp in mode.species
    }
    if not soa:
        for mode in MAM4_SPEC.modes:
            if mass_name("soa", mode.short) in masses:
                masses[mass_name("soa", mode.short)] = jnp.zeros((nlev, ncols))
    numbers = {
        number_name(mode.short): jnp.asarray(
            rng.uniform(1.0e6, 1.0e9, (nlev, ncols)))
        for mode in MAM4_SPEC.modes
    }
    return masses, numbers


def _state_and_diagnostics(tracers, nlev=NLEV, ncols=NCOLS):
    state = PhysicsState.zeros((nlev, ncols)).copy(
        temperature=jnp.full((nlev, ncols), TEMPERATURE),
        specific_humidity=jnp.full((nlev, ncols), HUMIDITY),
        tracers=tracers,
    )
    diagnostics = {
        "pressure_full": jnp.full((nlev, ncols), PRESSURE),
        "_dt_seconds": DT,
    }
    return state, diagnostics


def _environment(masses, numbers, *, h2so4_production=0.0):
    """Build an environment matching what the driver builds from the fixture."""
    temperature = jnp.full(SHAPE, TEMPERATURE)
    pressure = jnp.full(SHAPE, PRESSURE)
    saturation = saturation_ratio(temperature, jnp.full(SHAPE, HUMIDITY),
                                  pressure)
    return MicrophysicsEnvironment(
        temperature=temperature, pressure=pressure, saturation=saturation,
        air_density=pressure / (287.04 * temperature),
        state_start=equilibrium_modal_state(masses, numbers, MAM4_SPEC,
                                            saturation),
        h2so4_production=jnp.full(SHAPE, h2so4_production),
        height_above_surface=jnp.full(SHAPE, 500.0),
        pbl_height=jnp.zeros(SHAPE[1:]),
        dt=DT,
        dry_volume_start={mode.short: moments.mode_volume(masses, mode, MAM4_SPEC)
                          for mode in MAM4_SPEC.modes},
    )


class ScaffoldEquivalenceTest(unittest.TestCase):
    """What survives of the AER-01 criterion once a process is real."""

    def _assert_jam_state_equal(self, modal, place):
        for field in ("r_dry", "r_wet", "rho", "kappa", "mass", "number"):
            np.testing.assert_array_equal(
                np.asarray(getattr(modal, field)),
                np.asarray(getattr(place, field)),
                err_msg=f"_jam_state.{field} diverged from the placeholder",
            )

    def test_scaffold_alone_is_bitwise_equal_to_the_placeholder(self):
        masses, numbers = _population()
        state, diagnostics = _state_and_diagnostics({**masses, **numbers})
        _, modal = JamModalMicrophysics(processes=())(
            state, diagnostics, None, None)
        _, place = PlaceholderMicrophysics()(state, diagnostics, None, None)
        self._assert_jam_state_equal(modal["_jam_state"], place["_jam_state"])

    def test_full_core_matches_the_placeholder_on_a_clean_atmosphere(self):
        """Empty modes and no gas: nothing to condense, form or collide.

        Through AER-04 a POPULATED population with no gas and no SOA was
        inert too; coagulation (AER-05) acts on any two particles, so the
        criterion that survives is the clean-atmosphere one.
        """
        masses, numbers = _population(soa=False)
        masses = {k: jnp.zeros_like(v) for k, v in masses.items()}
        numbers = {k: jnp.zeros_like(v) for k, v in numbers.items()}
        state, diagnostics = _state_and_diagnostics({**masses, **numbers})
        tendency, modal = JamModalMicrophysics()(state, diagnostics, None, None)
        _, place = PlaceholderMicrophysics()(state, diagnostics, None, None)
        self._assert_jam_state_equal(modal["_jam_state"], place["_jam_state"])
        for key, value in tendency.tracers.items():
            self.assertTrue(bool(jnp.all(value == 0.0)),
                            msg=f"{key} tendency is nonzero on a clean atmosphere")

    def test_populated_modes_coagulate_even_with_nothing_to_exchange(self):
        """The reason the populated no-gas criterion no longer holds, pinned."""
        masses, numbers = _population(soa=False)
        state, diagnostics = _state_and_diagnostics({**masses, **numbers})
        tendency, _ = JamModalMicrophysics()(state, diagnostics, None, None)
        # Aitken and primary carbon only lose; accumulation also receives
        # the aged carbon (AER-06), so the invariant there is the total.
        for short in ("ait", "pcm"):
            self.assertTrue(bool(jnp.all(tendency.tracers[number_name(short)] < 0.0)))
        total = sum(tendency.tracers[number_name(s)] for s in ("ait", "acc", "pcm"))
        self.assertTrue(bool(jnp.all(total < 0.0)))
        self.assertTrue(bool(jnp.all(tendency.tracers[number_name("cor")] == 0.0)))

    def test_full_core_evaporates_soa_when_soag_is_absent(self):
        """The reason the old bitwise criterion no longer holds, pinned."""
        masses, numbers = _population(soa=True)
        state, diagnostics = _state_and_diagnostics({**masses, **numbers})
        tendency, modal = JamModalMicrophysics()(state, diagnostics, None, None)
        _, place = PlaceholderMicrophysics()(state, diagnostics, None, None)
        # Net evaporation into the empty gas. Individual modes may gain SOA
        # from others (a mode with more oxygenated POA has a lower
        # saturation), so the invariant is the TOTAL, not each mode.
        soa_keys = [k for k in tendency.tracers if k.startswith("m_soa_")]
        self.assertTrue(soa_keys)
        total = sum(tendency.tracers[k] for k in soa_keys)
        self.assertTrue(bool(jnp.all(total <= 0.0)))
        self.assertTrue(bool(jnp.any(total < 0.0)))
        self.assertTrue(bool(jnp.all(tendency.tracers[gas_name("soag")] >= 0.0)))
        # And so the population, hence the state, no longer matches a core
        # that cannot evaporate anything.
        self.assertFalse(np.array_equal(np.asarray(modal["_jam_state"].r_dry),
                                        np.asarray(place["_jam_state"].r_dry)))


class TendencySeamTest(unittest.TestCase):

    def test_key_set_is_static_and_covers_population_and_gases(self):
        masses, numbers = _population()
        expected = set(masses) | set(numbers) | {gas_name(g) for g in SULFUR_GASES}
        # With every tracer present ...
        state, diagnostics = _state_and_diagnostics({**masses, **numbers})
        tendency, _ = JamModalMicrophysics()(state, diagnostics, None, None)
        self.assertEqual(set(tendency.tracers), expected)
        # ... and with none at all (the structural probe): same keys.
        probe, _ = JamModalMicrophysics()(
            PhysicsState.zeros(SHAPE).copy(
                temperature=jnp.full(SHAPE, TEMPERATURE),
                specific_humidity=jnp.full(SHAPE, HUMIDITY)),
            diagnostics, None, None)
        self.assertEqual(set(probe.tracers), expected)

    def test_tendency_equals_population_change_over_dt(self):
        masses, numbers = _population()
        gases = {gas_name("h2so4"): jnp.full(SHAPE, 1.0e-12),
                 gas_name("soag"): jnp.full(SHAPE, 1.0e-10)}
        tracers = {**masses, **numbers, **gases}
        state, diagnostics = _state_and_diagnostics(tracers)
        term = JamModalMicrophysics()
        tendency, _ = term(state, diagnostics, None, None)
        # Reproduce the driver's fold by hand.
        before = term._gather(state, diagnostics)
        env = _environment(masses, numbers)
        after = before
        for name, process in term.processes:
            after = process(after, env, MAM4_SPEC, term._process_params().get(name))
        for group in ("masses", "numbers", "gases"):
            for key in getattr(before, group):
                expected = (getattr(after, group)[key]
                            - getattr(before, group)[key]) / DT
                np.testing.assert_allclose(
                    np.asarray(tendency.tracers[key]), np.asarray(expected),
                    rtol=1e-6, atol=0, err_msg=key)


class ProcessScaffoldTest(unittest.TestCase):

    def test_order_matches_mam4_amicphys(self):
        self.assertEqual(
            [name for name, _ in PROCESS_ORDER],
            ["gas_aerosol_exchange", "rename", "nucleation",
             "coagulation", "carbonaceous_ageing"],
        )

    def test_process_sequence_can_be_overridden(self):
        masses, numbers = _population()
        state, diagnostics = _state_and_diagnostics({**masses, **numbers})
        term = JamModalMicrophysics(processes=PROCESS_ORDER[1:2])
        self.assertEqual([n for n, _ in term.processes], ["rename"])
        _, diag = term(state, diagnostics, None, None)
        self.assertIn("_jam_state", diag)


class HarnessContractTest(unittest.TestCase):

    def test_required_tracers_match_the_population(self):
        self.assertEqual(
            {s.name for s in JamModalMicrophysics().required_tracers()},
            {s.name for s in tracer_specs(MAM4_SPEC)},
        )

    def test_missing_dt_falls_back_rather_than_raising(self):
        masses, numbers = _population()
        state, diagnostics = _state_and_diagnostics({**masses, **numbers})
        del diagnostics["_dt_seconds"]
        _, diag = JamModalMicrophysics()(state, diagnostics, None, None)
        self.assertIn("_jam_state", diag)

    def test_condensation_parameters_are_gradient_visible_leaves(self):
        term = JamModalMicrophysics()
        params = term.condensation.get_value()
        leaves = jax.tree_util.tree_leaves(params)
        # max_substeps is static aux and must NOT appear among the leaves.
        self.assertEqual(len(leaves), 8)
        self.assertEqual(params.max_substeps, 400)

    def test_gradient_through_the_core_is_finite_at_an_empty_mode(self):
        """Smoke test for differentiability; AER-11 does it properly.

        Empty modes are the sharp edge. Differentiates with one mode's
        mass driven to zero AND no gas tracers, the combination that gave
        a float32 NaN before condensation's floors were made float32-safe.
        """
        masses, numbers = _population()
        acc = MAM4_SPEC.modes[0].short
        for sp in MAM4_SPEC.modes[0].species:
            masses[mass_name(sp, acc)] = jnp.zeros(SHAPE)
        base, diagnostics = _state_and_diagnostics({})
        target = mass_name("so4", MAM4_SPEC.modes[1].short)

        def total_wet_radius(mass):
            state = base.copy(tracers={**masses, **numbers, target: mass})
            _, diag = JamModalMicrophysics()(state, diagnostics, None, None)
            return jnp.sum(diag["_jam_state"].r_wet)

        grad = jax.grad(total_wet_radius)(masses[target])
        self.assertTrue(np.all(np.isfinite(np.asarray(grad))))


if __name__ == "__main__":
    unittest.main()


# A MAM4 population under other names: every mode renamed (name and short),
# the role fields pointed at the new names, the emission table re-keyed.
_RENAMED = {
    "accum": ("soluble_accumulation", "sa"),
    "aitken": ("soluble_aitken", "sk"),
    "coarse": ("soluble_coarse", "sc"),
    "primary_carbon": ("insoluble_fresh", "if"),
}


def _renamed_mam4():
    short_of = {m.short: _RENAMED[m.name][1] for m in MAM4_SPEC.modes}
    modes = tuple(
        dataclasses.replace(m, name=_RENAMED[m.name][0], short=_RENAMED[m.name][1])
        for m in MAM4_SPEC.modes)
    emission = {
        sp: tuple((short_of[short], frac) for short, frac in entries)
        for sp, entries in MAM4_SPEC.primary_emission.items()}
    return dataclasses.replace(
        MAM4_SPEC, modes=modes, primary_emission=emission,
        accumulation_mode="soluble_accumulation", aitken_mode="sk",
        primary_carbon_mode="insoluble_fresh", coarse_modes=("soluble_coarse",)), short_of


class PopulationRolesTest(unittest.TestCase):
    """The core reads mode roles off the population, never MAM4's names."""

    def test_mam4_defaults_resolve_to_its_modes(self):
        spec = JamModalMicrophysics().spec
        self.assertEqual(spec.mode(spec.accumulation_mode).short, "acc")
        self.assertEqual(spec.mode(spec.aitken_mode).short, "ait")
        self.assertEqual(spec.mode(spec.primary_carbon_mode).short, "pcm")

    def test_renamed_population_gives_bitwise_identical_tendencies(self):
        """Same population, other names: nothing in the core may notice."""
        spec, short_of = _renamed_mam4()
        masses, numbers = _population()
        gases = {gas_name(g): jnp.full(SHAPE, 1.0e-11) for g in ("h2so4", "soag")}
        tracers = {**masses, **numbers, **gases}

        def rename(key):
            prefix, *rest = key.split("_")
            if prefix in ("m", "mc"):
                return f"{prefix}_{rest[0]}_{short_of[rest[1]]}"
            if prefix in ("n", "nc"):
                return f"{prefix}_{short_of[rest[0]]}"
            return key

        state, diagnostics = _state_and_diagnostics(tracers)
        renamed_state, renamed_diag = _state_and_diagnostics(
            {rename(k): v for k, v in tracers.items()})
        tend, diag = JamModalMicrophysics()(state, diagnostics, None, None)
        tend_r, diag_r = JamModalMicrophysics(spec=spec)(
            renamed_state, renamed_diag, None, None)
        self.assertEqual({rename(k) for k in tend.tracers}, set(tend_r.tracers))
        for key, value in tend.tracers.items():
            np.testing.assert_array_equal(
                np.asarray(value), np.asarray(tend_r.tracers[rename(key)]),
                err_msg=f"{key} differs under the renamed population")
        for field in ("r_dry", "r_wet", "mass", "number"):
            np.testing.assert_array_equal(
                np.asarray(getattr(diag["_jam_state"], field)),
                np.asarray(getattr(diag_r["_jam_state"], field)))

    def test_population_without_the_roles_is_refused_at_construction(self):
        with self.assertRaisesRegex(ValueError, "aitken_mode.*primary_carbon_mode"):
            JamModalMicrophysics(spec=M7_SPEC)
        with self.assertRaisesRegex(ValueError, "primary_carbon_mode=None"):
            JamModalMicrophysics(
                spec=dataclasses.replace(MAM4_SPEC, primary_carbon_mode=None))

