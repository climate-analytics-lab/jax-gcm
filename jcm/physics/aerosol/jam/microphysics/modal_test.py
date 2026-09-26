"""AER-01: the in-tree modal core's scaffold.

The acceptance criterion for this card is that the new core is
*behaviourally identical* to :class:`PlaceholderMicrophysics` — same
``_jam_state``, same zero tendency — because every process is still an
identity. That makes the first real process (AER-03) a change with a
baseline to diff against, instead of a new core landing with no reference.

Several tests here are DELIBERATELY temporary and named so: the
identity tests must be updated by the card that implements the process
they pin. A process that is filled in without its test changing is a
process nobody checked.
"""

import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.microphysics.modal import (
    PROCESS_ORDER,
    JamModalMicrophysics,
    MicrophysicsEnvironment,
    ModalPopulation,
)
from jcm.physics.aerosol.jam.microphysics.placeholder import (
    PlaceholderMicrophysics,
)
from jcm.physics.aerosol.jam.tracer_layout import (
    mass_name,
    number_name,
    tracer_specs,
)
from jcm.physics_interface import PhysicsState

NLEV, NCOLS = 4, 3


def _population(nlev=NLEV, ncols=NCOLS, seed=0):
    """Build a non-uniform population, so an identity test cannot pass trivially.

    Uniform fields would let a process that (say) overwrote a mode with its
    column mean still compare equal.
    """
    rng = np.random.default_rng(seed)
    masses = {
        mass_name(sp, mode.short): jnp.asarray(
            rng.uniform(1.0e-12, 1.0e-9, (nlev, ncols)))
        for mode in MAM4_SPEC.modes for sp in mode.species
    }
    numbers = {
        number_name(mode.short): jnp.asarray(
            rng.uniform(1.0e6, 1.0e9, (nlev, ncols)))
        for mode in MAM4_SPEC.modes
    }
    return masses, numbers


def _state_and_diagnostics(nlev=NLEV, ncols=NCOLS, seed=0):
    masses, numbers = _population(nlev, ncols, seed)
    state = PhysicsState.zeros((nlev, ncols)).copy(
        temperature=jnp.full((nlev, ncols), 285.0),
        specific_humidity=jnp.full((nlev, ncols), 0.005),
        tracers={**masses, **numbers},
    )
    diagnostics = {
        "pressure_full": jnp.full((nlev, ncols), 9.0e4),
        "_dt_seconds": 1800.0,
    }
    return state, diagnostics


class ScaffoldEquivalenceTest(unittest.TestCase):
    """The AER-01 criterion: identical to the placeholder, field by field."""

    def test_jam_state_is_bitwise_equal_to_the_placeholder(self):
        state, diagnostics = _state_and_diagnostics()
        _, modal_diag = JamModalMicrophysics()(state, diagnostics, None, None)
        _, place_diag = PlaceholderMicrophysics()(state, diagnostics, None, None)

        modal, place = modal_diag["_jam_state"], place_diag["_jam_state"]
        for field in ("r_dry", "r_wet", "rho", "kappa", "mass", "number"):
            # Exact, not almost-equal: both cores call the same
            # equilibrium_modal_state on the same inputs, so any difference
            # at all means a process stopped being an identity.
            np.testing.assert_array_equal(
                np.asarray(getattr(modal, field)),
                np.asarray(getattr(place, field)),
                err_msg=f"_jam_state.{field} diverged from the placeholder",
            )

    def test_tendency_is_zero_everywhere(self):
        state, diagnostics = _state_and_diagnostics()
        tendency, _ = JamModalMicrophysics()(state, diagnostics, None, None)
        for field in ("u_wind", "v_wind", "temperature", "specific_humidity"):
            self.assertTrue(
                bool(jnp.all(getattr(tendency, field) == 0.0)),
                msg=f"{field} tendency is nonzero at AER-01",
            )
        # AER-03 replaces this: the tracer dict becomes the population
        # change over dt, and this assertion must become a budget check.
        self.assertEqual(tendency.tracers, {})


class ProcessScaffoldTest(unittest.TestCase):

    def test_order_matches_mam4_amicphys(self):
        # Load-bearing, not incidental: condensation must precede
        # nucleation because both consume g_h2so4 (see PROCESS_ORDER).
        self.assertEqual(
            [name for name, _ in PROCESS_ORDER],
            ["gas_aerosol_exchange", "rename", "nucleation",
             "coagulation", "carbonaceous_ageing"],
        )

    def test_every_process_is_still_an_identity(self):
        """TEMPORARY — each card must delete its process from this list.

        If this test still passes in full after AER-03..AER-06 land, those
        cards did nothing.
        """
        masses, numbers = _population()
        population = ModalPopulation(masses=masses, numbers=numbers)
        env = MicrophysicsEnvironment(
            temperature=jnp.full((NLEV, NCOLS), 285.0),
            pressure=jnp.full((NLEV, NCOLS), 9.0e4),
            saturation=jnp.full((NLEV, NCOLS), 0.8),
            dt=1800.0,
        )
        for name, process in PROCESS_ORDER:
            out = process(population, env, MAM4_SPEC)
            for key, before in population.masses.items():
                np.testing.assert_array_equal(
                    np.asarray(out.masses[key]), np.asarray(before),
                    err_msg=f"{name} changed mass {key}")
            for key, before in population.numbers.items():
                np.testing.assert_array_equal(
                    np.asarray(out.numbers[key]), np.asarray(before),
                    err_msg=f"{name} changed number {key}")

    def test_process_sequence_can_be_overridden(self):
        """AER-09 bisects a budget failure by running one process at a time."""
        state, diagnostics = _state_and_diagnostics()
        term = JamModalMicrophysics(processes=PROCESS_ORDER[:1])
        self.assertEqual(len(term.processes), 1)
        _, diag = term(state, diagnostics, None, None)
        self.assertIn("_jam_state", diag)

    def test_empty_process_sequence_still_publishes_state(self):
        state, diagnostics = _state_and_diagnostics()
        _, diag = JamModalMicrophysics(processes=())(
            state, diagnostics, None, None)
        self.assertEqual(diag["_jam_state"].r_dry.shape,
                         (len(MAM4_SPEC.modes), NLEV, NCOLS))


class HarnessContractTest(unittest.TestCase):

    def test_required_tracers_match_the_population(self):
        self.assertEqual(
            {s.name for s in JamModalMicrophysics().required_tracers()},
            {s.name for s in tracer_specs(MAM4_SPEC)},
        )

    def test_runs_with_no_tracers_at_all(self):
        """``state.tracers`` is empty during Model.get_empty_data's probe."""
        state = PhysicsState.zeros((NLEV, NCOLS)).copy(
            temperature=jnp.full((NLEV, NCOLS), 285.0),
            specific_humidity=jnp.full((NLEV, NCOLS), 0.005),
        )
        diagnostics = {"pressure_full": jnp.full((NLEV, NCOLS), 9.0e4)}
        _, diag = JamModalMicrophysics()(state, diagnostics, None, None)
        self.assertTrue(
            np.all(np.isfinite(np.asarray(diag["_jam_state"].r_wet))))

    def test_missing_dt_falls_back_rather_than_raising(self):
        state, diagnostics = _state_and_diagnostics()
        del diagnostics["_dt_seconds"]
        _, diag = JamModalMicrophysics()(state, diagnostics, None, None)
        self.assertIn("_jam_state", diag)

    def test_gradient_through_the_core_is_finite(self):
        """Smoke test for the differentiability claim; AER-11 does it properly.

        Empty modes are the sharp edge — the guarded divisions in the
        κ-Köhler diagnostics must not emit NaN gradients — so this
        differentiates with one mode's mass driven to zero.
        """
        masses, numbers = _population()
        acc = MAM4_SPEC.modes[0].short
        for sp in MAM4_SPEC.modes[0].species:
            masses[mass_name(sp, acc)] = jnp.zeros((NLEV, NCOLS))
        base = _state_and_diagnostics()[0]
        diagnostics = {
            "pressure_full": jnp.full((NLEV, NCOLS), 9.0e4),
            "_dt_seconds": 1800.0,
        }
        target = mass_name("so4", MAM4_SPEC.modes[1].short)

        def total_wet_radius(mass):
            tracers = {**masses, **numbers, target: mass}
            state = base.copy(tracers=tracers)
            _, diag = JamModalMicrophysics()(state, diagnostics, None, None)
            return jnp.sum(diag["_jam_state"].r_wet)

        grad = jax.grad(total_wet_radius)(masses[target])
        self.assertTrue(np.all(np.isfinite(np.asarray(grad))))


if __name__ == "__main__":
    unittest.main()
