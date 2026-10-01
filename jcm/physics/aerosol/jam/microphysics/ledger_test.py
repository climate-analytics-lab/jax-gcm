"""AER-09: every process keeps the budgets its contract says it keeps.

The core is run with the per-process audit on, over populations chosen to
make every process act — vapour to condense and form particles, a cold
column for the binary pathway, a populated primary-carbon mode with a
shell to age, a cloud-borne Aitken phase to rename — and each process is
held to :data:`ledger.CONTRACT`. A deliberately leaking process is used
to show the ledger catches what it is for.
"""

import unittest

import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.cloud_borne_store import CARRY_KEY, mirror_names
from jcm.physics.aerosol.jam.microphysics import ledger
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.microphysics.modal import (
    JamModalMicrophysics,
    ModalPopulation,
    PROCESS_ORDER,
)
from jcm.physics.aerosol.jam.tracer_layout import gas_name, mass_name, number_name
from jcm.physics_interface import PhysicsState

NLEV, NCOLS = 4, 3
SHAPE = (NLEV, NCOLS)
DT = 1800.0
#: float32 sums of ~10 terms: 1e-7 relative each; the gate is well above.
RTOL = 1e-5


def _tracers(seed, *, gas, number_scale=1.0):
    rng = np.random.default_rng(seed)
    tracers = {mass_name(sp, m.short): jnp.asarray(rng.uniform(1e-12, 1e-9, SHAPE))
               for m in MAM4_SPEC.modes for sp in m.species}
    tracers.update({number_name(m.short): jnp.asarray(
        rng.uniform(1e6, 1e9, SHAPE) * number_scale) for m in MAM4_SPEC.modes})
    tracers.update({gas_name("dms"): jnp.full(SHAPE, 1e-11),
                    gas_name("so2"): jnp.full(SHAPE, 1e-10),
                    gas_name("h2so4"): jnp.full(SHAPE, gas),
                    gas_name("soag"): jnp.full(SHAPE, 1e-10)})
    return tracers


def _state(tracers, *, temperature=285.0):
    return PhysicsState.zeros(SHAPE).copy(
        temperature=jnp.full(SHAPE, temperature),
        specific_humidity=jnp.full(SHAPE, 0.004),
        tracers=tracers)


def _diagnostics(*, carry=True, pbl=None):
    diagnostics = {"pressure_full": jnp.full(SHAPE, 9.0e4), "_dt_seconds": DT}
    if pbl is not None:
        diagnostics["vertical_diffusion"] = pbl
    if carry:
        store = {name: jnp.zeros(SHAPE) for name in mirror_names(MAM4_SPEC)}
        store[number_name("ait", cloud_borne=True)] = jnp.full(SHAPE, 3.0e7)
        store[mass_name("so4", "ait", cloud_borne=True)] = jnp.full(SHAPE, 2.0e-11)
        store[number_name("acc", cloud_borne=True)] = jnp.full(SHAPE, 1.0e7)
        diagnostics[CARRY_KEY] = store
    return diagnostics


def _audit(state, diagnostics, **kwargs):
    _, out = JamModalMicrophysics(audit=True, **kwargs)(state, diagnostics, None, None)
    changes = {}
    for name, _ in PROCESS_ORDER:
        prefix = f"_modal_ledger_{name}_"
        changes[name] = {k[len(prefix):]: v for k, v in out.items() if k.startswith(prefix)}
    return changes, out


CASES = {
    "warm_modest_vapour": dict(tracers=_tracers(0, gas=1e-10), temperature=285.0),
    "warm_heavy_vapour": dict(tracers=_tracers(1, gas=5e-9), temperature=290.0),
    "cold_upper_troposphere": dict(tracers=_tracers(2, gas=2e-10, number_scale=1e-2),
                                   temperature=236.0),
    "near_empty": dict(tracers=_tracers(3, gas=1e-12, number_scale=1e-6), temperature=280.0),
}


class ContractTest(unittest.TestCase):

    def test_every_process_keeps_its_contract_in_every_case(self):
        for case, spec in CASES.items():
            state = _state(spec["tracers"], temperature=spec["temperature"])
            changes, _ = _audit(state, _diagnostics())
            for name, _ in PROCESS_ORDER:
                self.assertEqual(set(changes[name]), set(ledger.budgets(
                    _population(spec["tracers"]), MAM4_SPEC)), (case, name))
                bad = ledger.violations(name, changes[name], rtol=RTOL)
                self.assertEqual(bad, {}, f"{case}: {name} broke {bad}")

    def test_the_processes_actually_acted(self):
        """A ledger that passes because nothing happened proves nothing."""
        state = _state(CASES["warm_heavy_vapour"]["tracers"], temperature=290.0)
        changes, out = _audit(state, _diagnostics())
        # Exchange moved sulfur from gas to aerosol: the split changed even
        # though the total did not.
        self.assertGreater(float(jnp.max(out["_modal_condensation_h2so4_depletion"])), 0.0)
        # Coagulation lowered number; new-particle formation raised it
        # somewhere; renaming and ageing moved number between modes.
        self.assertLess(float(jnp.min(changes["coagulation"]["number"])), 0.0)
        self.assertGreater(float(jnp.max(changes["nucleation"]["number"])), 0.0)
        self.assertGreater(float(jnp.max(out["_modal_rename_number_fraction"])), 0.0)
        self.assertGreater(float(jnp.max(out["_modal_ageing_fraction"])), 0.0)

    def test_step_closure_is_always_published(self):
        state = _state(CASES["warm_modest_vapour"]["tracers"])
        _, out = JamModalMicrophysics()(state, _diagnostics(), None, None)
        for budget in ("sulfur", "organic", "number"):
            self.assertIn(f"_modal_ledger_step_{budget}", out)
        for budget in ("sulfur", "organic"):
            self.assertLess(float(jnp.max(jnp.abs(out[f"_modal_ledger_step_{budget}"]))), RTOL)
        self.assertFalse(any(k.startswith("_modal_ledger_gas_aerosol") for k in out))

    def test_audit_catches_a_leak(self):
        """The contract is only worth what the audit detects."""
        def leaky_exchange(population, env, spec, params):
            masses = dict(population.masses)
            key = mass_name("so4", "acc")
            masses[key] = masses[key] * 0.99
            return ModalPopulation(masses=masses, numbers=population.numbers,
                                   gases=population.gases, cloud_borne=population.cloud_borne,
                                   coating=population.coating,
                                   diagnostics=population.diagnostics)

        processes = (("gas_aerosol_exchange", leaky_exchange), *PROCESS_ORDER[1:])
        state = _state(CASES["warm_modest_vapour"]["tracers"])
        changes, _ = _audit(state, _diagnostics(), processes=processes)
        bad = ledger.violations("gas_aerosol_exchange",
                                changes["gas_aerosol_exchange"], rtol=RTOL)
        self.assertEqual(set(bad), {"sulfur"})
        for name, _ in PROCESS_ORDER[1:]:
            self.assertEqual(ledger.violations(name, changes[name], rtol=RTOL), {})

    def test_without_the_cloud_borne_phase(self):
        state = _state(CASES["warm_heavy_vapour"]["tracers"], temperature=290.0)
        import dataclasses
        spec = dataclasses.replace(MAM4_SPEC, cloud_borne=False)
        changes, _ = _audit(state, _diagnostics(carry=False), spec=spec)
        for name, _ in PROCESS_ORDER:
            self.assertEqual(ledger.violations(name, changes[name], rtol=RTOL), {}, name)


def _population(tracers):
    masses = {k: v for k, v in tracers.items() if k.startswith("m_")}
    numbers = {k: v for k, v in tracers.items() if k.startswith("n_")}
    gases = {k: v for k, v in tracers.items() if k.startswith("g_")}
    return ModalPopulation(masses=masses, numbers=numbers, gases=gases)


class BudgetTest(unittest.TestCase):

    def test_budgets_count_both_phases_and_every_gas(self):
        tracers = _tracers(0, gas=1e-10)
        population = _population(tracers)
        cloud_borne = {mass_name("so4", "ait", cloud_borne=True): jnp.full(SHAPE, 1e-10),
                       number_name("ait", cloud_borne=True): jnp.full(SHAPE, 5e6)}
        with_cb = ModalPopulation(masses=population.masses, numbers=population.numbers,
                                  gases=population.gases, cloud_borne=cloud_borne)
        plain, both = ledger.budgets(population, MAM4_SPEC), ledger.budgets(with_cb, MAM4_SPEC)
        np.testing.assert_allclose(np.asarray(both["sulfur"] - plain["sulfur"]), 1e-10 / 0.115,
                                   rtol=1e-5)
        # 5e6 added to totals of ~1e9: float32 resolves the difference to ~1e-4.
        np.testing.assert_allclose(np.asarray(both["number"] - plain["number"]), 5e6, rtol=1e-3)
        # Gases: 1e-11 DMS + 1e-10 SO2 + 1e-10 H2SO4 moles of S.
        expected_gas = 1e-11 / 0.0621324 + 1e-10 / 0.0640648 + 1e-10 / 0.0980784
        aerosol = sum(tracers[mass_name("so4", m.short)] for m in MAM4_SPEC.modes
                      if "so4" in m.species) / 0.115
        np.testing.assert_allclose(np.asarray(plain["sulfur"]), np.asarray(aerosol + expected_gas),
                                   rtol=1e-5)
        self.assertEqual(set(plain), {"sulfur", "organic", "bc", "du", "ss", "number"})

    def test_relative_change_of_an_empty_budget_is_zero(self):
        zeros = {"x": jnp.zeros(SHAPE)}
        change = ledger.relative_change(zeros, zeros)
        self.assertTrue(bool(jnp.all(change["x"] == 0.0)))


if __name__ == "__main__":
    unittest.main()
