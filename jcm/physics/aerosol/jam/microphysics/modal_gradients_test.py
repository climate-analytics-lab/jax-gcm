"""AER-11: the modal core differentiates, end to end and at every degenerate input.

Three questions, each answered on the full core rather than per module
(the per-module tests pin the closed forms):

* **Does AD agree with a central difference through the whole fold?** In
  float64 — the float32 roughness that defeats a two-rung difference test
  on erfc-shaped fractions is a known property, not a defect (AER-07) —
  with respect to O(1) multipliers on the tracer groups and to every
  parameter struct.
* **Is every derivative finite in float32 at the inputs a model actually
  produces?** Forward and reverse mode, through each process alone and the
  full core, at: a clean atmosphere; empty modes with vapour; mass with no
  number and number with no mass; negative tracers (spectral transport's
  Gibbs ringing is real input); cold and warm columns; saturation at its
  clip; an overwhelming vapour load; with and without the cloud-borne
  store. These are the ``x / max(y, tiny)`` traps found in AER-02..07, now
  swept systematically.
* **Does every tunable reach the output?** Each parameter leaf is either
  live at the populated point or named here as legitimately dead, so a
  leaf the code stops reading is caught.

The repo-wide ``jcm/physics/echam/term_gradients_test.py`` checks the same
core as a term in the ECHAM+JAM composition on a replayed column (slow
suite); this module is the fast, population-level complement.
"""

import dataclasses
import unittest

import jax
import jax.numpy as jnp
import numpy as np

from jcm.physics.aerosol.jam.cloud_borne_store import CARRY_KEY, mirror_names
from jcm.physics.aerosol.jam.microphysics import ageing
from jcm.physics.aerosol.jam.microphysics import condensation as cond
from jcm.physics.aerosol.jam.microphysics import new_particle_formation as npf
from jcm.physics.aerosol.jam.microphysics.mam4_data import MAM4_SPEC
from jcm.physics.aerosol.jam.microphysics.modal import (
    JamModalMicrophysics,
    PROCESS_ORDER,
)
from jcm.physics.aerosol.jam.tracer_layout import gas_name, mass_name, number_name
from jcm.physics_interface import PhysicsState
from jcm.testing import _cotangent, _tangent, check_gradients

SHAPE = (4, 3)
DT = 1800.0

MASS_KEYS = [mass_name(sp, m.short) for m in MAM4_SPEC.modes for sp in m.species]
NUMBER_KEYS = [number_name(m.short) for m in MAM4_SPEC.modes]
GAS_KEYS = [gas_name(g) for g in ("dms", "so2", "h2so4", "soag")]


def _tracers(seed=0, *, mass=1e-9, number=1e8, gas=1e-10):
    rng = np.random.default_rng(seed)
    out = {k: jnp.asarray(rng.uniform(0.1, 1.0, SHAPE) * mass) for k in MASS_KEYS}
    out.update({k: jnp.asarray(rng.uniform(0.1, 1.0, SHAPE) * number) for k in NUMBER_KEYS})
    out.update({k: jnp.full(SHAPE, gas) for k in GAS_KEYS})
    return out


def _state(tracers, *, temperature=285.0, humidity=0.004):
    return PhysicsState.zeros(SHAPE).copy(
        temperature=jnp.full(SHAPE, temperature),
        specific_humidity=jnp.full(SHAPE, humidity),
        tracers=tracers)


def _diagnostics(*, carry=True, pbl=None):
    diagnostics = {"pressure_full": jnp.full(SHAPE, 9.0e4), "_dt_seconds": DT}
    if carry:
        store = {name: jnp.zeros(SHAPE) for name in mirror_names(MAM4_SPEC)}
        store[number_name("ait", cloud_borne=True)] = jnp.full(SHAPE, 2.0e7)
        store[mass_name("so4", "ait", cloud_borne=True)] = jnp.full(SHAPE, 1.0e-11)
        diagnostics[CARRY_KEY] = store
    return diagnostics


def _outputs(core, state, diagnostics):
    tendency, out = core(state, diagnostics, None, None)
    result = dict(tendency.tracers)
    if CARRY_KEY in out:
        result.update({f"carry/{k}": v for k, v in out[CARRY_KEY].items()})
    return result


def _assert_finite_both_modes(f, args, label):
    """One jvp and one vjp, both finite — the repo-wide test's criterion."""
    primal, vjp_fun = jax.vjp(f, *args)
    _, forward = jax.jvp(f, args, _tangent(args, 0))
    reverse = vjp_fun(_cotangent(primal, 1))
    for mode, tree in (("forward", forward), ("reverse", reverse)):
        for leaf in jax.tree.leaves(tree):
            if jnp.issubdtype(jnp.result_type(leaf), jnp.floating):
                assert np.all(np.isfinite(np.asarray(leaf))), f"{label}: {mode} not finite"


# --- degenerate operating points ------------------------------------------

def _negative(tracers):
    out = dict(tracers)
    out[mass_name("so4", "ait")] = -jnp.abs(out[mass_name("so4", "ait")])
    out[number_name("acc")] = -jnp.abs(out[number_name("acc")])
    out[gas_name("h2so4")] = -jnp.abs(out[gas_name("h2so4")])
    return out


POINTS = {
    "clean": (_tracers(mass=0.0, number=0.0, gas=0.0), 285.0, 0.004, True),
    "empty_modes_with_vapour": (_tracers(mass=0.0, number=0.0, gas=1e-9), 285.0, 0.004, True),
    "mass_without_number": (_tracers(number=0.0), 285.0, 0.004, True),
    "number_without_mass": (_tracers(mass=0.0), 285.0, 0.004, True),
    "negative_tracers": (_negative(_tracers()), 285.0, 0.004, True),
    "cold": (_tracers(number=1e6), 236.0, 1e-5, True),
    "warm_saturated": (_tracers(), 310.0, 0.08, True),
    "vapour_overload": (_tracers(gas=1e-6), 285.0, 0.004, True),
    "populated": (_tracers(), 285.0, 0.004, True),
    "populated_no_store": (_tracers(), 285.0, 0.004, False),
}


class FiniteDerivativesTest(unittest.TestCase):

    def test_full_core_both_modes_at_every_point(self):
        for name, (tracers, temperature, humidity, carry) in POINTS.items():
            with self.subTest(point=name):
                core = JamModalMicrophysics()
                diagnostics = _diagnostics(carry=carry)

                def f(tracers_, carry_, core=core, diagnostics=diagnostics,
                      temperature=temperature, humidity=humidity):
                    diag = dict(diagnostics)
                    if carry_ is not None:
                        diag[CARRY_KEY] = carry_
                    return _outputs(core, _state(tracers_, temperature=temperature,
                                                 humidity=humidity), diag)

                args = (tracers, diagnostics.get(CARRY_KEY))
                if args[1] is None:
                    _assert_finite_both_modes(lambda t, f=f: f(t, None), (tracers,), name)
                else:
                    _assert_finite_both_modes(f, args, name)

    def test_each_process_alone_both_modes_at_every_point(self):
        """A trap in one process would be masked if an earlier one zeroed its input."""
        for process_name, process in PROCESS_ORDER:
            for name, (tracers, temperature, humidity, carry) in POINTS.items():
                with self.subTest(process=process_name, point=name):
                    core = JamModalMicrophysics(processes=((process_name, process),))
                    diagnostics = _diagnostics(carry=carry)

                    def f(tracers_, core=core, diagnostics=diagnostics,
                          temperature=temperature, humidity=humidity):
                        return _outputs(core, _state(tracers_, temperature=temperature,
                                                     humidity=humidity), diagnostics)

                    _assert_finite_both_modes(f, (tracers,), f"{process_name}/{name}")

    def test_parameter_derivatives_finite_at_degenerate_points(self):
        """d(outputs)/d(parameter), where the #663 class of trap lives."""
        for name, (tracers, temperature, humidity, carry) in POINTS.items():
            with self.subTest(point=name):
                diagnostics = _diagnostics(carry=carry)
                state = _state(tracers, temperature=temperature, humidity=humidity)

                def f(cond_p, npf_p, age_p, state=state, diagnostics=diagnostics):
                    core = JamModalMicrophysics(condensation_params=cond_p,
                                                npf_params=npf_p, ageing_params=age_p)
                    return _outputs(core, state, diagnostics)

                _assert_finite_both_modes(
                    f, (cond.CondensationParameters(), npf.NewParticleFormationParameters(),
                        ageing.AgeingParameters()), name)


class ReferenceTest(unittest.TestCase):
    """AD against a central difference through the whole fold, in float64."""

    def _in_x64(self, body):
        was = bool(jax.config.read("jax_enable_x64"))
        jax.config.update("jax_enable_x64", True)
        try:
            body()
        finally:
            jax.config.update("jax_enable_x64", was)

    def test_against_central_difference_in_state_and_parameters(self):
        def body():
            tracers = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in _tracers().items()}
            diagnostics = _diagnostics()
            diagnostics[CARRY_KEY] = {k: jnp.asarray(v, dtype=jnp.float64)
                                      for k, v in diagnostics[CARRY_KEY].items()}
            state = _state(tracers)
            state = state.copy(temperature=state.temperature.astype(jnp.float64),
                               specific_humidity=state.specific_humidity.astype(jnp.float64))

            def f(mass_scale, number_scale, gas_scale, cond_p, npf_p, age_p):
                scaled = {}
                for k, v in tracers.items():
                    scale = (mass_scale if k.startswith("m_") else
                             number_scale if k.startswith("n_") else gas_scale)
                    scaled[k] = v * scale
                core = JamModalMicrophysics(condensation_params=cond_p, npf_params=npf_p,
                                            ageing_params=age_p)
                return _outputs(core, state.copy(tracers=scaled), diagnostics)

            params = (cond.CondensationParameters(), npf.NewParticleFormationParameters(),
                      ageing.AgeingParameters())
            params = jax.tree.map(lambda x: jnp.asarray(x, dtype=jnp.float64), params)
            check_gradients(
                f, (jnp.asarray(1.0), jnp.asarray(1.0), jnp.asarray(1.0), *params),
                rtol=2e-2, atol=1e-8,
                # The empirical-pathway switches are steps in these leaves.
                fixed_inputs=("[4]/.pbl_minimum_depth", "[4]/.pbl_enabled"))

        self._in_x64(body)


#: Parameter leaves that are legitimately dead at the populated point, each
#: with the reason. A leaf that comes to life fails the test so the entry is
#: revisited, as the repo-wide test does.
DEAD_AT_POPULATED = {
    "pbl_minimum_depth": "a step: the level is 500 m up, the PBL height zero, "
                         "so the 100 m floor is not crossed by a perturbation",
}


class LivenessTest(unittest.TestCase):

    def test_every_parameter_leaf_reaches_the_outputs_or_is_named_dead(self):
        tracers = _tracers(gas=5e-9)
        diagnostics = _diagnostics()
        state = _state(tracers)

        def f(cond_p, npf_p, age_p):
            core = JamModalMicrophysics(condensation_params=cond_p, npf_params=npf_p,
                                        ageing_params=age_p)
            return _outputs(core, state, diagnostics)

        params = (cond.CondensationParameters(), npf.NewParticleFormationParameters(),
                  ageing.AgeingParameters())
        primal, vjp_fun = jax.vjp(f, *params)
        grads = vjp_fun(_cotangent(primal, 1))
        for struct in grads:
            for field in dataclasses.fields(struct):
                if not jnp.issubdtype(jnp.result_type(getattr(struct, field.name)),
                                      jnp.floating):
                    continue
                value = np.asarray(getattr(struct, field.name))
                live = bool(np.any(value != 0.0))
                if field.name in DEAD_AT_POPULATED:
                    self.assertFalse(live, f"{field.name} is listed dead but moves the "
                                           "outputs: drop it from the table")
                else:
                    self.assertTrue(live, f"{field.name} has a zero derivative: "
                                          "is it still read?")


if __name__ == "__main__":
    unittest.main()
