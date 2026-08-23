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


def _sl_available() -> bool:
    from jcm.dycore.dinosaur.dycore import semi_lagrangian_available

    return semi_lagrangian_available()


def _small_dycore(**kwargs):
    from jcm.dycore.dinosaur.dycore import DinosaurDycore
    from jcm.physics.speedy.speedy_coords import get_speedy_coords
    from jcm.terrain import TerrainData

    coords = get_speedy_coords(layers=8, spectral_truncation=21)
    return DinosaurDycore(
        coords=coords,
        terrain=TerrainData.aquaplanet(coords),
        dt_seconds=2400.0,
        **kwargs,
    )


@unittest.skipUnless(_sl_available(), "needs the semi-Lagrangian dinosaur")
class TracerRegistrationSyncTest(unittest.TestCase):
    """Late tracer registration must reconfigure the SL transport.

    ``Model.__init__`` writes ``dycore.tracer_specs`` after construction
    (the supported pre-built-dycore path ships default empty specs). The
    nodal registration is baked into the primitive, filters and step
    function, so that write has to rebuild them — otherwise every
    late-registered tracer would silently ride modal, defeating the
    nodal/monotone transport guarantee (#625 review P1).
    """

    def test_assigning_tracer_specs_rebuilds_the_nodal_registration(self):
        from jcm.physics.physics_term import TracerSpec

        dycore = _small_dycore()
        self.assertEqual(dycore.primitive.nodal_tracers, ())

        stale_primitive = dycore.primitive
        dycore.tracer_specs = {"dust": TracerSpec(name="dust")}
        self.assertIsNot(dycore.primitive, stale_primitive)
        self.assertEqual(tuple(dycore.primitive.nodal_tracers), ("dust",))

    def test_reassigning_identical_specs_does_not_rebuild(self):
        from jcm.physics.physics_term import TracerSpec

        specs = {"dust": TracerSpec(name="dust")}
        dycore = _small_dycore(tracer_specs=specs)
        primitive = dycore.primitive
        dycore.tracer_specs = dict(specs)
        self.assertIs(dycore.primitive, primitive)

    def test_model_with_prebuilt_dycore_registers_physics_tracers(self):
        from typing import ClassVar

        from jcm.model import Model
        from jcm.physics.composable_physics import ComposablePhysics
        from jcm.physics.physics_term import PhysicsTerm, TracerSpec
        from jcm.physics_interface import PhysicsTendency

        class TracerTerm(PhysicsTerm):
            name: ClassVar[str] = "tracer_term"
            category: ClassVar[str] = "test"

            @classmethod
            def required_tracers(cls):
                return (TracerSpec("test_tracer"),)

            def __call__(self, state, diagnostics, forcing, terrain):
                return PhysicsTendency.zeros(state.temperature.shape), diagnostics

        # The supported pre-built-dycore path: specs default empty here...
        dycore = _small_dycore()
        model = Model(dycore, physics=ComposablePhysics(terms=[TracerTerm()]))
        # ...and Model's post-construction sync must reach the primitive.
        self.assertIn("test_tracer", model.dycore.primitive.nodal_tracers)


@unittest.skipUnless(_sl_available(), "needs the semi-Lagrangian dinosaur")
class OffCenteringDefaultTest(unittest.TestCase):
    """Direct construction must default to the validated off-centering.

    Zero off-centering is unstable over real orography (see
    docs/source/design/dinosaur_sl_jam_configuration.md), so the
    constructor default has to match the runner's validated value rather
    than silently handing non-Hydra users the unstable configuration
    (#625 review P1).
    """

    def test_constructor_default_is_the_shared_constant(self):
        from jcm.dycore.dinosaur.dycore import DEFAULT_OFF_CENTERING

        self.assertEqual(DEFAULT_OFF_CENTERING, 0.2)
        self.assertEqual(_small_dycore().off_centering, DEFAULT_OFF_CENTERING)

    def test_sl_options_still_override(self):
        dycore = _small_dycore(sl_options={"off_centering": 0.05})
        self.assertEqual(dycore.off_centering, 0.05)


@unittest.skipUnless(_sl_available(), "needs the semi-Lagrangian dinosaur")
class TracerMassFixerTest(unittest.TestCase):
    """The SL global mass fixer (#713).

    Semi-Lagrangian transport does not conserve tracer mass; with the
    quasi-monotone limiter the error is systematically POSITIVE wherever
    strong sinks leave sharp minima (the limiter can only add mass when it
    clips an interpolation undershoot at a minimum). The 2026-08 aerosol
    runaway compounded exactly this. The fixer restores each nodal
    tracer's global ``integral(q dp)`` to its pre-transport value every
    step.
    """

    def _dycore_with_tracer(self):
        from jcm.physics.physics_term import TracerSpec

        dycore = _small_dycore(
            tracer_specs={"dust": TracerSpec(name="dust")},
        )
        state = dycore.initial_state(None, random_seed=0)
        return dycore, state

    def _mass(self, dycore, state):
        import jax.numpy as jnp
        import numpy as np

        w = np.asarray(dycore.coords.horizontal.quadrature_weights)
        dp = dycore._nodal_tracer_column_weight(state)
        return float(jnp.sum(state.tracers["dust"] * dp * w))

    def test_fixer_restores_transport_mass_error(self):
        import jax.numpy as jnp

        dycore, state = self._dycore_with_tracer()
        # A rough positive field (the regime where SL leaks).
        q = 1e-9 * (1.0 + 0.9 * jnp.sin(
            jnp.arange(state.tracers["dust"].size, dtype=jnp.float64)
        ).reshape(state.tracers["dust"].shape))
        state.tracers = {**state.tracers, "dust": jnp.abs(q)}
        # Fabricate a 1% transport gain on the same state.
        gained = state.replace(
            tracers={**state.tracers, "dust": 1.01 * state.tracers["dust"]},
        )
        fixed = dycore._fix_nodal_tracer_mass(state, gained)
        self.assertAlmostEqual(
            self._mass(dycore, fixed) / self._mass(dycore, state), 1.0,
            places=10,
        )

    def test_fixer_guards_empty_fields_and_clips(self):
        import jax.numpy as jnp
        import numpy as np

        dycore, state = self._dycore_with_tracer()
        zero = state.replace(
            tracers={**state.tracers,
                     "dust": jnp.zeros_like(state.tracers["dust"])},
        )
        # Zero reference AND zero current: passes through untouched.
        out = dycore._fix_nodal_tracer_mass(zero, zero)
        np.testing.assert_array_equal(np.asarray(out.tracers["dust"]), 0.0)
        # A 10x discrepancy is not interpolation error: the clip bounds the
        # correction at 1.5x rather than silently absorbing a real bug.
        small = state.replace(
            tracers={**state.tracers,
                     "dust": jnp.full_like(state.tracers["dust"], 1e-10)},
        )
        big = state.replace(
            tracers={**state.tracers,
                     "dust": jnp.full_like(state.tracers["dust"], 1e-9)},
        )
        out = dycore._fix_nodal_tracer_mass(big, small)
        np.testing.assert_allclose(
            np.asarray(out.tracers["dust"]), 1.5e-10, rtol=1e-6,
        )

    def test_step_conserves_nodal_tracer_mass(self):
        import jax.numpy as jnp

        dycore, state = self._dycore_with_tracer()
        # Sharp positive blob: the hard case for SL conservation.
        shape = state.tracers["dust"].shape
        q = jnp.zeros(shape, dtype=jnp.float64)
        q = q.at[:, 10:14, 20:24].set(1e-8)
        state.tracers = {**state.tracers, "dust": q}
        m0 = self._mass(dycore, state)
        s = state
        for _ in range(5):
            s = dycore.step(s, None)
        m5 = self._mass(dycore, s)
        # ps evolves, so integral(q dp) is conserved per step against the
        # step's own reference; over 5 dry adiabatic steps the drift must
        # be at the clip/roundoff level, not the raw SL leak.
        self.assertAlmostEqual(m5 / m0, 1.0, places=6)
