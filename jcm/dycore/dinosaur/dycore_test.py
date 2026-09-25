"""Tests for the dinosaur dycore backend's transport contract."""

import unittest


class SemiLagrangianRequiredTest(unittest.TestCase):
    """The SL core is a hard requirement of the dinosaur backend.

    Every tracer-carrying configuration must use it — Eulerian spectral
    transport rang negative on sharp emission sources and NaN'd the aerosol
    microphysics (#521) — so a dinosaur without it is not a usable install
    even for the tracer-free Eulerian path.
    """

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
class AdvectionSelectionTest(unittest.TestCase):
    """``advection`` selection: explicit, physics-decided, and the #521 guard.

    Eulerian spectral transport is offered only for tracer-free physics
    (SPEEDY declares it — it carries no extra tracers and SL costs ~4x its
    CPU step for nothing). Spectral transport of a sharp tracer rings
    negative and NaN'd the aerosol microphysics (#521), so no path may put
    a tracer on it: an explicit request raises, and the physics-decided
    mode falls back to semi-Lagrangian.
    """

    def _dust(self):
        from jcm.physics.physics_term import TracerSpec

        return {"dust": TracerSpec(name="dust")}

    def test_unknown_scheme_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "advection must be one of"):
            _small_dycore(advection="spectral")

    def test_explicit_eulerian_builds_the_spectral_core(self):
        from dinosaur import primitive_equations

        dycore = _small_dycore(advection="eulerian")
        self.assertEqual(dycore.advection, "eulerian")
        self.assertIsInstance(dycore.primitive, primitive_equations.PrimitiveEquations)
        self.assertNotIsInstance(
            dycore.primitive, primitive_equations.SemiLagrangianPrimitiveEquations)
        self.assertEqual(dycore._nodal_tracers, ())

    def test_explicit_eulerian_with_tracers_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "#521"):
            _small_dycore(advection="eulerian", tracer_specs=self._dust())

    def test_explicit_eulerian_rejects_late_tracer_registration(self):
        dycore = _small_dycore(advection="eulerian")
        primitive = dycore.primitive
        with self.assertRaisesRegex(ValueError, "#521"):
            dycore.tracer_specs = self._dust()
        # The refusal is all-or-nothing: a caller that catches it keeps a
        # tracer-free Eulerian dycore, not tracer specs on spectral transport.
        self.assertEqual(dycore.tracer_specs, {})
        self.assertIs(dycore.primitive, primitive)
        self.assertEqual(dycore.advection, "eulerian")
        dycore.tracer_specs = {}  # still consistent: no rebuild needed

    def test_failed_rebuild_restores_transport_state(self):
        from unittest import mock

        dycore = _small_dycore()
        before = (dycore.tracer_specs, dycore.advection, dycore.primitive,
                  dycore._dynamics_step_fn)
        with mock.patch.object(type(dycore), "_build_filters",
                               side_effect=RuntimeError("boom")):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                dycore.tracer_specs = self._dust()
            with self.assertRaisesRegex(RuntimeError, "boom"):
                dycore.resolve_advection("eulerian")
        self.assertEqual(
            (dycore.tracer_specs, dycore.advection, dycore.primitive,
             dycore._dynamics_step_fn), before)

    def test_unresolved_default_is_semi_lagrangian(self):
        dycore = _small_dycore()
        self.assertIsNone(dycore.advection_requested)
        self.assertEqual(dycore.advection, "semi_lagrangian")

    def test_auto_mode_adopts_a_tracer_free_eulerian_preference(self):
        dycore = _small_dycore()
        self.assertEqual(dycore.resolve_advection("eulerian"), "eulerian")
        self.assertEqual(dycore._nodal_tracers, ())

    def test_auto_mode_keeps_tracers_semi_lagrangian(self):
        dycore = _small_dycore(tracer_specs=self._dust())
        self.assertEqual(dycore.resolve_advection("eulerian"), "semi_lagrangian")
        self.assertEqual(tuple(dycore.primitive.nodal_tracers), ("dust",))

    def test_auto_mode_leaves_eulerian_when_tracers_arrive(self):
        dycore = _small_dycore()
        dycore.resolve_advection("eulerian")
        dycore.tracer_specs = self._dust()
        self.assertEqual(dycore.advection, "semi_lagrangian")
        self.assertEqual(tuple(dycore.primitive.nodal_tracers), ("dust",))

    def test_explicit_choice_beats_the_physics_preference(self):
        dycore = _small_dycore(advection="semi_lagrangian")
        self.assertEqual(dycore.resolve_advection("eulerian"), "semi_lagrangian")

    def test_bad_physics_preference_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "preferred_advection"):
            _small_dycore().resolve_advection("spectral")

    def test_model_resolves_speedy_to_eulerian(self):
        from jcm.model import Model
        from jcm.physics.speedy.speedy_terms import speedy_physics

        model = Model(coords=_small_dycore().coords, physics=speedy_physics())
        self.assertEqual(model.dycore.advection, "eulerian")
        # Explicit dycore in auto mode resolves the same way.
        model = Model(_small_dycore(), physics=speedy_physics())
        self.assertEqual(model.dycore.advection, "eulerian")

    def test_model_puts_speedy_plus_tracers_on_semi_lagrangian(self):
        from typing import ClassVar

        from jcm.model import Model
        from jcm.physics.physics_term import PhysicsTerm, TracerSpec
        from jcm.physics.speedy.speedy_terms import speedy_physics
        from jcm.physics_interface import PhysicsTendency

        class DustTerm(PhysicsTerm):
            name: ClassVar[str] = "dust_term"
            category: ClassVar[str] = "test"

            @classmethod
            def required_tracers(cls):
                return (TracerSpec("dust"),)

            def __call__(self, state, diagnostics, forcing, terrain):
                return PhysicsTendency.zeros(state.temperature.shape), diagnostics

        physics = speedy_physics() + DustTerm()
        self.assertEqual(physics.preferred_advection(), "eulerian")
        model = Model(_small_dycore(), physics=physics)
        self.assertEqual(model.dycore.advection, "semi_lagrangian")
        self.assertIn("dust", model.dycore.primitive.nodal_tracers)

    def test_model_leaves_held_suarez_semi_lagrangian(self):
        from jcm.model import Model
        from jcm.physics.held_suarez.held_suarez_physics import held_suarez_physics

        model = Model(_small_dycore(), physics=held_suarez_physics())
        self.assertEqual(model.dycore.advection, "semi_lagrangian")

    def test_eulerian_speedy_steps_finite(self):
        import jax
        import numpy as np

        from jcm.forcing import ForcingData
        from jcm.model import Model
        from jcm.physics.speedy.speedy_terms import speedy_physics

        dycore = _small_dycore(advection="eulerian")
        model = Model(dycore, physics=speedy_physics())
        preds = model.run(
            forcing=ForcingData.zeros(dycore.coords.horizontal.nodal_shape),
            save_interval=0.25, total_time=0.5)
        for leaf in jax.tree_util.tree_leaves(preds.dynamics):
            self.assertTrue(np.all(np.isfinite(np.asarray(leaf))))


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

    def setUp(self):
        # The fixer closes integral(q dp) to roundoff, so the closure
        # assertions below only hold in float64: in float32 the residual is
        # ~1e-6, the precision of the sum itself rather than a transport leak.
        import jax

        prior = jax.config.read("jax_enable_x64")
        jax.config.update("jax_enable_x64", True)
        self.addCleanup(jax.config.update, "jax_enable_x64", prior)

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
