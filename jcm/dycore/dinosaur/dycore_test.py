"""Tests for the dinosaur dycore backend: transport and trajectory output."""

import unittest

import numpy as np


class SemiLagrangianRequiredTest(unittest.TestCase):
    """The SL core is a hard requirement of the dinosaur backend.

    It is the default transport and what the physics-decided mode always uses
    for tracer-carrying physics — Eulerian spectral transport rings negative on
    sharp emission sources and NaN'd the aerosol microphysics (#521), so an
    explicit Eulerian request with tracers only warns — and a dinosaur without
    it is not a usable install even for the tracer-free Eulerian path.
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
    """``advection`` selection: explicit, physics-decided, and the #521 warning.

    Eulerian spectral transport is meant for tracer-free physics (SPEEDY
    declares it — it carries no extra tracers and SL costs ~4x its CPU step
    for nothing). Spectral transport of a sharp tracer rings negative and
    NaN'd the aerosol microphysics (#521), so the physics-decided mode never
    picks it for tracers, and an explicit request with tracers warns.
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

    def test_explicit_eulerian_with_tracers_warns(self):
        with self.assertLogs("jcm.dycore.dinosaur.dycore", "WARNING") as logs:
            dycore = _small_dycore(advection="eulerian", tracer_specs=self._dust())
        self.assertIn("#521", logs.output[0])
        self.assertEqual(dycore.advection, "eulerian")

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


class TrajectoryToXarrayTest(unittest.TestCase):
    """``DinosaurDycore.to_xarray`` converts a real run's predictions.

    The protocol makes each backend own its trajectory conversion, and
    ``ModelPredictions.to_xarray`` (what the chunked CLI writes) delegates
    here, so a direct ``model.dycore.to_xarray(predictions, labels)`` has to
    accept what a run returns — physics diagnostics nested in dicts such as
    ``_prev_step`` and ``water_positivity_correction`` — name them as the
    written files do, and keep the exact ``datetime64`` labels it is given as
    the time axis (#951).
    """

    @classmethod
    def setUpClass(cls):
        from jcm.model import Model
        from jcm.physics.held_suarez.held_suarez_physics import (
            held_suarez_physics,
        )
        from jcm.physics.held_suarez.utils import get_held_suarez_coords
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.physics.speedy.speedy_terms import speedy_physics
        from jcm.terrain import TerrainData

        # The reproduction from the issue, verbatim.
        c = get_held_suarez_coords()
        cls.held_suarez = Model(coords=c, terrain=TerrainData.from_coords(c),
                                time_step=180, physics=held_suarez_physics())
        cls.held_suarez_run = cls.held_suarez.run(
            save_interval="3 hours", total_time="6 hours")

        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        cls.speedy = Model(coords=coords,
                           terrain=TerrainData.from_coords(coords),
                           physics=speedy_physics(), time_step=30.0,
                           start_time="2001-07-01")
        cls.speedy_run = cls.speedy.run(save_interval="1 hour",
                                        total_time="2 hours")
        cls.speedy_means = cls.speedy.run(save_interval="1 hour",
                                          total_time="2 hours",
                                          output_averages=True)

    def _direct(self, model, predictions):
        return model.dycore.to_xarray(predictions._predictions,
                                      predictions.time_labels())

    def test_the_issue_reproduction_converts(self):
        # Raised ``AttributeError: 'dict' object has no attribute 'shape'``.
        predictions = self.held_suarez_run
        self.assertIn("_prev_step", predictions.physics)
        self.assertIn("water_positivity_correction", predictions.physics)
        ds = self._direct(self.held_suarez, predictions)
        self.assertIn("water_positivity_correction.total_water_tendency", ds)
        self.assertFalse([name for name in ds.data_vars
                          if name.startswith("_prev_step")])

    def test_variables_match_what_the_written_file_holds(self):
        """The CLI writes ``ModelPredictions.to_xarray()``; so must this.

        The only variable the wrapper adds is ``time_bounds``; for interval
        means it also drops the categorical diagnostics it lists in
        ``omitted_interval_mean_variables``, which a trajectory conversion
        (before cell methods exist) keeps.
        """
        cases = (("held_suarez", self.held_suarez, self.held_suarez_run),
                 ("speedy", self.speedy, self.speedy_run),
                 ("speedy means", self.speedy, self.speedy_means))
        for name, model, predictions in cases:
            with self.subTest(case=name):
                direct = self._direct(model, predictions)
                written = predictions.to_xarray()
                omitted = written.attrs.get(
                    "omitted_interval_mean_variables", "")
                expected = (set(written.data_vars) - {"time_bounds"}
                            | set(filter(None, omitted.split(","))))
                self.assertEqual(set(direct.data_vars), expected)
                for var in ("temperature", "u_wind"):
                    self.assertEqual(direct[var].dims, written[var].dims)
                    self.assertEqual(direct[var].attrs.get("units"),
                                     written[var].attrs.get("units"))
                    np.testing.assert_array_equal(direct[var].values,
                                                  written[var].values)
        # A SPEEDY run's typed sub-structs are named by field, not position.
        self.assertIn("condensation.precls", self._direct(
            self.speedy, self.speedy_run))
        self.assertIn("iptop", " ".join(
            self.speedy_means.to_xarray().attrs[
                "omitted_interval_mean_variables"].split(",")))

    def test_a_trajectory_fetched_to_the_host_writes_the_same_file(self):
        """Host arrays write the same variables as device arrays.

        ``jax.device_get`` turns every leaf into a numpy array; the file a
        trajectory writes must not depend on where its arrays live.
        """
        import jax

        for model, predictions in ((self.held_suarez, self.held_suarez_run),
                                   (self.speedy, self.speedy_run)):
            on_host = jax.device_get(predictions).with_context(model)
            self.assertEqual(set(on_host.to_xarray().data_vars),
                             set(predictions.to_xarray().data_vars))

    def test_time_axis_is_the_exact_labels_given(self):
        for model, predictions in ((self.held_suarez, self.held_suarez_run),
                                   (self.speedy, self.speedy_run),
                                   (self.speedy, self.speedy_means)):
            labels = predictions.time_labels()
            self.assertTrue(np.issubdtype(labels.dtype, np.datetime64))
            ds = self._direct(model, predictions)
            np.testing.assert_array_equal(ds["time"].values, labels)
        # Dated, not elapsed: the SPEEDY run starts on 2001-07-01.
        np.testing.assert_array_equal(
            self._direct(self.speedy, self.speedy_run)["time"].values,
            np.array(["2001-07-01T01:00", "2001-07-01T02:00"],
                     dtype="datetime64[ms]"))
        # An elapsed-time axis carries no date, so it is refused rather than
        # silently relabelled.
        with self.assertRaisesRegex(TypeError, "datetime64"):
            self.held_suarez.dycore.to_xarray(
                self.held_suarez_run._predictions, np.array([0.125, 0.25]))

    def test_explicit_physics_wins_and_none_bound_is_refused(self):
        from jcm.dycore.dinosaur.dycore import DinosaurDycore

        predictions = self.held_suarez_run
        # Passing the physics explicitly is the same conversion.
        ds = self.held_suarez.dycore.to_xarray(
            predictions._predictions, predictions.time_labels(),
            physics=self.held_suarez.physics)
        self.assertEqual(set(ds.data_vars),
                         set(self._direct(self.held_suarez,
                                          predictions).data_vars))
        # A dycore no Model composed has nothing to name the diagnostics
        # with; that is an error, not a guess at their names.
        standalone = DinosaurDycore(coords=self.held_suarez.coords,
                                    terrain=self.held_suarez.terrain,
                                    dt_seconds=180.0)
        self.assertIsNone(standalone.output_physics)
        with self.assertRaisesRegex(TypeError, "no physics package"):
            standalone.to_xarray(predictions._predictions,
                                 predictions.time_labels())
