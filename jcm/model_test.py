import logging
import unittest
import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_vjp, check_jvp
import functools

class TestModelUnit(unittest.TestCase):
    def setUp(self):
        global speedy_physics, Parameters
        from jcm.physics.speedy.speedy_terms import speedy_physics
        from jcm.physics.speedy.params import Parameters

    def test_held_suarez_model(self):
        from jcm.physics.held_suarez.held_suarez_physics import held_suarez_physics
        from jcm.model import Model
        from jcm.terrain import TerrainData
        from jcm.physics.held_suarez.utils import get_held_suarez_coords

        coords = get_held_suarez_coords()
        terrain = TerrainData.from_coords(coords)
        model = Model(
            coords=coords,
            terrain=terrain,
            time_step=180,
            physics=held_suarez_physics(),
        )

        save_interval, total_time = 1, 2
        predictions = model.run(
            total_time=total_time,
            save_interval=save_interval,
        )
        final_state, dynamics_predictions = model._final_dycore_state, predictions.dynamics

        modal_zxy, nodal_zxy = model.coords.modal_shape, model.coords.nodal_shape
        nodal_tzxy = (int(total_time / save_interval),) + nodal_zxy

        self.assertIsNotNone(final_state.log_surface_pressure)
        self.assertIsNotNone(final_state.tracers['specific_humidity'])

        self.assertIsNotNone(dynamics_predictions.u_wind)
        self.assertIsNotNone(dynamics_predictions.v_wind)
        self.assertIsNotNone(dynamics_predictions.temperature)
        self.assertIsNotNone(dynamics_predictions.specific_humidity)
        self.assertIsNotNone(dynamics_predictions.geopotential)
        self.assertIsNotNone(dynamics_predictions.normalized_surface_pressure)

        self.assertTupleEqual(final_state.divergence.shape, modal_zxy)
        self.assertTupleEqual(final_state.vorticity.shape, modal_zxy)
        self.assertTupleEqual(final_state.temperature_variation.shape, modal_zxy)
        self.assertTupleEqual(final_state.log_surface_pressure.shape, (1,) + modal_zxy[1:])
        self.assertTupleEqual(final_state.tracers['specific_humidity'].shape, modal_zxy)

        self.assertTupleEqual(dynamics_predictions.u_wind.shape, nodal_tzxy)
        self.assertTupleEqual(dynamics_predictions.v_wind.shape, nodal_tzxy)
        self.assertTupleEqual(dynamics_predictions.temperature.shape, nodal_tzxy)
        self.assertTupleEqual(dynamics_predictions.specific_humidity.shape, nodal_tzxy)
        self.assertTupleEqual(dynamics_predictions.geopotential.shape, nodal_tzxy)
        self.assertTupleEqual(dynamics_predictions.normalized_surface_pressure.shape, (nodal_tzxy[0],) + nodal_tzxy[2:])
        
    def test_speedy_model(self):
        from jcm.model import Model
        from jcm.physics.speedy.speedy_coords import get_speedy_coords

        # Create model that goes through one timestep
        model = Model(
            coords=get_speedy_coords(),
            time_step=720,
        )

        save_interval, total_time = 1, 2
        predictions = model.run(
            save_interval=save_interval,
            total_time=total_time,
        )
        final_state, dynamics_predictions = model._final_dycore_state, predictions.dynamics

        modal_zxy, nodal_zxy = model.coords.modal_shape, model.coords.nodal_shape
        nodal_tzxy = (int(total_time / save_interval),) + nodal_zxy

        self.assertIsNotNone(final_state)
        self.assertIsNotNone(dynamics_predictions)

        self.assertIsNotNone(final_state.divergence)
        self.assertIsNotNone(final_state.vorticity)
        self.assertIsNotNone(final_state.temperature_variation)
        self.assertIsNotNone(final_state.log_surface_pressure)
        self.assertIsNotNone(final_state.tracers['specific_humidity'])

        self.assertIsNotNone(dynamics_predictions.u_wind)
        self.assertIsNotNone(dynamics_predictions.v_wind)
        self.assertIsNotNone(dynamics_predictions.temperature)
        self.assertIsNotNone(dynamics_predictions.specific_humidity)
        self.assertIsNotNone(dynamics_predictions.geopotential)
        self.assertIsNotNone(dynamics_predictions.normalized_surface_pressure)

        self.assertTupleEqual(final_state.divergence.shape, modal_zxy)
        self.assertTupleEqual(final_state.vorticity.shape, modal_zxy)
        self.assertTupleEqual(final_state.temperature_variation.shape, modal_zxy)
        self.assertTupleEqual(final_state.log_surface_pressure.shape, (1,) + modal_zxy[1:])
        self.assertTupleEqual(final_state.tracers['specific_humidity'].shape, modal_zxy)

        self.assertTupleEqual(dynamics_predictions.u_wind.shape, nodal_tzxy)
        self.assertTupleEqual(dynamics_predictions.v_wind.shape, nodal_tzxy)
        self.assertTupleEqual(dynamics_predictions.temperature.shape, nodal_tzxy)
        self.assertTupleEqual(dynamics_predictions.specific_humidity.shape, nodal_tzxy)
        self.assertTupleEqual(dynamics_predictions.geopotential.shape, nodal_tzxy)
        self.assertTupleEqual(dynamics_predictions.normalized_surface_pressure.shape, (nodal_tzxy[0],) + nodal_tzxy[2:])

    @pytest.mark.slow
    def test_speedy_model_averages(self):
        from jcm.model import Model
        from jcm.physics.speedy.speedy_coords import get_speedy_coords

        model = Model(
            coords=get_speedy_coords(),
            time_step=30, # to make sure this test stays valid if we ever change the default timestep
        )
        preds = model.run(save_interval=.5/24., total_time=2/24.)

        # Compare only the dynamics fields. Both paths save the same
        # end-of-step state samples:
        #
        # - snapshot mode saves ``state_k`` at outer steps k=1..N
        # - op-split averaged mode sums ``state_k`` for k=1..N inside
        #   the inner scan (Issue #471 P1 follow-up: switched the
        #   averaged accumulator from pre-step to post-step to match
        #   the snapshot path; the legacy one-timestep offset was
        #   tolerable for slow fields but op-split's larger per-step
        #   transient amplified it past rtol=1e-2).
        #
        # Tolerance is loosened to ~1e-3 because the output-boundary
        # ``verify_state`` clamp on non-negative tracers makes
        # ``mean(clamp(x_k)) != clamp(mean(x_k))`` in the rare
        # subgrid where small-amplitude Gibbs ringing dips q below
        # zero. The clamp is cheap (one ``max`` at the modal→nodal
        # output boundary, no extra spectral round-trip) and the
        # discrepancy is bounded by the ringing magnitude (~1e-4
        # kg/kg at T21L8), so the test still catches a *broken*
        # averaging mechanism (which would diverge by orders of
        # magnitude more) while tolerating the clamp gap.
        true_avg_dynamics = jtu.tree_map(
            lambda a: jnp.mean(a, axis=0), preds.dynamics,
        )

        avg_model = Model(
            coords=get_speedy_coords(),
            time_step=30,
        )
        avg_preds = avg_model.run(
            save_interval=2/24.,
            total_time=2/24.,
            output_averages=True,
        )

        jtu.tree_map(
            lambda a1, a2: self.assertTrue(
                jnp.allclose(a1, a2, rtol=1e-3, atol=1e-3),
                msg=f"max abs diff = {float(jnp.max(jnp.abs(a1 - a2)))}",
            ),
            true_avg_dynamics,
            avg_preds.dynamics,
        )

    @pytest.mark.slow
    def test_echam_hybrid_model_output_averages(self):
        """Regression test for #463.

        ``output_averages=True`` on hybrid vertical coordinates used to crash
        in ``compute_diagnostic_state_hybrid`` because the post-processor was
        applied once to the stacked trajectory (with a leading time axis on
        the surface pressure) instead of per-save. The fix moves
        post-processing inside the scan body. Sigma coords masked the bug
        because their ``a_thickness`` is zero so the bad broadcast happened
        to succeed.
        """
        from jcm.model import Model
        from jcm.utils import get_coords
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.echam.echam_terms import echam_physics

        # Smallest hybrid setup that exercises the same code path as the
        # T63L47 + real-terrain configuration that surfaced the bug.
        coords = get_coords(get_echam_levels(47), spectral_truncation=31)
        model = Model(
            coords=coords,
            physics=echam_physics(radiation_scheme="grey", checkpoint_terms=False),
            time_step=3.0,
        )

        save_interval = 1.0 / 24.0  # 1 hour
        total_time = 2.0 / 24.0     # 2 hours -> 2 saves
        preds = model.run(
            save_interval=save_interval,
            total_time=total_time,
            output_averages=True,
        )

        # Predictions should carry a leading time axis matching the number
        # of saves and the spatial dims should match the model grid — i.e.
        # the post-processor ran per-save on a single state, not once on
        # the stacked trajectory.
        n_saves = int(total_time / save_interval)
        self.assertEqual(preds.dynamics.temperature.shape[0], n_saves)
        self.assertEqual(
            preds.dynamics.temperature.shape[1:], coords.nodal_shape,
        )

        # Regression for the post-#463 output_averages NaN bug
        # (https://github.com/climate-analytics-lab/jax-gcm/...): #463 fixed
        # the broadcasting crash but the saved averages were still 100%
        # NaN at T63L47. Root cause was the DiagnosticsCollector seeding
        # ``physics_data_cache`` with zero-state probe output, which a
        # downstream radiation term consumed and propagated 0/0 = NaN
        # through the dynamic tendency. The fix in physics_interface.py
        # bypasses the seeded cache. Spot-check that the averaged
        # dynamics state is finite end-to-end on hybrid coords.
        import numpy as np
        T = np.asarray(preds.dynamics.temperature)
        q = np.asarray(preds.dynamics.specific_humidity)
        u = np.asarray(preds.dynamics.u_wind)
        self.assertFalse(np.isnan(T).any(), "averaged temperature has NaN")
        self.assertFalse(np.isnan(q).any(), "averaged humidity has NaN")
        self.assertFalse(np.isnan(u).any(), "averaged u-wind has NaN")
        # Sanity ranges: with a balanced isothermal IC at 288 K and only
        # 2 hours of integration, the average should stay near IC.
        self.assertGreater(float(T.mean()), 200.0)
        self.assertLess(float(T.mean()), 320.0)

    @pytest.mark.slow
    def test_speedy_model_gradients_isnan(self):
        from jcm.model import Model
        from jcm.utils import ones_like
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        # Create model that goes through one timestep
        
        model = Model(coords=get_speedy_coords())
        state = model._prepare_initial_dycore_state()

        def fn(state):
            _ = model.run(total_time=0) # to set up model fields
            predictions = model.run(initial_state=state, save_interval=(1/48.), total_time=(1/48.))
            return model._final_dycore_state, predictions

        # Calculate gradients
        primals, f_vjp = jax.vjp(fn, state)
        
        input = (ones_like(primals[0]), ones_like(primals[1]))

        df_dstate = f_vjp(input)
        
        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].vorticity)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].divergence)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].temperature_variation)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].log_surface_pressure)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].tracers['specific_humidity'])))
        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].sim_time)))

    @pytest.mark.slow
    def test_speedy_model_gradients_multiple_timesteps_isnan(self):
        from jcm.model import Model
        from jcm.utils import ones_like
        from jcm.physics.speedy.speedy_coords import get_speedy_coords

        model = Model(coords=get_speedy_coords())
        state = model._prepare_initial_dycore_state()

        def fn(state):
            predictions = model.run(initial_state=state, save_interval=(1/48.), total_time=(1/24.))
            return model._final_dycore_state, predictions

        # Calculate gradients
        primals, f_vjp = jax.vjp(fn, state)
        input = (ones_like(primals[0]), ones_like(primals[1]))
        df_dstate = f_vjp(input)

        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].vorticity)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].divergence)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].temperature_variation)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].log_surface_pressure)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].tracers['specific_humidity'])))
        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].sim_time)))

    @pytest.mark.slow
    def test_speedy_model_param_gradients_isnan_vjp(self):
        from jcm.model import Model
        from jcm.terrain import TerrainData
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.forcing import ForcingData
        from jcm.utils import ones_like

        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        coords = get_speedy_coords()
        terrain = TerrainData.from_file(data_dir / 'terrain.nc', coords=coords)
        forcing = ForcingData.from_file(data_dir / 'forcing.nc', coords=coords)

        create_model = lambda params=Parameters.default(): Model(
            coords=coords,
            terrain=terrain,
            physics=speedy_physics(parameters=params),
        )

        fn = lambda params: create_model(params).run(save_interval=1/24., total_time=2./24., forcing=forcing)

        # Calculate gradients using VJP
        params = Parameters.default()
        primal, f_vjp = jax.vjp(fn, params)
        df_dparams = f_vjp(ones_like(primal))

        self.assertFalse(df_dparams[0].isnan().any_true())
        # The gradients must also be CONNECTED: at least one physics
        # parameter carries a nonzero adjoint. A refactor that makes the
        # Parameters struct static/aux data (hiding tunables from autodiff —
        # the anti-pattern CLAUDE.md forbids) yields all-zero gradients that
        # the NaN check alone cannot see.
        param_leaves = [
            leaf for leaf in jax.tree_util.tree_leaves(df_dparams[0])
            if jnp.result_type(leaf) != jax.dtypes.float0
        ]
        self.assertTrue(
            any(bool(jnp.any(leaf != 0.0)) for leaf in param_leaves),
            "all parameter gradients are exactly zero — parameters "
            "disconnected from the model output",
        )

    @pytest.mark.slow
    def test_speedy_model_param_gradients_isnan_jvp(self):
        from jcm.model import Model
        from jcm.terrain import TerrainData
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        from jcm.forcing import ForcingData
        from jcm.utils import ones_like_tangent
        
        from importlib import resources
        data_dir = resources.files('jcm.data.bc.t30.clim')

        coords = get_speedy_coords()
        # need coords to create terrain
        terrain = TerrainData.from_file(data_dir / 'terrain.nc', coords=coords)
        forcing = ForcingData.from_file(data_dir / 'forcing.nc', coords=coords)

        # coords need to be passed to model init
        create_model = lambda params=Parameters.default(): Model(
            coords=coords,
            terrain=terrain,
            physics=speedy_physics(parameters=params),
        )

        model_run_wrapper = lambda params: create_model(params).run(save_interval=1/24., total_time=2./24., forcing=forcing)

        # Calculate gradients using JVP
        params = Parameters.default()
        tangent = ones_like_tangent(params)
        _, jvp_sum = jax.jvp(model_run_wrapper, (params,), (tangent,))
        state = jvp_sum.dynamics
        physics_data = jvp_sum.physics

        # Check dynamics state
        self.assertFalse(jnp.any(jnp.isnan(state.u_wind)))
        self.assertFalse(jnp.any(jnp.isnan(state.v_wind)))
        self.assertFalse(jnp.any(jnp.isnan(state.temperature)))
        self.assertFalse(jnp.any(jnp.isnan(state.specific_humidity)))
        self.assertFalse(jnp.any(jnp.isnan(state.geopotential)))
        self.assertFalse(jnp.any(jnp.isnan(state.normalized_surface_pressure)))
        # Check physics diagnostics dict (composable physics returns a dict
        # rather than a tree_math struct, so .isnan() is no longer callable
        # on the container — walk the leaves instead). The JVP output also
        # contains float0 placeholders for non-differentiable params (bools/
        # ints); skip those since they don't support arithmetic.
        for leaf in jax.tree_util.tree_leaves(physics_data):
            if jnp.result_type(leaf) == jax.dtypes.float0:
                continue
            self.assertFalse(jnp.any(jnp.isnan(leaf)))
        # Connectivity: a unit parameter tangent must perturb the trajectory.
        # All-zero JVP output means the parameters are disconnected from the
        # model (e.g. accidentally marked static) — see the VJP twin above.
        self.assertTrue(
            bool(jnp.any(state.temperature != 0.0))
            or bool(jnp.any(state.u_wind != 0.0)),
            "unit parameter tangent produced an all-zero trajectory tangent",
        )

    # ~70 s: eight one-step T30 SPEEDY integrations (AD + central differences).
    @pytest.mark.slow
    def test_speedy_model_state_gradient_check(self):
        from jcm.model import Model
        from jcm.physics.speedy.speedy_coords import get_speedy_coords

        # Create model that goes through one timestep
        model = Model(coords=get_speedy_coords())
        state = model._prepare_initial_dycore_state()
        _ = model.run(total_time=0)  # to set up model fields

        # check_vjp/check_jvp probe with unit-normal tangents, but the initial
        # condition's spectral coefficients are O(1e-5): a unit perturbation
        # drives the model into its humidity/temperature limiters, where the
        # central difference is identically zero for any eps. Differentiating
        # w.r.t. a small-amplitude perturbation is the same Jacobian at the
        # same point, probed inside the model's linear regime.
        perturbation_scale = 1e-5
        zero_perturbation = jax.tree.map(jnp.zeros_like, state)

        def f(delta):
            perturbed = jax.tree.map(lambda x, d: x + perturbation_scale * d, state, delta)
            predictions = model.run(initial_state=perturbed, save_interval=(1/48.), total_time=(1/48.))
            # ModelPredictions also carries integer diagnostics (cloud-top
            # level, step counters) and a bool flag, and finite differencing
            # those is undefined ("numpy boolean subtract"), so the check
            # covers the differentiable float leaves.
            return [leaf for leaf in jax.tree.leaves(predictions)
                    if jnp.issubdtype(jnp.asarray(leaf).dtype, jnp.floating)]

        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f) 

        # One eps for both checks: f perturbs the state by
        # perturbation_scale * eps, so anything below ~1e-4 lands under a
        # float32 ulp of the state and the difference is rounding noise.
        # At 1e-3 the AD/FD gap is ~0.14 (vjp) / ~0.16 (jvp) and systematic,
        # not noisy — a physics step's humidity and temperature limiters are
        # where-branches the central difference straddles — so rtol keeps ~3x
        # margin for branch flips that move with the platform.
        check_vjp(f, f_vjp, args = (zero_perturbation,), 
                                atol=None, rtol=5e-1, eps=0.001)
        check_jvp(f, f_jvp, args = (zero_perturbation,), 
                                atol=None, rtol=5e-1, eps=0.001)    
    
    @pytest.mark.slow
    def test_speedy_model_default_statistics(self):
        from jcm.data.test.t30.generate_default_stats import run_default_speedy_model, default_stat_vars
        import xarray as xr
        from importlib import resources

        # load test file for comparison
        stats_file = resources.files('jcm.data.test.t30') / 'default_statistics.nc'
        default_stats = xr.open_dataset(stats_file)

        model, predictions = run_default_speedy_model(save_interval=30.)
        pred_ds = predictions.to_xarray()
        pred_ds_monthly = pred_ds.isel(time=-1).mean(dim={'lon', 'lat'}) # global monthly mean, take the last month

        # tolerance in # of standard deviations
        tol = 3

        # check whether zonal averages over the last month are within 2 std deviations of the expected values
        for var in default_stat_vars:
            if var == 'specific_humidity':
                assert default_stats[f'{var}.mean'].attrs['units'] == 'kg kg-1'
                assert pred_ds_monthly[var].attrs['units'] == 'kg kg-1'
            lower = default_stats[f'{var}.mean'] - tol*default_stats[f'{var}.std']
            upper = default_stats[f'{var}.mean'] + tol*default_stats[f'{var}.std']
            assert ((lower <= pred_ds_monthly[var]).all()) & ((pred_ds_monthly[var] <= upper).all())

class TestModelRepr(unittest.TestCase):
    def test_repr_summarizes_model(self):
        # One line naming backend, grid, levels, dt and physics terms (#322).
        from jcm.model import Model
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        model = Model(coords=get_speedy_coords())
        r = repr(model)
        self.assertIn("Model(dycore=DinosaurDycore", r)
        self.assertIn("grid=", r)
        self.assertIn("levels=8", r)
        self.assertIn("physics=[", r)


class TestCalendarDurations(unittest.TestCase):
    """Calendar-string save_interval / total_time."""

    def _build_held_suarez_model(self):
        from jcm.physics.held_suarez.held_suarez_physics import held_suarez_physics
        from jcm.model import Model
        from jcm.terrain import TerrainData
        from jcm.physics.held_suarez.utils import get_held_suarez_coords
        coords = get_held_suarez_coords()
        terrain = TerrainData.from_coords(coords)
        return Model(coords=coords, terrain=terrain, time_step=180,
                     physics=held_suarez_physics())

    def test_run_with_calendar_strings(self):
        """`save_interval='1 month'`, `total_time='2 months'` should yield 2 saves."""
        model = self._build_held_suarez_model()
        predictions = model.run(save_interval='1 month', total_time='2 months')
        # Under the default 365_day calendar, '1 month' is 365/12 days,
        # and total/save = 2 outer steps.
        self.assertEqual(predictions.dynamics.temperature.shape[0], 2)

    def test_xarray_resample_pattern(self):
        """Calendar-aligned aggregation is exposed via xarray's standard
        `resample` API on `to_xarray()` — no special model-level helper.
        Pin the pattern as it's documented in `getting_started.rst`.
        """
        model = self._build_held_suarez_model()
        # 90 days starting 2000-01-01 reaches the end of March, so the
        # trajectory spans 3 calendar months.
        predictions = model.run(save_interval='1 day', total_time='90 days')

        ds = predictions.to_xarray()
        self.assertEqual(ds.sizes['time'], 90)

        monthly = ds.resample(time='1MS').mean()
        self.assertEqual(monthly.sizes['time'], 3)


class TestOperatorSplitPhysics(unittest.TestCase):
    """Operator-split physics (issue #471).

    The op-split path calls physics exactly once per ``dt`` outside the
    IMEX-RK stages and applies the tendency as a forward-Euler add.
    These tests verify the path is wired correctly, exists as a JAX
    pytree, and produces finite atmospheric state in both snapshot and
    averaged modes.
    """

    def _speedy_model(self):
        from jcm.model import Model
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        coords = get_speedy_coords(layers=8, spectral_truncation=21)
        return Model(coords=coords)

    def _echam_hybrid_model(self):
        from jcm.model import Model
        from jcm.utils import get_coords
        from jcm.physics.echam.echam_levels import get_echam_levels
        from jcm.physics.echam.echam_terms import echam_physics
        coords = get_coords(get_echam_levels(47), spectral_truncation=31)
        return Model(
            coords=coords,
            physics=echam_physics(radiation_scheme="grey", checkpoint_terms=False),
            time_step=3.0,
        )

    def test_op_split_snapshot_speedy_finite(self):
        """SPEEDY in op-split snapshot mode produces a finite atmosphere."""
        import numpy as np

        model = self._speedy_model()
        preds = model.run(
            save_interval=1 / 48.0, total_time=1 / 12.0,
            
        )
        T = np.asarray(preds.dynamics.temperature)
        u = np.asarray(preds.dynamics.u_wind)
        q = np.asarray(preds.dynamics.specific_humidity)
        self.assertFalse(np.isnan(T).any(), "op-split snapshot T has NaN")
        self.assertFalse(np.isnan(u).any(), "op-split snapshot u has NaN")
        self.assertFalse(np.isnan(q).any(), "op-split snapshot q has NaN")
        self.assertGreater(float(T.mean()), 200.0)
        self.assertLess(float(T.mean()), 320.0)

    def test_op_split_averaged_speedy_finite(self):
        """SPEEDY in op-split averaged mode produces a finite atmosphere
        and a populated time-averaged diagnostics dict.
        """
        import numpy as np

        model = self._speedy_model()
        preds = model.run(
            save_interval=1 / 48.0, total_time=1 / 12.0,
            output_averages=True,
        )
        T = np.asarray(preds.dynamics.temperature)
        self.assertFalse(np.isnan(T).any(), "op-split averaged T has NaN")
        self.assertGreater(float(T.mean()), 200.0)
        self.assertLess(float(T.mean()), 320.0)
        # In averaged mode the ``physics`` attribute is the time-averaged
        # diagnostics dict (not None as in snapshot mode).
        self.assertIsInstance(preds.physics, dict)
        self.assertGreater(len(preds.physics), 0)

    def test_op_split_averaged_echam_hybrid_finite(self):
        """ECHAM hybrid in op-split averaged mode produces a finite atmosphere.

        This is the configuration that surfaced #470 (output-averages
        NaN). The op-split path threads the radiation cache as an
        explicit pytree carry instead of through the substage-gated
        ``DiagnosticsCollector.physics_data_cache``.
        """
        import numpy as np

        model = self._echam_hybrid_model()
        preds = model.run(
            save_interval=1 / 24.0, total_time=2 / 24.0,
            output_averages=True,
        )
        T = np.asarray(preds.dynamics.temperature)
        q = np.asarray(preds.dynamics.specific_humidity)
        u = np.asarray(preds.dynamics.u_wind)
        self.assertFalse(np.isnan(T).any(), "op-split echam T has NaN")
        self.assertFalse(np.isnan(q).any(), "op-split echam q has NaN")
        self.assertFalse(np.isnan(u).any(), "op-split echam u has NaN")
        self.assertGreater(float(T.mean()), 200.0)
        self.assertLess(float(T.mean()), 320.0)

    def test_op_split_step_is_jax_pure(self):
        """The op-split single-step function is a pure JAX function:
        ``(state, physics_state) -> (state, physics_state)`` and traces
        cleanly under jit + grad.
        """
        from jcm.forcing import default_forcing

        model = self._speedy_model()
        # Set up an initial state via the public API.
        _ = model.run(total_time=0)
        initial_state = model._final_dycore_state

        forcing = default_forcing(model.coords.horizontal)
        step = model._get_op_split_step_fn(forcing)
        initial_physics_state = model._build_initial_physics_carry()

        # Trace and execute one step under jit.
        jit_step = jax.jit(step)
        x1, ps1 = jit_step(initial_state, initial_physics_state)

        # Dynamics state pytree should round-trip.
        self.assertEqual(
            jax.tree_util.tree_structure(x1),
            jax.tree_util.tree_structure(initial_state),
        )
        self.assertFalse(bool(jnp.isnan(x1.temperature_variation).any()))

    def test_op_split_carry_threading(self):
        """``physics_state`` returned by step N is the same pytree shape
        as the input to step N+1 — the contract :class:`jax.lax.scan`
        requires for the carry. Verified by running two steps in
        sequence with the integration carry (post-step shape).
        """
        from jcm.forcing import default_forcing

        model = self._speedy_model()
        _ = model.run(total_time=0)
        initial_state = model._final_dycore_state

        forcing = default_forcing(model.coords.horizontal)
        step = jax.jit(model._get_op_split_step_fn(forcing))
        ps0 = model._build_initial_physics_carry()
        x1, ps1 = step(initial_state, ps0)
        x2, ps2 = step(x1, ps1)

        s0 = jax.tree_util.tree_structure(ps0)
        s1 = jax.tree_util.tree_structure(ps1)
        s2 = jax.tree_util.tree_structure(ps2)
        self.assertEqual(s0, s1)
        self.assertEqual(s1, s2)

    def test_op_split_carry_persists_across_resume(self):
        """``run()`` + ``resume()`` matches a single ``run()`` of the
        combined duration when the cross-step physics carry is
        threaded through (Issue #471 P1).

        Before P1 every call rebuilt the carry from
        ``initial_carry_state``, which reset sub-cycled radiation /
        prior-step TKE etc. at the API seam. With the persisted
        carry the bisected and contiguous trajectories agree to
        numerical roundoff.
        """
        import numpy as np

        model_split = self._speedy_model()
        # 5 + 5 step bisected run.
        _ = model_split.run(
            save_interval=1 / 48.0, total_time=5 / 48.0,
            
        )
        preds_part2 = model_split.resume(
            save_interval=1 / 48.0, total_time=5 / 48.0,
            
        )
        final_bisected = float(
            np.asarray(preds_part2.dynamics.temperature[-1]).mean()
        )

        # Contiguous 10-step run for the same total duration.
        model_one = self._speedy_model()
        preds_one = model_one.run(
            save_interval=1 / 48.0, total_time=10 / 48.0,
            
        )
        final_contiguous = float(
            np.asarray(preds_one.dynamics.temperature[-1]).mean()
        )

        # Tight tolerance — pure jitting roundoff. If the carry isn't
        # being threaded, this would fail by orders of magnitude more.
        self.assertAlmostEqual(
            final_bisected, final_contiguous, places=3,
            msg="bisected run + resume diverged from contiguous run — "
                "is the physics carry threaded across the API seam?",
        )

    def test_op_split_run_resets_carry(self):
        """``run()`` discards any carry left from a previous trajectory.

        Two ``run()`` calls on the same Model object (different initial
        states, default seed) should produce the same answer the first
        time and the second time — i.e. ``run()`` resets
        ``_final_physics_state`` so the second trajectory is not
        contaminated by leftover radiation cache / TKE from the first.
        """
        import numpy as np

        m = self._speedy_model()
        preds_a = m.run(
            save_interval=1 / 48.0, total_time=2 / 48.0,
            
        )
        T_a = float(np.asarray(preds_a.dynamics.temperature[-1]).mean())

        preds_b = m.run(
            save_interval=1 / 48.0, total_time=2 / 48.0,
            
        )
        T_b = float(np.asarray(preds_b.dynamics.temperature[-1]).mean())

        self.assertAlmostEqual(
            T_a, T_b, places=4,
            msg="repeated run() on same Model gave different answers — "
                "stale physics carry not cleared between runs",
        )

    def test_op_split_snapshot_physics_uses_integration_carry(self):
        """Snapshot ``predictions.physics`` is the carry the integration
        actually consumed (Issue #471 P2).

        Earlier revisions threw away the per-step carry and
        recomputed physics inside ``_post_process`` with
        ``prev_physics_data=None``, which silently reported a
        freshly-seeded radiation cache (zero / IC values) on
        non-radiation outer steps because the default
        ``radiation_interval`` is 7200 s and the dycore reuses the
        cached fields between recomputes. This test checks that the
        saved physics dict actually has the populated radiation
        fields the integration was using — not the zero-seeded IC.
        """
        import numpy as np

        model = self._echam_hybrid_model()
        # 30-minute outer save with a 3-second dt and grey radiation —
        # plenty of timesteps for the radiation cache to have evolved
        # well away from its zero-seeded initial value by the first
        # save.
        preds = model.run(
            save_interval=1 / 48.0, total_time=1 / 48.0,
            output_averages=False,
        )

        self.assertIsNotNone(
            preds.physics,
            "snapshot mode must populate predictions.physics from the carry",
        )

        # Walk the physics carry dict for a leaf array we know the
        # grey radiation term writes — any non-zero leaf is sufficient
        # evidence that the saved carry is the integration's, not
        # ``Physics.get_empty_data`` (which would be all zeros).
        leaves = jax.tree_util.tree_leaves(preds.physics)
        nonzero = any(
            bool(np.any(np.asarray(leaf) != 0.0)) for leaf in leaves
        )
        self.assertTrue(
            nonzero,
            "all leaves in saved physics carry are zero — looks like a "
            "freshly-seeded carry was saved instead of the one the "
            "integration consumed",
        )

    def test_run_threads_supplied_initial_physics_state(self):
        """``run(initial_physics_state=...)`` seeds the integration with the
        given carry instead of rebuilding a fresh one.

        A warm start off a donor checkpoint wants the donor's radiation
        sub-cycle cache / prior-step TKE preserved across the run seam rather
        than reset. ``run`` replaces the freshly-built carry with the supplied
        one after ``bootstrap_state`` and before the first ``resume``. We spy
        on ``run_from_state_with_carry`` — the seam ``resume`` delegates to —
        to capture the carry the integration actually receives, and check that
        supplying a stamped carry threads it through while the default rebuilds
        a fresh (unstamped) one.
        """
        import numpy as np

        model = self._speedy_model()

        captured = {}
        original = model.run_from_state_with_carry

        def spy(initial_state, forcing, **kwargs):
            captured["carry"] = kwargs.get("initial_physics_state")
            return original(initial_state, forcing, **kwargs)

        model.run_from_state_with_carry = spy

        # A carry structurally matching what the model builds, with a
        # recognizable stamp added to every floating-point leaf (integer
        # leaves are left alone so the pytree stays valid).
        fresh = model._build_initial_physics_carry()
        stamped = jax.tree_util.tree_map(
            lambda x: x + 0.0123
            if jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating) else x,
            fresh,
        )
        model.run(
            initial_physics_state=stamped,
            save_interval=1 / 48.0, total_time=1 / 48.0,
        )
        threaded = captured["carry"]
        self.assertIsNotNone(threaded, "run did not thread a carry to the seam")
        # The integration receives exactly the carry we supplied.
        for got, want in zip(jax.tree_util.tree_leaves(threaded),
                             jax.tree_util.tree_leaves(stamped)):
            np.testing.assert_allclose(np.asarray(got), np.asarray(want))
        # ... and it is genuinely the stamped carry, not the fresh rebuild.
        differs = any(
            not np.allclose(np.asarray(a), np.asarray(b))
            for a, b in zip(jax.tree_util.tree_leaves(stamped),
                            jax.tree_util.tree_leaves(fresh))
        )
        self.assertTrue(differs, "stamp was a no-op — test would be vacuous")

        # Default: no initial_physics_state -> the freshly-built carry.
        captured.clear()
        model.run(save_interval=1 / 48.0, total_time=1 / 48.0)
        default_carry = captured["carry"]
        for got, want in zip(jax.tree_util.tree_leaves(default_carry),
                             jax.tree_util.tree_leaves(fresh)):
            np.testing.assert_allclose(np.asarray(got), np.asarray(want))


class TestLegacyPathRemoved(unittest.TestCase):
    """Phase 4 of #471: legacy inside-RK physics path is gone.

    Confirms the removed symbols cannot be imported and no production
    code references the dead identifiers. ``DiagnosticsCollector``,
    ``averaged_trajectory_from_step``, and ``get_physical_tendencies``
    are all gone along with the ``use_op_split`` flag.
    """

    def test_legacy_symbols_removed(self):
        """Imports of legacy-path symbols should fail."""
        from jcm import model, physics_interface
        for name in (
            "DiagnosticsCollector",
            "averaged_trajectory_from_step",
            "_get_step_fn_factory",
            "_get_integrate_fn",
        ):
            self.assertFalse(
                hasattr(model, name) or hasattr(getattr(model, "Model", None), name),
                f"jcm.model.{name} should be removed (Phase 4)",
            )
        self.assertFalse(
            hasattr(physics_interface, "get_physical_tendencies"),
            "get_physical_tendencies should be removed (Phase 4)",
        )

    def test_physics_carry_state_alias_exists(self):
        """The :data:`PhysicsCarryState` type alias is still importable."""
        from jcm.physics_interface import PhysicsCarryState
        self.assertIsNotNone(PhysicsCarryState)

    def test_no_grep_legacy_identifiers(self):
        """Repository-level regression: no production code references
        the removed legacy-path identifiers.

        Excludes ``*_test.py`` (this file itself references the names
        in string literals) and ``*.md`` (design docs document the
        deletions).
        """
        import subprocess
        from pathlib import Path

        repo = Path(__file__).resolve().parent.parent
        # Tokens defeat self-match by string concatenation.
        legacy_tokens = [
            "physics" + "_data_" + "cache",
            "use_op" + "_split",
            "Diagnostics" + "Collector",
            "get_physical_" + "tendencies",
            "averaged_trajectory_" + "from_step",
        ]
        pattern = "|".join(rf"\b{t}\b" for t in legacy_tokens)
        out = subprocess.run(
            [
                "grep", "-rEn", pattern,
                "--include=*.py",
                "--exclude=*_test.py",
                "--exclude-dir=__pycache__",
                str(repo / "jcm"),
            ],
            capture_output=True, text=True,
        )
        lines = [ln for ln in out.stdout.splitlines() if ln.strip()]
        self.assertEqual(
            lines, [],
            f"Legacy identifiers must be fully removed; found: {lines}",
        )



class TestParameterBindingAndCompilation(unittest.TestCase):
    """Physics parameters are bound into the executable at trace time.

    ``Model._run_from_state`` takes ``self`` as a static jit argument, so
    every physics term and its parameters are constants inside the
    compiled executable, bound when the physics is first traced. Editing
    a live model's parameters afterwards is not reliably picked up by a
    later run (#735, and the warning ``ModelPredictions`` raises when it
    sees it). The supported way to vary a parameter is therefore to build
    a Model per parameter set — inside one jitted function, so the
    rebuild is a trace-time cost paid once instead of a recompile per
    iteration. These tests pin that pattern, which is what a sensitivity
    sweep or a gradient-based calibration loop is made of.
    """

    def setUp(self):
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        self.coords = get_speedy_coords(layers=8, spectral_truncation=21)

    def _mean_temperature(self, albsea, traces=None):
        """Mean T of a 2-hour SPEEDY forecast at a given sea albedo.

        Built the way a calibration loop must build it: the parameter is
        an argument, the Model is constructed from it here rather than
        mutated afterwards.
        """
        from jcm.model import Model
        from jcm.physics.speedy.params import Parameters
        from jcm.physics.speedy.speedy_terms import speedy_physics

        if traces is not None:
            traces.append(albsea)  # Python side effect: once per trace.
        params = Parameters.default()
        params = params.replace(
            mod_radcon=params.mod_radcon.replace(albsea=albsea))
        model = Model(coords=self.coords,
                      physics=speedy_physics(parameters=params),
                      time_step=30.0)
        preds = model.run(save_interval=1 / 48.0, total_time=1 / 12.0)
        return jnp.mean(preds.dynamics.temperature)

    def test_run_works_inside_an_outer_jit(self):
        """``jax.jit`` around ``model.run`` compiles once and stays exact.

        A regression test for #735: provenance briefly returned a trace
        id from the jitted run and read it with ``int()``, which is a
        concrete value only when ``run`` is called outside any enclosing
        transformation. Wrapping a run in ``jax.jit`` — the whole point
        of a calibration loop, since it turns a per-iteration recompile
        into a per-iteration executable reuse — raised
        ``ConcretizationTypeError``. Nothing on the ``run`` path may
        require a concrete value from the jitted computation.
        """
        traces = []
        jitted = jax.jit(
            lambda albsea: self._mean_temperature(albsea, traces))

        cold = float(jitted(jnp.float32(0.02)))
        bright = float(jitted(jnp.float32(0.9)))

        # One trace for both parameter values: the second call reuses the
        # executable rather than rebuilding and recompiling the model.
        self.assertEqual(len(traces), 1)
        # ...and the parameter still reaches the computation, so the
        # reuse is not the silent no-op an in-place edit would be. A
        # brighter ocean reflects more shortwave; over two hours that is
        # a small but unambiguous change in the global mean.
        self.assertGreater(abs(cold - bright), 1e-4)

    def test_predictions_record_the_parameters_that_ran(self):
        """Provenance survives the loss of the trace id (#732 via #735)."""
        from jcm.model import Model
        from jcm.physics.speedy.params import Parameters
        from jcm.physics.speedy.speedy_terms import speedy_physics

        key = "speedy_vertical_diffusion.params.trvdi"
        params = Parameters.default()
        params = params.replace(
            vertical_diffusion=params.vertical_diffusion.replace(
                trvdi=jnp.array(2.0)))
        model = Model(coords=self.coords,
                      physics=speedy_physics(parameters=params),
                      time_step=30.0)

        preds = model.run(save_interval=1 / 48.0, total_time=1 / 24.0)
        self.assertAlmostEqual(preds.params[key], 2.0, places=6)

    def test_a_traced_run_leaves_no_carry_on_the_model(self):
        """Running an existing model under a jit must not poison it.

        ``run``/``resume`` store the final dycore and physics states so a
        later ``resume`` can continue from them. Under an enclosing
        transformation those are tracers, and keeping them would let a
        value escape its trace: the next ``resume`` threads it into a new
        one and raises ``UnexpectedTracerError`` somewhere far from the
        cause. The run itself is fine, since its results are returned
        rather than read back off the model, so the carry is dropped and
        the next ``resume`` says why.
        """
        from jcm.model import Model
        from jcm.physics.speedy.speedy_terms import speedy_physics

        model = Model(coords=self.coords, physics=speedy_physics(),
                      time_step=30.0)
        model.bootstrap_state(None)
        state = model._final_dycore_state
        kw = dict(save_interval=1 / 48.0, total_time=1 / 48.0)

        value = float(jax.jit(lambda s: jnp.mean(
            model.run(s, **kw).dynamics.temperature))(state))
        self.assertTrue(np.isfinite(value))
        self.assertIsNone(model._final_dycore_state)

        with self.assertRaises(ValueError) as caught:
            model.resume(**kw)
        self.assertIn("run_from_state_with_carry", str(caught.exception))

        # ...and the model is not permanently broken: an explicit state
        # gives it a concrete carry again.
        model.run(state, **kw)
        self.assertTrue(np.isfinite(
            float(jnp.mean(model.resume(**kw).dynamics.temperature))))

    def test_an_ordinary_run_still_carries_into_resume(self):
        """The guard must not disturb the untransformed path."""
        from jcm.model import Model
        from jcm.physics.speedy.speedy_terms import speedy_physics

        model = Model(coords=self.coords, physics=speedy_physics(),
                      time_step=30.0)
        kw = dict(save_interval=1 / 48.0, total_time=1 / 48.0)
        model.run(**kw)
        self.assertIsNotNone(model._final_dycore_state)
        self.assertTrue(np.isfinite(
            float(jnp.mean(model.resume(**kw).dynamics.temperature))))

    @pytest.mark.slow
    def test_gradient_flows_through_a_model_built_inside_the_jit(self):
        """The calibration loop's actual shape: ``jit(grad(loss))``.

        Rebuilding the Model inside the traced function is what makes the
        parameter a traced value rather than a baked-in constant, so the
        gradient with respect to it is the gradient of the forecast, and
        one compilation serves every optimizer iteration.
        """
        traces = []
        target = float(self._mean_temperature(jnp.float32(0.5)))

        def loss(albsea):
            return (self._mean_temperature(albsea, traces) - target) ** 2

        grad_loss = jax.jit(jax.grad(loss))
        g1 = float(grad_loss(jnp.float32(0.2)))
        g2 = float(grad_loss(jnp.float32(0.8)))

        self.assertEqual(len(traces), 1)
        self.assertTrue(jnp.isfinite(g1) and jnp.isfinite(g2))
        # Either side of the target the pull is in opposite directions,
        # so a flat response (the signature of a parameter that never
        # reached the computation) fails here.
        self.assertLess(g1 * g2, 0.0)


class TestModelStateApi(unittest.TestCase):
    """Public initial/resumable-state contract (#755)."""

    @staticmethod
    def _model():
        from jcm.model import Model
        from jcm.physics.held_suarez.held_suarez_physics import (
            held_suarez_physics,
        )
        from jcm.physics.held_suarez.utils import get_held_suarez_coords
        from jcm.terrain import TerrainData

        coords = get_held_suarez_coords()
        return Model(
            coords=coords,
            terrain=TerrainData.from_coords(coords),
            time_step=180,
            physics=held_suarez_physics(),
        )

    def test_fresh_builders_are_pure_and_bootstrap_returns_installed_pair(self):
        model = self._model()

        state = model.initial_state(sim_time=123.0)
        carry = model.initial_physics_carry()
        self.assertIsNone(model.dycore_state)
        self.assertIsNone(model.physics_carry)
        self.assertAlmostEqual(float(model.dycore.sim_time(state)), 123.0)
        self.assertTrue(jax.tree_util.tree_leaves(carry))

        bootstrapped_state, bootstrapped_carry = model.bootstrap_state()
        self.assertIs(bootstrapped_state, model.dycore_state)
        self.assertIs(bootstrapped_carry, model.physics_carry)

    def test_resumable_properties_are_read_only_and_restore_is_atomic(self):
        model = self._model()
        state, carry = model.bootstrap_state()
        replacement_state = model.dycore.with_sim_time(
            state, jnp.asarray(4321.0),
        )
        replacement_carry = jax.tree.map(
            lambda value: jnp.ones_like(value), carry,
        )

        with self.assertRaises(AttributeError):
            model.dycore_state = replacement_state
        with self.assertRaises(AttributeError):
            model.physics_carry = replacement_carry

        model.restore_state(replacement_state, replacement_carry)
        self.assertIs(model.dycore_state, replacement_state)
        self.assertIs(model.physics_carry, replacement_carry)
        self.assertEqual(float(model.dycore.sim_time(model.dycore_state)),
                         4321.0)

        with self.assertRaisesRegex(ValueError, "requires both"):
            model.restore_state(replacement_state, None)

    def test_restore_rejects_tracers_instead_of_leaking_them(self):
        model = self._model()
        _, carry = model.bootstrap_state()

        def attempt_restore(traced_state):
            model.restore_state(traced_state, carry)
            return traced_state

        with self.assertRaisesRegex(ValueError, "cannot retain JAX tracers"):
            jax.make_jaxpr(attempt_restore)(jnp.asarray(1.0))


class TestModelLogging(unittest.TestCase):
    """jcm is a library: it emits log records and configures nothing.

    Level and handler policy belong to whoever assembles the process, so
    importing jcm or building a Model must leave the host's logging exactly
    as it found it. The CLI is the one place that does configure — Hydra's
    ``job_logging`` plus ``runners._apply_log_level`` — because there jcm
    *is* the application.

    Constructing a Model used to set a level on the ``jcm`` logger
    (previously ``CRITICAL`` on the ROOT logger), which reconfigured logging
    for the host as a side effect and, in the test suite, leaked between
    tests as an unreproducible xdist failure (#815).
    """

    def setUp(self):
        self._jcm_level = logging.getLogger("jcm").level
        self._root_level = logging.getLogger().level

    def tearDown(self):
        logging.getLogger("jcm").setLevel(self._jcm_level)
        logging.getLogger().setLevel(self._root_level)

    def _model(self, **kwargs):
        from jcm.model import Model
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        return Model(coords=get_speedy_coords(layers=8,
                                              spectral_truncation=21),
                     time_step=30.0, **kwargs)

    def test_construction_leaves_logging_exactly_as_it_found_it(self):
        """Neither logger the library could reach for is touched.

        Handlers as well as levels, on the ``jcm`` logger as well as root:
        attaching jcm's own handler to the ``jcm`` logger is a live option
        (see #817's discussion), and doing it from ``Model.__init__`` would
        put the library back to configuring the host's logging.
        """
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("jcm").setLevel(logging.ERROR)
        before = {
            name: (lg.level, lg.handlers[:], lg.propagate)
            for name, lg in (("", logging.getLogger()),
                             ("jcm", logging.getLogger("jcm")))
        }

        self._model()

        for name, (level, handlers, propagate) in before.items():
            lg = logging.getLogger(name) if name else logging.getLogger()
            with self.subTest(logger=name or "root"):
                self.assertEqual(lg.level, level)
                self.assertEqual(lg.handlers, handlers)
                self.assertEqual(lg.propagate, propagate)

    def test_an_applications_silence_is_respected(self):
        """The counterpart of the old behaviour, stated deliberately.

        #735 wanted jcm's warnings audible even from an application that had
        silenced its root logger. That is the library overriding a choice the
        host made, so it is gone: a caller who silences logging gets silence.
        Findings that must survive it belong in the run's provenance, which
        is a file rather than a stream — see ``jcm.provenance``.
        """
        logging.getLogger().setLevel(logging.CRITICAL)
        self._model()
        self.assertFalse(
            logging.getLogger("jcm.predictions").isEnabledFor(
                logging.WARNING))

    def test_importing_jcm_installs_no_handler(self):
        """A library that calls ``basicConfig`` decides the host's format.

        It was also pointless for the CLI, which is the only place it could
        have applied: Hydra's ``job_logging`` runs ``dictConfig`` with a
        ``root:`` section, which replaces root's handlers outright.
        """
        import os
        import subprocess
        import sys

        probe = (
            "import logging, sys\n"
            "before = list(logging.getLogger().handlers)\n"
            "import jcm\n"
            "after = list(logging.getLogger().handlers)\n"
            "sys.stdout.write(repr(before == after))\n"
        )
        # A bounded timeout and an explicit returncode: importing jcm pulls
        # in JAX, which on a GPU host without JAX_PLATFORMS=cpu can block,
        # and an unbounded child would hang the suite rather than fail it.
        out = subprocess.run([sys.executable, "-c", probe],
                             capture_output=True, text=True, timeout=300,
                             env={**os.environ, "JAX_PLATFORMS": "cpu"})
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertEqual(out.stdout.strip(), "True", out.stderr[-2000:])

    def test_no_jcm_module_logs_through_the_root_logger(self):
        """Records from jcm must be addressable as a group.

        The module-level ``logging.warning(...)`` helpers, ``logging.root``
        and the ``from logging import warning`` aliases all emit on the ROOT
        logger, so a message sent that way sits outside the ``jcm``
        hierarchy — beyond the reach of ``run.log_level``, and of anything a
        host application sets for ``jcm`` specifically. Whether jcm's records
        can be addressed as a group is the whole point of the naming
        convention.

        Matched on the parsed AST rather than the source text, so a call
        nested in an expression or reached through an alias counts, and a
        usage example in a docstring does not.
        """
        import ast
        import pathlib

        root = pathlib.Path(__file__).resolve().parent
        # Names ``logging`` exports that emit on the root logger.
        emitters = {"debug", "info", "warning", "warn", "error", "critical",
                    "exception", "log"}
        offenders = []
        for path in sorted(root.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            # Whatever this module calls the logging package (``import
            # logging as log``), and any emitter pulled into its namespace
            # directly (``from logging import warning``).
            modules, aliased = set(), set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        root_name = alias.name.split(".")[0]
                        if root_name != "logging":
                            continue
                        # ``import logging.config`` binds the bare name
                        # ``logging`` too; ``... as c`` binds only ``c``.
                        modules.add(alias.asname or root_name)
                elif (isinstance(node, ast.ImportFrom)
                      and node.module == "logging"):
                    aliased |= {alias.asname or alias.name
                                for alias in node.names
                                if alias.name in emitters}

            def _is_logging_module(node):
                # ``logging`` itself, or ``logging.root`` — both emit on root.
                if isinstance(node, ast.Name):
                    return node.id in modules
                return (isinstance(node, ast.Attribute)
                        and node.attr == "root"
                        and isinstance(node.value, ast.Name)
                        and node.value.id in modules)

            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                hit = (
                    (isinstance(func, ast.Attribute)
                     and func.attr in emitters
                     and _is_logging_module(func.value))
                    or (isinstance(func, ast.Name) and func.id in aliased))
                if hit:
                    offenders.append(
                        f"{path.relative_to(root.parent)}:{node.lineno}")
        self.assertEqual(
            offenders, [],
            "these emit on the root logger, where ``log_level`` cannot reach "
            "them; use a module-level ``logger = "
            "logging.getLogger(__name__)`` instead. A message deliberately "
            "aimed at the root logger — and so deliberately outside the "
            "knob — must say so explicitly via ``logging.getLogger()``.")


class TestObserversUnderJit(unittest.TestCase):
    """Observers under an enclosing ``jax.jit`` (#735).

    The sampling itself is pure JAX and differentiates with respect to the
    state; what is not traceable is *building* the sampling tables, which
    happens on the host and needs the window's absolute start time as a
    number. That is normally read from the initial state's ``sim_time``,
    which is a tracer when a caller feeds a per-sample initial state
    through their own jit — the calibration shape. The caller passes
    ``observer_t0_days`` instead, and gets told to when they have not.
    """

    def setUp(self):
        from jcm.observers import TrackObserver
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        self.coords = get_speedy_coords(layers=8, spectral_truncation=21)
        self.observer = TrackObserver.stations(
            latitudes=[0.0, 45.0], longitudes=[0.0, 180.0],
            pressures=[85000.0, 85000.0], variables=("temperature",),
            name="stations")

    def _model(self):
        from jcm.model import Model
        from jcm.physics.speedy.speedy_terms import speedy_physics
        return Model(coords=self.coords, physics=speedy_physics(),
                     time_step=30.0, observers=[self.observer])

    def _seed_state(self):
        model = self._model()
        model.bootstrap_state(None)
        return model, model._final_dycore_state

    def test_traced_initial_state_asks_for_the_window_start(self):
        """The error names the argument to pass, not the tracer it met."""
        model, state = self._seed_state()

        def sample(state):
            preds = model.run(state, save_interval=1 / 48.0,
                              total_time=1 / 48.0)
            return jnp.nanmean(preds.observations[0]["temperature"])

        with self.assertRaises(ValueError) as caught:
            jax.jit(sample)(state)
        self.assertIn("observer_t0_days", str(caught.exception))

    def test_an_explicit_window_start_makes_the_run_jittable(self):
        model, state = self._seed_state()
        t0 = model._observer_window_start(state)
        traces = []

        def sample(state):
            traces.append(state)
            preds = model.run(state, save_interval=1 / 48.0,
                              total_time=1 / 48.0, observer_t0_days=t0)
            return jnp.nanmean(preds.observations[0]["temperature"])

        jitted = jax.jit(sample)
        first = float(jitted(state))
        second = float(jitted(state))

        self.assertEqual(len(traces), 1)
        self.assertTrue(np.isfinite(first))
        self.assertEqual(first, second)
        # A plausible 850 hPa temperature, i.e. the observer really sampled
        # the atmosphere rather than returning the NaN of a masked step.
        self.assertGreater(first, 200.0)
        self.assertLess(first, 320.0)

    def test_run_from_state_also_takes_the_window_start(self):
        """The stateless API must accept what the error tells callers to pass.

        ``run_from_state`` reaches the same host-side table build, so an
        error naming an argument it did not accept would send callers to
        the lower-level carry API for no reason.
        """
        from jcm.forcing import default_forcing

        model, state = self._seed_state()
        _, preds = model.run_from_state(
            state, default_forcing(self.coords.horizontal),
            save_interval=1 / 48.0, total_time=1 / 48.0,
            observer_t0_days=model._observer_window_start(state))
        self.assertTrue(np.isfinite(
            float(jnp.nanmean(preds.observations[0]["temperature"]))))

    def test_a_traced_window_start_points_at_prepared_tables(self):
        """A per-sample window start cannot be a traced value either.

        Passing it as a traced argument is the natural next thing to try
        after the traced-initial-state error, and the tables it feeds are
        host numpy, so it cannot work. Marking it static in the caller's
        jit would, but at one compilation per window — which is the cost
        the caller was trying to avoid. So the error names the way out
        that actually reuses a compilation.
        """
        model, state = self._seed_state()
        t0 = model._observer_window_start(state)

        with self.assertRaises(ValueError) as caught:
            jax.jit(lambda s, t: model.run(
                s, save_interval=1 / 48.0, total_time=1 / 48.0,
                observer_t0_days=t))(state, t0)
        self.assertIn("observer_xs", str(caught.exception))
        self.assertIn("prepare_observers", str(caught.exception))

    def test_prepared_tables_reuse_one_compilation_across_windows(self):
        """Different windows, one compilation — the point of the argument.

        The tables are a dynamic argument of the compiled run, so windows
        that differ only in sampling geometry share an executable. Building
        them inside the run instead needs a concrete start time, which as a
        static jit argument would compile once per window.
        """
        model, state = self._seed_state()
        t0 = model._observer_window_start(state)
        kw = dict(save_interval=1 / 48.0, total_time=1 / 48.0)
        traces = []

        def sampled(state, xs):
            traces.append(xs)
            preds = model.run(state, observer_xs=xs, **kw)
            return jnp.nanmean(preds.observations[0]["temperature"])

        jitted = jax.jit(sampled)
        values = [float(jitted(state, model.prepare_observers(day, **kw)))
                  for day in (t0, t0 + 30.0, t0 + 400.0)]

        self.assertEqual(len(traces), 1)
        for value in values:
            self.assertTrue(np.isfinite(value))
            self.assertGreater(value, 200.0)
            self.assertLess(value, 320.0)

    def test_tables_built_for_another_window_are_rejected(self):
        """A length mismatch is caught here, not deep inside the scan."""
        model, state = self._seed_state()
        t0 = model._observer_window_start(state)
        mismatched = model.prepare_observers(
            t0, save_interval=1 / 48.0, total_time=1 / 12.0)

        with self.assertRaises(ValueError) as caught:
            model.run(state, save_interval=1 / 48.0, total_time=1 / 48.0,
                      observer_xs=mismatched)
        self.assertIn("steps", str(caught.exception))

    def test_prepared_tables_still_produce_a_dated_dataset(self):
        """Prepared tables must not cost the observation output its time axis.

        ``observer_xs`` skips the host-side build, which is where the
        window start used to be resolved, so a run given tables and no
        ``observer_t0_days`` had nothing to date its samples by and
        ``observation_datasets()`` failed on the missing value. The start
        time is recovered from the state whenever that is possible, which
        is every use outside a transformation.
        """
        model, state = self._seed_state()
        kw = dict(save_interval=1 / 48.0, total_time=1 / 48.0)
        t0 = model._observer_window_start(state)

        preds = model.run(state, observer_xs=model.prepare_observers(t0, **kw),
                          **kw)
        self.assertEqual(preds._obs_t0_days, t0)
        datasets = model.run(state, **kw).observation_datasets()
        with_tables = preds.observation_datasets()
        self.assertIn("stations", with_tables)
        np.testing.assert_array_equal(
            with_tables["stations"].time.values,
            datasets["stations"].time.values)

    def test_undatable_samples_say_so_rather_than_failing_on_none(self):
        """Name the argument that records a start time, when none exists.

        Under tracing there is nothing to recover it from.
        """
        from jcm.predictions import ModelPredictions

        model, state = self._seed_state()
        preds = ModelPredictions(
            None, None, None, observations=({"temperature": jnp.zeros((2, 1))},),
            observers=tuple(model.observers), obs_t0_days=None,
            obs_dt_seconds=1800.0)

        with self.assertRaises(ValueError) as caught:
            preds.observation_datasets()
        self.assertIn("observer_t0_days", str(caught.exception))

    def test_the_recovered_window_start_is_unchanged_by_the_argument(self):
        """Passing what the model would have read gives the same samples."""
        model, state = self._seed_state()
        implicit = model.run(state, save_interval=1 / 48.0,
                             total_time=1 / 48.0)
        explicit = model.run(
            state, save_interval=1 / 48.0, total_time=1 / 48.0,
            observer_t0_days=model._observer_window_start(state))
        np.testing.assert_allclose(
            np.asarray(implicit.observations[0]["temperature"]),
            np.asarray(explicit.observations[0]["temperature"]))


class TestReleaseMatrixStatistics(unittest.TestCase):
    """The supported-matrix climatology regression.

    Its own TestCase rather than a method on ``TestModelUnit``: it shares
    nothing with that class's ``setUp``, which imports SPEEDY symbols for
    other tests, and it is a different kind of test — an integration run per
    supported configuration rather than a unit check on ``Model``.
    """

    @pytest.mark.slow
    def test_release_matrix_default_statistics(self):
        """Every supported-matrix member still produces what it produced.

        One sub-test per member of ``tools/release_validation/matrix.yaml``,
        each built through that member's validated preset, resumed from its
        init state on the data mirror and integrated for the stats window,
        asserting every stored variable's global mean falls inside its band.

        These are **regression** bands, not a climatology: they are drawn
        from a short window following a short spin-up, so a failure means
        "something changed", not "the physics is wrong". See
        ``jcm.data.test.release_matrix.generate_stats``, which regenerates a
        member's band file and its init state together — the bands describe
        the window that follows that exact state, so the two are only
        meaningful as a pair.

        Too heavy for CPU CI, so gated behind
        ``JCM_RUN_GPU_INTEGRATION_TESTS=1``.
        """
        import importlib.util
        import os
        import tempfile

        import xarray as xr

        if os.environ.get("JCM_RUN_GPU_INTEGRATION_TESTS") != "1":
            pytest.skip(
                "set JCM_RUN_GPU_INTEGRATION_TESTS=1 to run; the matrix "
                "members are too heavy for CPU CI",
            )

        # The workers inherit the environment as the operator set it;
        # ``_run_worker`` adds only ``XLA_PYTHON_CLIENT_PREALLOCATE=false``.
        # This process needs no device of its own — it reads band files and
        # spawns one worker per member — and the session-wide preallocation
        # guard in the root ``conftest.py`` is what keeps it from holding the
        # card anyway. That guard has to live there because merely *importing*
        # this module initialises a CUDA backend: measured at 61,214 MiB of an
        # 80 GB A100 before any test body runs, against 428 MiB with the guard
        # in place. No pin applied from inside a test body can be early
        # enough, which is why this does not try.
        worker_env = dict(os.environ)

        from jcm.data.test.release_matrix.generate_stats import (
            band_path,
            members,
            resolve_state,
            stats_window_global_mean_isolated,
        )

        #: Optional extras a member needs before it can even be composed.
        extras = {"echam-jam-t63-l47": "mam4_jax",
                  "echam-jam-t63-l95": "mam4_jax"}

        checked = 0
        not_local = []
        for member in members():
            bands_file = band_path(member)
            if not bands_file.exists():
                # A member whose fixture has not been generated yet is passed
                # over rather than failed — the set is filled in member by
                # member, each needing its own GPU run. The whole test skips
                # if that leaves nothing checked, so an empty fixture set can
                # never be mistaken for a pass.
                continue
            extra = extras.get(member)
            if extra and importlib.util.find_spec(extra) is None:
                continue
            with self.subTest(member=member):
                bands = xr.open_dataset(bands_file)
                # The band file names the variables it carries; deriving the
                # list here instead would let a regenerated fixture and the
                # assertion drift apart silently.
                stat_vars = sorted(
                    v[: -len(".mean")] for v in bands.data_vars
                    if v.endswith(".mean")
                )
                self.assertTrue(stat_vars, f"{bands_file} carries no bands")

                # The state path comes from the band file, digest and all,
                # so these bands are always checked against the state they
                # were generated against. A mirror fetch that fails raises —
                # its message names the prefetch command — rather than being
                # swallowed into a pass. ``JCM_FIXTURE_STATE_DIR`` points
                # this at locally generated states instead, and returns None
                # for a member absent from that directory, since states are
                # generated one member at a time and the point of the
                # override is to validate the pair before publishing it.
                # A member whose state is deliberately not published yet is
                # a declared gap, not a pass: skip it, with the reason the
                # band file itself carries. Every other member's 404 stays a
                # hard failure — a missing state must never read as success.
                if bands.attrs.get("hosted_state") == "pending":
                    self.skipTest(
                        f"{member}: init state not published — "
                        f"{bands.attrs.get('hosted_state_reason', 'no reason recorded')}")
                state = resolve_state(bands.attrs["init_state"])
                if state is None:
                    not_local.append(member)
                    continue
                # Each member's window runs in its own interpreter. JAX never
                # returns pool memory, so walking the whole matrix in one
                # process starves whichever member comes last — reproducibly
                # the T63 L95 JAM one, several times the footprint of the
                # rest, which failed here with RESOURCE_EXHAUSTED while the
                # six lighter members ahead of it passed.
                with tempfile.TemporaryDirectory() as tmp:
                    pred = stats_window_global_mean_isolated(
                        member, state, tmp, env=worker_env)

                tol = 3  # tolerance in standard deviations
                # #744's degenerate-band fallback, scoped to ``std`` being
                # *exactly* zero: there the band carries no information at all
                # — the specific/relative-humidity tail is physically
                # negligible (~1e-24…1e-37 kg kg-1) and a hybrid grid's upper
                # ``pressure_full`` levels are pure a-coefficient constants —
                # so a relative+absolute tolerance stands in for it. It must
                # not be extended to merely *small* ``std``: doing that is a
                # far worse bug than the one it would fix, handing seven
                # ``pressure_full`` levels of these fixtures half-widths of
                # 400-2800 Pa, wide enough to pass a gross pressure error.
                rtol, atol = 0.25, 1e-8
                # A strictly positive ``std`` can still be finer than float32
                # resolves, and then the band is narrower than the arithmetic
                # underneath it: ``pressure_full`` near the pure-a levels
                # stores 4.9e-4 Pa at 7405.9 Pa — 0.55 of a ULP — giving a
                # three-ULP band that an independent run, in another process
                # on identical code, sat exactly three ULP from. That is a
                # pass by equality, one ULP from red. Floor such a band at a
                # few ULP of its own magnitude instead: 1e-6 (~8 ULP) lifts
                # that level to 7.4e-3 Pa and leaves every informative band
                # untouched (it widens nothing else in these fixtures).
                ulp_floor = 1e-6
                # Second floor, on the half-width: ``<var>.noise`` is the
                # measured peak-to-peak spread of this same window across
                # independent repeats in separate processes, i.e. what the
                # band must absorb with no physics having changed. Applied to
                # the half-width rather than folded into ``std`` so it can
                # only widen a band — folding it in would narrow the
                # degenerate levels, whose fallback is deliberately far wider
                # than their reproducibility.
                noise_tol = 3
                for var in stat_vars:
                    mean = bands[f"{var}.mean"]
                    std = bands[f"{var}.std"]
                    self.assertIn(
                        f"{var}.noise", bands,
                        f"{var}.noise missing from {bands_file} — the fixture "
                        "predates the reproducibility floor; regenerate it",
                    )
                    half_width = xr.where(
                        std > 0,
                        np.maximum(tol * std, ulp_floor * abs(mean)),
                        rtol * abs(mean) + atol,
                    )
                    half_width = np.maximum(
                        half_width, noise_tol * bands[f"{var}.noise"])
                    lower, upper = mean - half_width, mean + half_width
                    assert ((lower <= pred[var]).all()) & (
                        (pred[var] <= upper).all()
                    ), (
                        f"{member}: {var} fell outside its band (±3σ, floored "
                        "at 3× the measured run-to-run reproducibility and at "
                        "a relative+absolute tolerance where σ is below "
                        "float32 resolution). Regenerate this member's band "
                        "file AND its init state together with "
                        "jcm.data.test.release_matrix.generate_stats.generate"
                        f"({member!r}) if the deviation is intentional."
                    )
                checked += 1
        if not_local:
            print(
                f"\nJCM_FIXTURE_STATE_DIR held no state for: "
                f"{', '.join(not_local)} (checked {checked} member(s))")
        if not checked:
            pytest.skip(
                "no matrix member had a band file, its optional extras and "
                "its init state available",
            )



