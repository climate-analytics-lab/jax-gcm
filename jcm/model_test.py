import logging
import unittest
import jax
import jax.tree_util as jtu
import jax.numpy as jnp
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
        import logging
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
            log_level=logging.CRITICAL,
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

    @pytest.mark.skip(reason="finite differencing produces nans")
    def test_speedy_model_state_gradient_check(self):
        from jcm.model import Model
        from jcm.physics.speedy.speedy_coords import get_speedy_coords

        # Create model that goes through one timestep
        model = Model(coords=get_speedy_coords())
        state = model._prepare_initial_dycore_state()

        def f(state_f):
            _ = model.run(total_time=0) # to set up model fields
            predictions = model.run(initial_state=state_f, save_interval=(1/48.), total_time=(1/48.))
            return model._final_dycore_state, predictions
        
        # Calculate gradient
        f_jvp = functools.partial(jax.jvp, f)
        f_vjp = functools.partial(jax.vjp, f) 

        check_vjp(f, f_vjp, args = (state,), 
                                atol=None, rtol=1, eps=0.00001)
        check_jvp(f, f_jvp, args = (state,), 
                                atol=None, rtol=1, eps=0.001)    
    
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
            lower = default_stats[f'{var}.mean'] - tol*default_stats[f'{var}.std']
            upper = default_stats[f'{var}.mean'] + tol*default_stats[f'{var}.std']
            assert ((lower <= pred_ds_monthly[var]).all()) & ((pred_ds_monthly[var] <= upper).all())

    @pytest.mark.slow
    def test_echam_model_default_statistics(self):
        """Run default ``echam_physics()`` on T63L47 + real terrain from a
        spun-up state and assert each variable's global mean falls in its
        stored climatology band (mean ± 3·std).

        Mirrors :meth:`test_speedy_model_default_statistics` but uses the
        production wiring the user's ``debug/echam-2m-micro-stability``
        work runs against: T63L47 hybrid coords, real ECHAM terrain +
        forcing, ``echam_physics(grey) + UpperSponge``, started from a
        saved 5-day spun-up state, integrated 5 more days with daily
        averages.

        T63L47 is too heavy for CPU CI; the test is gated behind
        ``JCM_RUN_GPU_INTEGRATION_TESTS=1`` and skipped otherwise.

        Stats and the spun-up state live in
        ``jcm/data/test/echam_t63l47/`` and are regenerated by
        ``jcm.data.test.echam_t63l47.generate_default_stats.generate()``
        when the physics changes intentionally.
        """
        import os
        from pathlib import Path

        import xarray as xr

        if os.environ.get("JCM_RUN_GPU_INTEGRATION_TESTS") != "1":
            pytest.skip(
                "set JCM_RUN_GPU_INTEGRATION_TESTS=1 to run; T63L47 is "
                "too heavy for CPU CI",
            )

        bc_dir = Path("jcm/data/bc/t63")
        stats_dir = Path("jcm/data/test/echam_t63l47")
        for fname in ("terrain.nc", "forcing.nc"):
            if not (bc_dir / fname).exists():
                pytest.skip(
                    f"{bc_dir / fname} missing; run "
                    "utils/convert_echam_bc.py to generate it",
                )
        if not (stats_dir / "spinup_state.nc").exists():
            pytest.skip(
                f"{stats_dir / 'spinup_state.nc'} missing; run "
                "jcm.data.test.echam_t63l47.generate_default_stats.generate() "
                "on a GPU to create it",
            )

        from jcm.data.test.echam_t63l47.generate_default_stats import (
            default_echam_t63l47_stat_vars,
            run_default_echam_t63l47_model,
        )

        default_stats = xr.open_dataset(stats_dir / "default_statistics.nc")

        # Resume from the saved spun-up state and integrate 5 more days
        # of daily *snapshots*. ``output_averages=True`` on hybrid coords
        # trips a shape-broadcast bug in
        # ``compute_diagnostic_state_hybrid`` that the existing T63L47
        # tests don't exercise; the mean of 5 daily snapshots is a close
        # approximation of the true 5-day mean for the slow-varying
        # global statistics the assertion uses.
        _, predictions = run_default_echam_t63l47_model(
            save_interval=1.0, total_time=5.0,
        )
        pred_ds = predictions.to_xarray()
        pred_ds_mean = pred_ds.mean(dim={"time", "lon", "lat"})

        tol = 3  # tolerance in standard deviations
        # Degenerate-band guard. At the top of the atmosphere several stored
        # bands have a temporal std that underflows to *exactly* 0.0 in float32:
        # the specific/relative-humidity tail is physically negligible
        # (~1e-24…1e-37 kg kg⁻¹) and the upper ``pressure_full`` levels are the
        # pure a-coefficient constants. A ±3σ band there collapses to a single
        # point, which cannot absorb the few-percent cross-process
        # nondeterminism XLA autotuning introduces at that underflow tail —
        # runs are bit-identical *in-process* but drift across the
        # generation↔test process boundary. Where std==0, fall back to a
        # relative+absolute tolerance; every physically meaningful level has
        # std>0 and keeps the strict ±3σ test unchanged. See #744.
        rtol, atol = 0.25, 1e-8
        for var in default_echam_t63l47_stat_vars:
            mean = default_stats[f"{var}.mean"]
            std = default_stats[f"{var}.std"]
            half_width = xr.where(std > 0, tol * std, rtol * abs(mean) + atol)
            lower = mean - half_width
            upper = mean + half_width
            assert ((lower <= pred_ds_mean[var]).all()) & (
                (pred_ds_mean[var] <= upper).all()
            ), (
                f"{var} fell outside the climatology band (±3σ, with a "
                "relative+absolute floor where σ underflowed to 0); "
                "regenerate jcm/data/test/echam_t63l47/default_statistics.nc "
                "(and spinup_state.nc) if the deviation is intentional."
            )


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


class TestModelLogging(unittest.TestCase):
    """Warnings jcm raises about a run have to reach the user.

    The default was ``logging.CRITICAL`` applied to the ROOT logger, which
    silenced every jcm warning — including the one saying an in-place
    parameter change never reached the computation (#735) — and
    reconfigured logging for the host application as a side effect. Both
    halves of that are tested here: the default is audible, and it is
    scoped to the ``jcm`` hierarchy.
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

    def test_warnings_are_audible_even_from_a_quiet_application(self):
        # An application that has silenced the root logger still hears
        # jcm's warnings about its own results, because the level is set
        # on the jcm logger: level lookup stops at the first logger with
        # one set, and propagation to handlers does not consult the root
        # logger's level.
        logging.getLogger().setLevel(logging.CRITICAL)
        self._model()
        self.assertTrue(
            logging.getLogger("jcm.predictions").isEnabledFor(
                logging.WARNING))

    def test_construction_leaves_the_root_logger_alone(self):
        logging.getLogger().setLevel(logging.DEBUG)
        self._model()
        self.assertEqual(logging.getLogger().level, logging.DEBUG)
        self.assertEqual(logging.getLogger("jcm").level, logging.WARNING)

    def test_log_level_argument_still_quietens_jcm(self):
        self._model(log_level=logging.CRITICAL)
        self.assertFalse(
            logging.getLogger("jcm.predictions").isEnabledFor(
                logging.WARNING))
