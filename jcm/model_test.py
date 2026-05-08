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
        final_state, dynamics_predictions = model._final_modal_state, predictions.dynamics

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
        final_state, dynamics_predictions = model._final_modal_state, predictions.dynamics

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

        # Compare only the dynamics fields. Manual mean over saved snapshots
        # uses end-of-step states; the output_averages path uses the inner
        # x_sum which is built from BEFORE-step states (note the
        # `x_sum += x` placement in averaged_trajectory_from_step). For the
        # dynamics state these two windows agree to <1e-4 over a short run;
        # for the physics diagnostics dict (cloud cover, surface fluxes,
        # land surface temperature) the offset can produce O(1) differences
        # that are not a meaningful regression test.
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

        # Tolerance: the manual save path captures end-of-step states (1..N)
        # while output_averages sums BEFORE-step states (0..N-1) — they
        # average windows offset by one timestep. rtol=1e-2 lets this test
        # catch a broken averaging mechanism without flagging the legitimate
        # one-timestep offset (worst-case ~0.3% on q over a 2-hour run).
        jtu.tree_map(
            lambda a1, a2: self.assertTrue(
                jnp.allclose(a1, a2, rtol=1e-2, atol=1e-2),
                msg=f"max abs diff = {float(jnp.max(jnp.abs(a1 - a2)))}",
            ),
            true_avg_dynamics,
            avg_preds.dynamics,
        )

    @pytest.mark.slow
    def test_speedy_model_gradients_isnan(self):
        from jcm.model import Model
        from jcm.utils import ones_like
        from jcm.physics.speedy.speedy_coords import get_speedy_coords
        # Create model that goes through one timestep
        
        model = Model(coords=get_speedy_coords())
        state = model._prepare_initial_modal_state()

        def fn(state):
            _ = model.run(total_time=0) # to set up model fields
            predictions = model.run(initial_state=state, save_interval=(1/48.), total_time=(1/48.))
            return model._final_modal_state, predictions

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
        state = model._prepare_initial_modal_state()

        def fn(state):
            predictions = model.run(initial_state=state, save_interval=(1/48.), total_time=(1/24.))
            return model._final_modal_state, predictions

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

    @pytest.mark.skip(reason="finite differencing produces nans")
    def test_speedy_model_state_gradient_check(self):
        from jcm.model import Model
        from jcm.physics.speedy.speedy_coords import get_speedy_coords

        # Create model that goes through one timestep
        model = Model(coords=get_speedy_coords())
        state = model._prepare_initial_modal_state()

        def f(state_f):
            _ = model.run(total_time=0) # to set up model fields
            predictions = model.run(initial_state=state_f, save_interval=(1/48.), total_time=(1/48.))
            return model._final_modal_state, predictions
        
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
        for var in default_echam_t63l47_stat_vars:
            lower = default_stats[f"{var}.mean"] - tol * default_stats[f"{var}.std"]
            upper = default_stats[f"{var}.mean"] + tol * default_stats[f"{var}.std"]
            assert ((lower <= pred_ds_mean[var]).all()) & (
                (pred_ds_mean[var] <= upper).all()
            ), (
                f"{var} fell outside the [-3σ, +3σ] climatology band; "
                "regenerate jcm/data/test/echam_t63l47/default_statistics.nc "
                "(and spinup_state.nc) if the deviation is intentional."
            )


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


