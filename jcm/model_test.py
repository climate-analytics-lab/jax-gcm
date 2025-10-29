import unittest
import jax.tree_util as jtu
import numpy as np
import pytest
from jax.test_util import check_vjp, check_jvp
import functools

class TestModelUnit(unittest.TestCase):
    def setUp(self):
        global SpeedyPhysics, Parameters
        from jcm.physics.speedy.speedy_physics import SpeedyPhysics
        from jcm.physics.speedy.params import Parameters

    def test_held_suarez_model(self):
        from jcm.physics.held_suarez.held_suarez_physics import HeldSuarezPhysics
        from jcm.model import Model
        layers = 8
        model = Model(
            layers=layers,
            time_step=180,
            physics=HeldSuarezPhysics(),
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

        # optionally add a boundary conditions file
        model = Model(
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

    # @pytest.mark.slow
    @pytest.mark.skip(reason="finite differencing produces nans")
    def test_speedy_model_averages(self):
        from jcm.model import Model

        model = Model(
            time_step=30, # to make sure this test stays valid if we ever change the default timestep
        )
        preds = model.run(save_interval=.5/24., total_time=2/24.)

        true_avg_preds = jtu.tree_map(lambda a: np.mean(a, axis=0), preds)

        avg_model = Model(
            time_step=30,
        )
        avg_preds = avg_model.run(
            save_interval=2/24.,
            total_time=2/24.,
            output_averages=True,
        )

        jtu.tree_map(
            lambda a1, a2: self.assertTrue(np.allclose(a1, a2, atol=1e-4)),
            true_avg_preds,
            avg_preds
        )

    # @pytest.mark.slow
    @pytest.mark.skip(reason="finite differencing produces nans")
    def test_speedy_model_gradients_isnan(self):
        import jax
        import jax.numpy as jnp
        from jcm.model import Model
        from jcm.utils import ones_like

        # Create model that goes through one timestep
        model = Model()
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
        # self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].sim_time))) FIXME: this is ending up nan

    # @pytest.mark.slow
    @pytest.mark.skip(reason="finite differencing produces nans")
    def test_speedy_model_gradients_multiple_timesteps_isnan(self):
        import jax
        import jax.numpy as jnp
        from jcm.model import Model
        from jcm.utils import ones_like

        model = Model()
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
        # self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].sim_time))) FIXME: this is ending up nan

    # @pytest.mark.slow
    @pytest.mark.skip(reason="finite differencing produces nans")
    def test_speedy_model_param_gradients_isnan_vjp(self):
        import jax
        from jcm.model import Model, get_coords
        from jcm.boundaries import boundaries_from_file
        from jcm.utils import ones_like

        from pathlib import Path
        boundaries_dir = Path(__file__).resolve().parent / 'data/bc/t30/clim'
        
        if not (boundaries_dir / 'boundaries_daily.nc').exists():
            import subprocess
            import sys
            subprocess.run([sys.executable, str(boundaries_dir / 'interpolate.py')], check=True)
        
        boundaries = boundaries_from_file(
            boundaries_dir / 'boundaries_daily.nc',
            get_coords().horizontal
        )

        create_model = lambda params=Parameters.default(): Model(
            orography=boundaries.orog,
            physics=SpeedyPhysics(parameters=params),
        )
        
        fn = lambda params: create_model(params).run(save_interval=1/24., total_time=2./24.)

        # Calculate gradients using VJP
        params = Parameters.default()
        primal, f_vjp = jax.vjp(fn, params)
        df_dparams = f_vjp(ones_like(primal))

        self.assertFalse(df_dparams[0].isnan().any_true())
    
    # @pytest.mark.slow
    @pytest.mark.skip(reason="finite differencing produces nans")
    def test_speedy_model_param_gradients_isnan_jvp(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jcm.model import Model, get_coords
        from jcm.boundaries import boundaries_from_file

        def make_ones_parameters_object(params):
            def make_tangent(x):
                if jnp.issubdtype(jnp.result_type(x), jnp.bool_):
                    return np.ones((), dtype=jax.dtypes.float0)
                elif jnp.issubdtype(jnp.result_type(x), jnp.integer):
                    return np.ones((), dtype=jax.dtypes.float0)
                else:
                    return jnp.ones_like(x)
            return jtu.tree_map(make_tangent, params)
        
        from pathlib import Path
        boundaries_dir = Path(__file__).resolve().parent / 'data/bc/t30/clim'
        
        if not (boundaries_dir / 'boundaries_daily.nc').exists():
            import subprocess
            import sys
            subprocess.run([sys.executable, str(boundaries_dir / 'interpolate.py')], check=True)

        boundaries = boundaries_from_file(
            boundaries_dir / 'boundaries_daily.nc',
            get_coords().horizontal
        )

        create_model = lambda params=Parameters.default(): Model(
            orography=boundaries.orog,
            physics=SpeedyPhysics(parameters=params),
        )

        model_run_wrapper = lambda params: create_model(params).run(save_interval=1/24., total_time=2./24.)

        # Calculate gradients using JVP
        params = Parameters.default()
        tangent = make_ones_parameters_object(params)
        y, jvp_sum = jax.jvp(model_run_wrapper, (params,), (tangent,))
        state = jvp_sum.dynamics
        physics_data = jvp_sum.physics

        # Check dynamics state
        self.assertFalse(jnp.any(jnp.isnan(state.u_wind)))
        self.assertFalse(jnp.any(jnp.isnan(state.v_wind)))
        self.assertFalse(jnp.any(jnp.isnan(state.temperature)))
        self.assertFalse(jnp.any(jnp.isnan(state.specific_humidity)))
        self.assertFalse(jnp.any(jnp.isnan(state.geopotential)))
        self.assertFalse(jnp.any(jnp.isnan(state.normalized_surface_pressure)))
        # self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].sim_time))) FIXME: this is ending up nan
        # Check Physics Data object
        # self.assertFalse(physics_data.isnan().any_true())  FIXME: shortwave_rad has integer value somewehre
    @pytest.mark.skip(reason="finite differencing produces nans")
    def test_speedy_model_state_gradient_check(self):
        import jax
        import jax.numpy as jnp
        from jcm.model import Model
        from jcm.utils import ones_like, convert_back, convert_to_float

        # Create model that goes through one timestep
        model = Model()
        state = model._prepare_initial_modal_state()

        state_floats = convert_to_float(state)

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
        from jcm.model import Model, get_coords
        from jcm.boundaries import boundaries_from_file
        import xarray as xr

        from pathlib import Path
        boundaries_dir = Path(__file__).resolve().parent / 'data/bc/t30/clim'
        
        if not (boundaries_dir / 'boundaries_daily.nc').exists():
            import subprocess
            import sys
            subprocess.run([sys.executable, str(boundaries_dir / 'interpolate.py')], check=True)

        realistic_boundaries = boundaries_from_file(
            boundaries_dir / 'boundaries_daily.nc',
            get_coords().horizontal
        )

        stats_file = Path(__file__).resolve().parent / 'data/test/t30/default_statistics.nc'

        # default model settings
        # use reslistic orography from boundaries file
        model = Model(
            orography=realistic_boundaries.orog,
            time_step=60.,
        )

        # save every month, run for 2 months, turn on averaging so that the last prediction is the last monthly average
        predictions = model.run(
            save_interval=30, 
            total_time=60,
            output_averages=True,
            boundaries=realistic_boundaries
        )
        pred_ds = model.predictions_to_xarray(predictions)

        # load test file for comparison
        default_stats = xr.open_dataset(stats_file)

        # tolerance in # of standard deviations
        tol = 3

        # check statistics of the model state
        lower_q = default_stats['q_mean'] - tol*default_stats['q_std']
        upper_q = default_stats['q_mean'] + tol*default_stats['q_std']
        q = pred_ds['specific_humidity'].isel(time=-1)
        assert ((lower_q <= q).all()) & ((q <= upper_q).all())

        lower_t = default_stats['t_mean'] - tol*default_stats['t_std']
        upper_t = default_stats['t_mean'] + tol*default_stats['t_std']
        t = pred_ds['temperature'].isel(time=-1)
        assert ((lower_t <= t).all()) & ((t <= upper_t).all())

        lower_u = default_stats['u_mean'] - tol*default_stats['u_std']
        upper_u = default_stats['u_mean'] + tol*default_stats['u_std']
        u = pred_ds['u_wind'].isel(time=-1)
        assert ((lower_u <= u).all()) & ((u <= upper_u).all())

        lower_v = default_stats['v_mean'] - tol*default_stats['v_std']
        upper_v = default_stats['v_mean'] + tol*default_stats['v_std']
        v = pred_ds['v_wind'].isel(time=-1)
        assert ((lower_v <= v).all()) & ((v <= upper_v).all())

        lower_nsp = default_stats['nsp_mean'] - tol*default_stats['nsp_std']
        upper_nsp = default_stats['nsp_mean'] + tol*default_stats['nsp_std']
        nsp = pred_ds['normalized_surface_pressure'].isel(time=-1)
        assert ((lower_nsp <= nsp).all()) & ((nsp <= upper_nsp).all())

        lower_phi = default_stats['phi_mean'] - tol*default_stats['phi_std']
        upper_phi = default_stats['phi_mean'] + tol*default_stats['phi_std']
        phi = pred_ds['geopotential'].isel(time=-1)
        assert ((lower_phi <= phi).all()) & ((phi <= upper_phi).all())

        # check statistics of some of the physics outputs
        TOA_rad = (pred_ds['shortwave_rad.ftop']-pred_ds['longwave_rad.ftop']).isel(time=-1)
        lower_TOA = default_stats['TOA_rad_mean'] - tol*default_stats['TOA_rad_std']
        upper_TOA = default_stats['TOA_rad_mean'] + tol*default_stats['TOA_rad_std']
        assert ((lower_TOA <= TOA_rad).all()) & ((TOA_rad <= upper_TOA).all())

        lower_rh = default_stats['rh_mean'] - tol*default_stats['rh_std']
        upper_rh = default_stats['rh_mean'] + tol*default_stats['rh_std']
        rh = pred_ds['humidity.rh'].isel(time=-1)
        assert ((lower_rh <= rh).all()) & ((rh <= upper_rh).all())

        # this is working, except cloudstr, precnv, and precls all have spots where the variability is 0 (std dev of 0) and
        # those tests fail (likely because there are very small differences)...
        # lower_cloudstr = default_stats['cloudstr_mean'] - tol*default_stats['cloudstr_std']
        # upper_cloudstr = default_stats['cloudstr_mean'] + tol*default_stats['cloudstr_std']
        # cloudstr = pred_ds['shortwave_rad.cloudstr'].isel(time=-1)
        # assert ((lower_cloudstr <= cloudstr).all()) & ((cloudstr <= upper_cloudstr).all())

        lower_qcloud = default_stats['qcloud_mean'] - tol*default_stats['qcloud_std']
        upper_qcloud = default_stats['qcloud_mean'] + tol*default_stats['qcloud_std']
        qcloud = pred_ds['shortwave_rad.qcloud'].isel(time=-1)
        assert ((lower_qcloud <= qcloud).all()) & ((qcloud <= upper_qcloud).all())

        # lower_precnv = default_stats['precnv_mean'] - tol*default_stats['precnv_std']
        # upper_precnv = default_stats['precnv_mean'] + tol*default_stats['precnv_std']
        # precnv = pred_ds['convection.precnv'].isel(time=-1)
        # assert ((lower_precnv <= precnv).all()) & ((precnv <= upper_precnv).all())

        # lower_precls = default_stats['precls_mean'] - tol*default_stats['precls_std']
        # upper_precls = default_stats['precls_mean'] + tol*default_stats['precls_std']
        # precls = pred_ds['condensation.precls'].isel(time=-1)
        # assert ((lower_precls <= precls).all()) & ((precls <= upper_precls).all())




