import unittest
import jax.tree_util as jtu

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
            return jtu.tree_map(lambda x: make_tangent(x), params)
        
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

    def test_speedy_model_default_statistics(self):
        from jcm.model import Model, get_coords
        from jcm.boundaries import boundaries_from_file
        import numpy as np

        realistic_boundaries = boundaries_from_file(
            'jcm/data/bc/t30/clim/boundaries.nc',
            get_coords().horizontal,
)
        # default model settings
        # use reslistic orography from boundaries file
        model = Model(
            orography=realistic_boundaries.orog,
        )

        # save every month, run for 3 months, turn on averaging
        predictions = model.run(
            save_interval=30,
            total_time=90,
            # add output averaging flag
            # output_averages=True
        )
        pred_ds = model.predictions_to_xarray(predictions)

        # load test file for comparison
        default_stats = np.load('jcm/data/test_data/speedy_default_statistics.npz')

        # check statistics of the model state
        lower_q = default_stats['q_mean'] - 2*default_stats['q_std']
        upper_q = default_stats['q_mean'] + 2*default_stats['q_std']
        assert lower_q <= pred_ds['specific_humidity'].isel(time=-1) <= upper_q

        lower_t = default_stats['t_mean'] - 2*default_stats['t_std']
        upper_t = default_stats['t_mean'] + 2*default_stats['t_std']
        assert lower_t <= pred_ds['temperature'].isel(time=-1) <= upper_t

        lower_u = default_stats['u_mean'] - 2*default_stats['u_std']
        upper_u = default_stats['u_mean'] + 2*default_stats['u_std']
        assert lower_u <= pred_ds['u_wind'].isel(time=-1) <= upper_u

        lower_v = default_stats['v_mean'] - 2*default_stats['v_std']
        upper_v = default_stats['v_mean'] + 2*default_stats['v_std']
        assert lower_v <= pred_ds['v_wind'].isel(time=-1) <= upper_v

        lower_nsp = default_stats['nsp_mean'] - 2*default_stats['nsp_std']
        upper_nsp = default_stats['nsp_mean'] + 2*default_stats['nsp_std']
        assert lower_nsp <= pred_ds['normalized_surface_pressure'].isel(time=-1) <= upper_nsp

        lower_phi = default_stats['phi_mean'] - 2*default_stats['phi_std']
        upper_phi = default_stats['phi_mean'] + 2*default_stats['phi_std']
        assert lower_phi <= pred_ds['geopotential'].isel(time=-1) <= upper_phi

        # check statistics of some of the physics outputs
        TOA_rad = pred_ds['shortwave_rad.ftop']-pred_ds['longwave_rad.ftop']
        lower_TOA = default_stats['TOA_rad_mean'] - 2*default_stats['TOA_rad_std']
        upper_TOA = default_stats['TOA_rad_mean'] + 2*default_stats['TOA_rad_std']
        assert lower_TOA <= TOA_rad.isel(time=-1) <= upper_TOA

        lower_rh = default_stats['rh_mean'] - 2*default_stats['rh_std']
        upper_rh = default_stats['rh_mean'] + 2*default_stats['rh_std']
        assert lower_rh <= pred_ds['humidity.rh'].isel(time=-1) <= upper_rh

        lower_cloudstr = default_stats['cloudstr_mean'] - 2*default_stats['cloudstr_std']
        upper_cloudstr = default_stats['cloudstr_mean'] + 2*default_stats['cloudstr_std']
        assert lower_cloudstr <= pred_ds['shortwave_rad.cloudstr'].isel(time=-1) <= upper_cloudstr

        lower_qcloud = default_stats['qcloud_mean'] - 2*default_stats['qcloud_std']
        upper_qcloud = default_stats['qcloud_mean'] + 2*default_stats['qcloud_std']
        assert lower_qcloud <= pred_ds['shortwave_rad.qcloud'].isel(time=-1) <= upper_qcloud

        lower_precnv = default_stats['precnv_mean'] - 2*default_stats['precnv_std']
        upper_precnv = default_stats['precnv_mean'] + 2*default_stats['precnv_std']
        assert lower_precnv <= pred_ds['convection.precnv'].isel(time=-1) <= upper_precnv

        lower_precls = default_stats['precls_mean'] - 2*default_stats['precls_std']
        upper_precls = default_stats['precls_mean'] + 2*default_stats['precls_std']
        assert lower_precls <= pred_ds['condensation.precls'].isel(time=-1) <= upper_precls




