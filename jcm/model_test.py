import unittest
from dinosaur import primitive_equations_states
from jcm.physics.speedy.speedy_physics import SpeedyPhysics
from jcm.params import Parameters
import jax.tree_util as jtu

class TestModelUnit(unittest.TestCase):

    def test_held_suarez_model(self):
        from jcm.physics.held_suarez.held_suarez_physics import HeldSuarezPhysics
        from jcm.model import Model
        layers = 8
        model = Model(
            layers=layers,
            time_step=180,
            total_time=2,
            save_interval=1,
            physics=HeldSuarezPhysics(),
        )

        state = model.get_initial_state()
        state.tracers = {'specific_humidity': 1e-4 * primitive_equations_states.gaussian_scalar(model.coords, model.physics_specs)}

        modal_zxy, nodal_zxy = model.coords.modal_shape, model.coords.nodal_shape
        nodal_tzxy = (model.outer_steps,) + nodal_zxy

        final_state, predictions = model.unroll(state)
        dynamics_predictions = predictions['dynamics']

        self.assertIsNotNone(final_state.log_surface_pressure)
        self.assertIsNotNone(final_state.tracers['specific_humidity'])

        self.assertIsNotNone(dynamics_predictions.u_wind)
        self.assertIsNotNone(dynamics_predictions.v_wind)
        self.assertIsNotNone(dynamics_predictions.temperature)
        self.assertIsNotNone(dynamics_predictions.specific_humidity)
        self.assertIsNotNone(dynamics_predictions.geopotential)
        self.assertIsNotNone(dynamics_predictions.surface_pressure)

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
        self.assertTupleEqual(dynamics_predictions.surface_pressure.shape, (nodal_tzxy[0],) + nodal_tzxy[2:])
        
    def test_speedy_model(self):
        from jcm.model import Model

        layers = 8
        # optionally add a boundary conditions file
        model = Model(
            layers=layers,
            time_step=720,
            total_time=2,
            save_interval=1,
        )
    
        state = model.get_initial_state()

        # Specify humidity perturbation in kg/kg
        state.tracers = {'specific_humidity': 1e-2 * primitive_equations_states.gaussian_scalar(model.coords, model.physics_specs)}

        modal_zxy, nodal_zxy = model.coords.modal_shape, model.coords.nodal_shape
        nodal_tzxy = (model.outer_steps,) + nodal_zxy
    
        final_state, predictions = model.unroll(state)
        dynamics_predictions = predictions['dynamics']

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
        self.assertIsNotNone(dynamics_predictions.surface_pressure)

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
        self.assertTupleEqual(dynamics_predictions.surface_pressure.shape, (nodal_tzxy[0],) + nodal_tzxy[2:])
        
    def test_speedy_model_gradients_isnan(self):
        import jax
        import jax.numpy as jnp
        from jcm.model import Model

        def make_ones_dinosaur_State_object(state, choose_sim_time = jnp.float32(1.0)):
            return jtu.tree_map(lambda x: jnp.ones_like(x), state)

        def make_ones_prediction_object(pred):
            return jtu.tree_map(lambda x: jnp.ones_like(x), pred)
        
        #create model that goes through one timestep
        model = Model(total_time=(1/48.),  save_interval=(1/48.))
        state = model.get_initial_state()

        # Calculate gradients
        primals, f_vjp = jax.vjp(model.unroll, state) 
        
        input = (make_ones_dinosaur_State_object(primals[0]), make_ones_prediction_object(primals[1]))

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

        def make_ones_dinosaur_State_object(state, choose_sim_time = jnp.float32(1.0)):
            return jtu.tree_map(lambda x: jnp.ones_like(x), state)
        
        def make_ones_prediction_object(pred): 
            return jtu.tree_map(lambda x: jnp.ones_like(x), pred)
        
        model = Model(total_time=(1/24.), save_interval=(1/48.))
        state = model.get_initial_state()

        # Calculate gradients
        primals, f_vjp = jax.vjp(model.unroll, state) 
        input = (make_ones_dinosaur_State_object(primals[0]), make_ones_prediction_object(primals[1]))
        df_dstate = f_vjp(input) 

        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].vorticity)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].divergence)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].temperature_variation)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].log_surface_pressure)))
        self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].tracers['specific_humidity'])))
        # self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].sim_time))) FIXME: this is ending up nan

    def test_speedy_model_param_gradients_isnan_vjp(self):
        import jax
        import jax.numpy as jnp
        from jcm.model import Model, get_coords
        from jcm.boundaries import initialize_boundaries
        
        from pathlib import Path
        boundaries_dir = Path(__file__).resolve().parent / 'data/bc/t30/clim'
        
        if not (boundaries_dir / 'boundaries_daily.nc').exists():
            import subprocess, sys
            subprocess.run([sys.executable, str(boundaries_dir / 'interpolate.py')], check=True)
        
        default_boundaries = lambda coords=get_coords(): initialize_boundaries(
            boundaries_dir / 'boundaries_daily.nc',
            coords.horizontal
        )

        create_model = lambda params=Parameters.default(): Model(
            total_time=2./24.,
            save_interval=1/24.,
            parameters=params,
            boundaries=default_boundaries(),
        )
        
        def model_run_wrapper(params):
            model = create_model(params)
            state = model.get_initial_state()
            _, predictions = model.unroll(state)
            return predictions
        
        def make_ones_prediction_object(pred): 
            return jtu.tree_map(lambda x: jnp.ones_like(x), pred)
        
        # Calculate gradients using VJP
        params = Parameters.default()
        primal, f_vjp = jax.vjp(model_run_wrapper, params)
        df_dparams = f_vjp(make_ones_prediction_object(primal))

        self.assertFalse(df_dparams[0].isnan().any_true())
    

    def test_speedy_model_param_gradients_isnan_jvp(self):
        import jax
        import jax.numpy as jnp
        import numpy as np
        from jcm.model import Model, get_coords
        from jcm.boundaries import initialize_boundaries

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
            import subprocess, sys
            subprocess.run([sys.executable, str(boundaries_dir / 'interpolate.py')], check=True)
        
        default_boundaries = lambda coords=get_coords(): initialize_boundaries(
            boundaries_dir / 'boundaries_daily.nc',
            coords.horizontal
        )

        create_model = lambda params=Parameters.default(): Model(
            total_time=2./24.,
            save_interval=1/24.,
            parameters=params,
            boundaries=default_boundaries(),
        )
        
        def model_run_wrapper(params):
            model = create_model(params)
            state = model.get_initial_state()
            _, predictions = model.unroll(state)
            return predictions
        
        # Calculate gradients using JVP
        params = Parameters.default()
        tangent = make_ones_parameters_object(params)
        y, jvp_sum = jax.jvp(model_run_wrapper, (params,), (tangent,))
        state = jvp_sum['dynamics']
        physics_data = jvp_sum['physics']

        # Check dinosaur states
        self.assertFalse(jnp.any(jnp.isnan(state.u_wind)))
        self.assertFalse(jnp.any(jnp.isnan(state.v_wind)))
        self.assertFalse(jnp.any(jnp.isnan(state.temperature)))
        self.assertFalse(jnp.any(jnp.isnan(state.specific_humidity)))
        self.assertFalse(jnp.any(jnp.isnan(state.geopotential)))
        self.assertFalse(jnp.any(jnp.isnan(state.surface_pressure)))
        # self.assertFalse(jnp.any(jnp.isnan(df_dstate[0].sim_time))) FIXME: this is ending up nan
        # Check Physics Data object
        # self.assertFalse(physics_data.isnan().any_true())  FIXME: shortwave_rad has integer value somewehre


