import unittest
import jax
import jax.tree_util as jtu
import jax.numpy as jnp
import numpy as np
import pytest

class TestModelUnit(unittest.TestCase):
    def setUp(self):
        global HeldSuarezPhysics, SpeedyPhysics, Parameters, Model, ones_like, boundaries
        from jcm.physics.held_suarez.held_suarez_physics import HeldSuarezPhysics
        from jcm.physics.speedy.speedy_physics import SpeedyPhysics
        from jcm.physics.speedy.params import Parameters
        from jcm.model import Model, get_coords
        from jcm.utils import ones_like

        from pathlib import Path
        from jcm.boundaries import boundaries_from_file
        
        boundaries_dir = Path(__file__).resolve().parent / 'data/bc/t30/clim'
        
        if not (boundaries_dir / 'boundaries_daily.nc').exists():
            import subprocess
            import sys
            subprocess.run([sys.executable, str(boundaries_dir / 'interpolate.py')], check=True)
        
        boundaries = boundaries_from_file(
            boundaries_dir / 'boundaries_daily.nc',
            get_coords().horizontal
        )

    @pytest.mark.slow
    def test_held_suarez_model(self):
        layers = 8
        model = Model(
            layers=layers,
            physics=HeldSuarezPhysics(),
            orography=boundaries.orog,
        )

        save_interval, total_time = 90, 360
        predictions = model.run(
            total_time=total_time,
            save_interval=save_interval,
        )
        final_state, dynamics_predictions = model._final_modal_state, predictions.dynamics

        modal_zxy, nodal_zxy = model.coords.modal_shape, model.coords.nodal_shape
        nodal_tzxy = (int(total_time / save_interval),) + nodal_zxy

        self.assertIsNotNone(final_state)

        self.assertTupleEqual(final_state.divergence.shape, modal_zxy)
        self.assertTupleEqual(final_state.vorticity.shape, modal_zxy)
        self.assertTupleEqual(final_state.temperature_variation.shape, modal_zxy)
        self.assertTupleEqual(final_state.tracers['specific_humidity'].shape, modal_zxy)
        self.assertTupleEqual(final_state.log_surface_pressure.shape, (1,) + modal_zxy[1:])

        self.assertFalse(jnp.isnan(final_state.divergence).any())
        self.assertFalse(jnp.isnan(final_state.vorticity).any())
        self.assertFalse(jnp.isnan(final_state.temperature_variation).any())
        self.assertFalse(jnp.isnan(final_state.tracers['specific_humidity']).any())
        self.assertFalse(jnp.isnan(final_state.log_surface_pressure).any())

        self.assertIsNotNone(dynamics_predictions)

        self.assertTupleEqual(dynamics_predictions.u_wind.shape, nodal_tzxy)
        self.assertTupleEqual(dynamics_predictions.v_wind.shape, nodal_tzxy)
        self.assertTupleEqual(dynamics_predictions.temperature.shape, nodal_tzxy)
        self.assertTupleEqual(dynamics_predictions.specific_humidity.shape, nodal_tzxy)
        self.assertTupleEqual(dynamics_predictions.geopotential.shape, nodal_tzxy)
        self.assertTupleEqual(dynamics_predictions.normalized_surface_pressure.shape, (nodal_tzxy[0],) + nodal_tzxy[2:])
    
        self.assertFalse(jnp.isnan(dynamics_predictions.u_wind).any())
        self.assertFalse(jnp.isnan(dynamics_predictions.v_wind).any())
        self.assertFalse(jnp.isnan(dynamics_predictions.temperature).any())
        self.assertFalse(jnp.isnan(dynamics_predictions.specific_humidity).any())
        self.assertFalse(jnp.isnan(dynamics_predictions.geopotential).any())
        self.assertFalse(jnp.isnan(dynamics_predictions.normalized_surface_pressure).any())

    @pytest.mark.slow
    def test_held_suarez_gradients_jvp(self):
        state = Model()._prepare_initial_modal_state()

        def fn(state):
            model = Model(physics=HeldSuarezPhysics(), orography=boundaries.orog)
            predictions = model.run(initial_state=state, save_interval=30, total_time=60)
            return model._final_modal_state, predictions

        tangent = ones_like(state)
        _, jvp_sum = jax.jvp(fn, (state,), (tangent,))
        
        d_final_d_tangent, d_predictions_d_tangent = jvp_sum
        
        self.assertFalse(jnp.any(jnp.isnan(d_final_d_tangent.vorticity)))
        self.assertFalse(jnp.any(jnp.isnan(d_final_d_tangent.divergence)))
        self.assertFalse(jnp.any(jnp.isnan(d_final_d_tangent.temperature_variation)))
        self.assertFalse(jnp.any(jnp.isnan(d_final_d_tangent.log_surface_pressure)))
        self.assertFalse(jnp.any(jnp.isnan(d_final_d_tangent.tracers['specific_humidity'])))
        self.assertFalse(jnp.any(jnp.isnan(d_final_d_tangent.sim_time)))
        
        self.assertFalse(jnp.any(jnp.isnan(d_predictions_d_tangent.dynamics.u_wind)))
        self.assertFalse(jnp.any(jnp.isnan(d_predictions_d_tangent.dynamics.v_wind)))
        self.assertFalse(jnp.any(jnp.isnan(d_predictions_d_tangent.dynamics.temperature)))
        self.assertFalse(jnp.any(jnp.isnan(d_predictions_d_tangent.dynamics.specific_humidity)))
        self.assertFalse(jnp.any(jnp.isnan(d_predictions_d_tangent.dynamics.geopotential)))
        self.assertFalse(jnp.any(jnp.isnan(d_predictions_d_tangent.dynamics.normalized_surface_pressure)))

    @pytest.mark.slow
    def test_held_suarez_gradients_vjp(self):
        state = Model()._prepare_initial_modal_state()

        def fn(state):
            model = Model(physics=HeldSuarezPhysics(), orography=boundaries.orog)
            predictions = model.run(initial_state=state, save_interval=30, total_time=60)
            return model._final_modal_state, predictions

        (final_state, preds), f_vjp = jax.vjp(fn, state)
        cotangent = (ones_like(final_state), ones_like(preds))
        d_cotangent_d_state, = f_vjp(cotangent)
        
        self.assertFalse(jnp.any(jnp.isnan(d_cotangent_d_state.vorticity)))
        self.assertFalse(jnp.any(jnp.isnan(d_cotangent_d_state.divergence)))
        self.assertFalse(jnp.any(jnp.isnan(d_cotangent_d_state.temperature_variation)))
        self.assertFalse(jnp.any(jnp.isnan(d_cotangent_d_state.log_surface_pressure)))
        self.assertFalse(jnp.any(jnp.isnan(d_cotangent_d_state.tracers['specific_humidity'])))
        self.assertFalse(jnp.any(jnp.isnan(d_cotangent_d_state.sim_time))) # FIXME: this is ending up nan
        
    def test_speedy_model(self):
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

        self.assertTupleEqual(final_state.divergence.shape, modal_zxy)
        self.assertTupleEqual(final_state.vorticity.shape, modal_zxy)
        self.assertTupleEqual(final_state.temperature_variation.shape, modal_zxy)
        self.assertTupleEqual(final_state.tracers['specific_humidity'].shape, modal_zxy)
        self.assertTupleEqual(final_state.log_surface_pressure.shape, (1,) + modal_zxy[1:])

        self.assertFalse(jnp.isnan(final_state.divergence).any())
        self.assertFalse(jnp.isnan(final_state.vorticity).any())
        self.assertFalse(jnp.isnan(final_state.temperature_variation).any())
        self.assertFalse(jnp.isnan(final_state.tracers['specific_humidity']).any())
        self.assertFalse(jnp.isnan(final_state.log_surface_pressure).any())

        self.assertIsNotNone(dynamics_predictions)

        self.assertTupleEqual(dynamics_predictions.u_wind.shape, nodal_tzxy)
        self.assertTupleEqual(dynamics_predictions.v_wind.shape, nodal_tzxy)
        self.assertTupleEqual(dynamics_predictions.temperature.shape, nodal_tzxy)
        self.assertTupleEqual(dynamics_predictions.specific_humidity.shape, nodal_tzxy)
        self.assertTupleEqual(dynamics_predictions.geopotential.shape, nodal_tzxy)
        self.assertTupleEqual(dynamics_predictions.normalized_surface_pressure.shape, (nodal_tzxy[0],) + nodal_tzxy[2:])
    
        self.assertFalse(jnp.isnan(dynamics_predictions.u_wind).any())
        self.assertFalse(jnp.isnan(dynamics_predictions.v_wind).any())
        self.assertFalse(jnp.isnan(dynamics_predictions.temperature).any())
        self.assertFalse(jnp.isnan(dynamics_predictions.specific_humidity).any())
        self.assertFalse(jnp.isnan(dynamics_predictions.geopotential).any())
        self.assertFalse(jnp.isnan(dynamics_predictions.normalized_surface_pressure).any())

    @pytest.mark.slow
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

    @pytest.mark.slow
    def test_speedy_model_gradients_isnan(self):
        # Create model that goes through one timestep
        model = Model()
        state = model._prepare_initial_modal_state()

        def fn(state):
            _ = model.run(total_time=0) # to set up model fields
            predictions = model.run(initial_state=state, save_interval=(1/48.), total_time=(1/48.))
            return model._final_modal_state, predictions

        # Calculate gradients
        (final_state, preds), f_vjp = jax.vjp(fn, state)
        cotangent = (ones_like(final_state), ones_like(preds))
        d_cotangent_d_state, = f_vjp(cotangent)
        
        self.assertFalse(jnp.any(jnp.isnan(d_cotangent_d_state.vorticity)))
        self.assertFalse(jnp.any(jnp.isnan(d_cotangent_d_state.divergence)))
        self.assertFalse(jnp.any(jnp.isnan(d_cotangent_d_state.temperature_variation)))
        self.assertFalse(jnp.any(jnp.isnan(d_cotangent_d_state.log_surface_pressure)))
        self.assertFalse(jnp.any(jnp.isnan(d_cotangent_d_state.tracers['specific_humidity'])))
        # self.assertFalse(jnp.any(jnp.isnan(d_cotangent_d_state.sim_time))) FIXME: this is ending up nan

    @pytest.mark.slow
    def test_speedy_model_gradients_multiple_timesteps_isnan(self):
        model = Model()
        state = model._prepare_initial_modal_state()

        def fn(state):
            predictions = model.run(initial_state=state, save_interval=(1/48.), total_time=(1/24.))
            return model._final_modal_state, predictions

        # Calculate gradients
        (final_state, preds), f_vjp = jax.vjp(fn, state)
        cotangent = (ones_like(final_state), ones_like(preds))
        d_cotangent_d_state, = f_vjp(cotangent)

        self.assertFalse(jnp.any(jnp.isnan(d_cotangent_d_state.vorticity)))
        self.assertFalse(jnp.any(jnp.isnan(d_cotangent_d_state.divergence)))
        self.assertFalse(jnp.any(jnp.isnan(d_cotangent_d_state.temperature_variation)))
        self.assertFalse(jnp.any(jnp.isnan(d_cotangent_d_state.log_surface_pressure)))
        self.assertFalse(jnp.any(jnp.isnan(d_cotangent_d_state.tracers['specific_humidity'])))
        # self.assertFalse(jnp.any(jnp.isnan(d_cotangent_d_state.sim_time))) FIXME: this is ending up nan

    @pytest.mark.slow
    def test_speedy_model_param_gradients_isnan_vjp(self):
        create_model = lambda params=Parameters.default(): Model(
            orography=boundaries.orog,
            physics=SpeedyPhysics(parameters=params),
        )
        
        fn = lambda params: create_model(params).run(save_interval=1/24., total_time=2./24.)

        # Calculate gradients using VJP
        params = Parameters.default()
        primal, f_vjp = jax.vjp(fn, params)
        df_dparams, = f_vjp(ones_like(primal))

        self.assertFalse(df_dparams.isnan().any_true())
    
    @pytest.mark.slow
    def test_speedy_model_param_gradients_isnan_jvp(self):
        import numpy as np
        def make_ones_parameters_object(params):
            def make_tangent(x):
                if jnp.issubdtype(jnp.result_type(x), jnp.bool_):
                    return np.ones((), dtype=jax.dtypes.float0)
                elif jnp.issubdtype(jnp.result_type(x), jnp.integer):
                    return np.ones((), dtype=jax.dtypes.float0)
                else:
                    return jnp.ones_like(x)
            return jtu.tree_map(make_tangent, params)

        create_model = lambda params=Parameters.default(): Model(
            orography=boundaries.orog,
            physics=SpeedyPhysics(parameters=params),
        )

        model_run_wrapper = lambda params: create_model(params).run(save_interval=1/24., total_time=2./24.)

        # Calculate gradients using JVP
        params = Parameters.default()
        tangent = make_ones_parameters_object(params)
        y, jvp_sum = jax.jvp(model_run_wrapper, (params,), (tangent,))
        dynamics_gradients, physics_gradients = jvp_sum.dynamics, jvp_sum.physics

        self.assertFalse(jnp.any(jnp.isnan(dynamics_gradients.u_wind)))
        self.assertFalse(jnp.any(jnp.isnan(dynamics_gradients.v_wind)))
        self.assertFalse(jnp.any(jnp.isnan(dynamics_gradients.temperature)))
        self.assertFalse(jnp.any(jnp.isnan(dynamics_gradients.specific_humidity)))
        self.assertFalse(jnp.any(jnp.isnan(dynamics_gradients.geopotential)))
        self.assertFalse(jnp.any(jnp.isnan(dynamics_gradients.normalized_surface_pressure)))

        grads_dict = SpeedyPhysics().data_struct_to_dict(physics_gradients, create_model().geometry)

        for k, v in grads_dict.items():
            if v.dtype == jax.dtypes.float0:
                continue
            self.assertFalse(jnp.any(jnp.isnan(v)), f"NaN in gradient for {k}")
