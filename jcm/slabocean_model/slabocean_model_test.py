import unittest
from dinosaur import primitive_equations_states

import jax.tree_util as jtu
import pandas as pd

from jcm.params import Parameters
from jcm.physics_data import PhysicsData


class TestModelUnit(unittest.TestCase):

    def test_model_init(self):
        
        import jax
        import jax.numpy as jnp
        import numpy as np
        import jcm.slabocean_model as som

        from jcm.boundaries import initialize_boundaries
        from pathlib import Path
        
        """
        def make_ones_parameters_object(params):
            def make_tangent(x):
                if jnp.issubdtype(jnp.result_type(x), jnp.bool_):
                    return np.ones((), dtype=jax.dtypes.float0)
                elif jnp.issubdtype(jnp.result_type(x), jnp.integer):
                    return np.ones((), dtype=jax.dtypes.float0)
                else:
                    return jnp.ones_like(x)
            return jtu.tree_map(lambda x: make_tangent(x), params)
        """

        boundaries_dir = Path(__file__).resolve().parent / ".." / 'data/bc/t30/clim'

        # Generate daily values from monthly climatology
        daily_boundary_condition_file = boundaries_dir / 'boundaries_daily.nc'
        if not daily_boundary_condition_file.exists():
            import subprocess, sys
            subprocess.run([sys.executable, str(boundaries_dir / 'interpolate.py')], check=True)
        
        coords = som.misc.get_coords()
        
        hori_shape  = coords.horizontal.nodal_shape
        vert_layers = coords.vertical.layers
        
        
        boundaries = initialize_boundaries(
            daily_boundary_condition_file,
            coords.horizontal,
        )
        boundaries.ocn_coupling_flag = True
        
        params = Parameters.default()
        physics_data = PhysicsData.zeros(hori_shape, vert_layers)
       
        ev = som.Env(
            time_step = 600.0,                # 10 min
            save_interval = 86400.0,          # 1 day
            total_time = 86400.0 * 10,        # 10 days
            start_date = pd.Timestamp("2000-01-01"),
            horizontal_resolution = 31,
            coords = coords,
            boundaries = boundaries,
            parameters = params,
            physics_data = physics_data,
        )
        
        model = som.Model(ev=ev)
        self.assertTrue(model is not None)
        
        
        hours_per_day = 24
        steps_per_hour = 6

        recorder = som.Recorder(
            count_per_avg = steps_per_hour,
            model = model,
        )

         
        for day in range(2):
            
            for h in range(hours_per_day):

                for step in range(steps_per_hour):

                    hfluxn = ev.physics_data.surface_flux.hfluxn
                    ev.physics_data.surface_flux.hfluxn = hfluxn.at[:].set(1000.0)
                    model.couple_ocn_atm()
                    
                    print("Mean of sst = ", np.nanmean(model.st.sst))

                    recorder.record()


            recorder.output("SOM_STATE_day{day:02d}.nc".format(
                day = day,
            ))

