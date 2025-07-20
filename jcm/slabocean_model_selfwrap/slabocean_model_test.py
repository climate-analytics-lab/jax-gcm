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
       
        # ==== Making / Interpolating Boundary Condition from a reference file ====
 
        boundaries_dir = Path(__file__).resolve().parent / ".." / 'data/bc/t30/clim'

        # Generate daily values from monthly climatology
        daily_boundary_condition_file = boundaries_dir / 'boundaries_daily.nc'
        if not daily_boundary_condition_file.exists():
            import subprocess, sys
            subprocess.run([sys.executable, str(boundaries_dir / 'interpolate.py')], check=True)
        
        # ==== creating model ==== 
       
        coords = som.misc.get_coords()
        
        hori_shape  = coords.horizontal.nodal_shape
        vert_layers = coords.vertical.layers
        
        
        boundaries = initialize_boundaries(
            daily_boundary_condition_file,
            coords.horizontal,
        )
        boundaries.ocn_coupling_flag = True
       
        # Default parameter. See slabocean_model_parameters.py for default values. 
        params = Parameters.default()
        
        physics_data = PhysicsData.zeros(hori_shape, vert_layers)
 

        simulation_days = 20 
        hours_per_day   = 24
        steps_per_hour  = 6
        time_step = 600.0

        ev = som.Env(
            time_step = time_step,
            save_interval = 86400.0,                       # 1 day   : No effect currently
            total_time = 86400.0 * simulation_days,        # 10 days : No effect currently
            start_date = pd.Timestamp("2000-01-01"),       # No effect
            horizontal_resolution = 31,
            coords = coords,
            boundaries = boundaries,
            parameters = params,
            physics_data = physics_data,
        )
        
        model = som.Model(ev=ev)
        self.assertTrue(model is not None)
       
        # ==== stepping the model ==== 

        # Uncomment this line to set initial SST ad-hoc
        #model.st.sst = model.st.sst.at[:].set(100.0)

        recorder = som.Recorder(
            count_per_avg = steps_per_hour,
            model = model,
            output_style = "netcdf",
        )

        output_dir = Path("output")
        output_dir.mkdir(parents=True, exist_ok=True) 
        for day in range(simulation_days):
            for h in range(hours_per_day):
                for step in range(steps_per_hour):

                    # Here is an ad-hoc way to set heat flux. Positive downward.
                    hfluxn = ev.physics_data.surface_flux.hfluxn
                    #ev.physics_data.surface_flux.hfluxn = hfluxn.at[:].set(1000.0)
                    model.couple_ocn_atm()
                   
                    # Reocrd the state 
                    recorder.record()


            recorder.output( output_dir / ("SOM_STATE_day{day:02d}.nc".format(
                day = day,
            )))

