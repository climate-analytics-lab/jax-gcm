from jcm.model import Model
from jcm.boundaries import boundaries_from_file
import jax.numpy as jnp
import xarray as xr
from pathlib import Path

default_stat_vars = ['u_wind', 'v_wind', 'temperature', 'geopotential', 'specific_humidity',
                     'normalized_surface_pressure','humidity.rh','shortwave_rad.ftop','longwave_rad.ftop',
                     'shortwave_rad.cloudstr','shortwave_rad.qcloud','convection.precnv','condensation.precls']

def run_default_speedy_model(save_interval=None):
    '''
        Run the speedy physics at default settings with realistic forcing and terrain
        T31, 40min timestep
    '''
    boundaries_dir = Path(__file__).resolve().parent / '../../bc/t30/clim'
    interp_dir = Path(__file__).resolve().parent / '../../bc'
    
    if not (boundaries_dir / 'boundaries_daily_t31.nc').exists():
        import subprocess
        import sys
        subprocess.run([sys.executable, str(interp_dir / 'interpolate.py'), '31'], check=True)

    realistic_boundaries = boundaries_from_file(
        boundaries_dir / 'boundaries_daily_t31.nc',
    )
    realistic_orography = jnp.asarray(xr.open_dataarray(boundaries_dir / 'orography_t31.nc'))

    # in the default scenario output every timestep and don't average
    # in the test scenario, output as designated and average
    time_step = 40.0  # time step in minutes
    output_averages = False
    if save_interval is None:
        save_interval = time_step/1440.
    else:
        save_interval = save_interval
        output_averages = True

    model = Model(
        orography=realistic_orography,
        time_step=time_step,
    )

    predictions = model.run(
        save_interval=save_interval,
        total_time=90., # 90 days 
        output_averages=output_averages,
        boundaries=realistic_boundaries,
    )

    return model, predictions