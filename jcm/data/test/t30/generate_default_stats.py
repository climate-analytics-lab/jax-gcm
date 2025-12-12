from pathlib import Path

default_stat_vars = ['u_wind', 'v_wind', 'temperature', 'geopotential', 'specific_humidity',
                     'normalized_surface_pressure','humidity.rh','shortwave_rad.ftop','longwave_rad.ftop',
                     'shortwave_rad.cloudstr','shortwave_rad.qcloud','convection.precnv','condensation.precls']

def run_default_speedy_model(save_interval=None):
    """Run the speedy physics at default settings with realistic forcing and terrain
    T31, 40min timestep
    """
    from jcm.model import Model
    from jcm.geometry import Geometry
    from jcm.forcing import ForcingData

    forcing_dir = Path(__file__).resolve().parent / '../../bc/t30/clim'

    # Load the terrain and forcing data
    
    realistic_geometry = Geometry.from_file(forcing_dir / 'terrain.nc', target_resolution=31)
    realistic_forcing = ForcingData.from_file(forcing_dir / 'forcing.nc', target_resolution=31)

    # in the default scenario output every timestep and don't average
    # in the test scenario, output as designated and average
    time_step = 40.0  # default time step in minutes
    output_averages = False
    if save_interval is None:
        save_interval = time_step/1440.
    else:
        save_interval = save_interval
        output_averages = True

    model = Model(
        geometry=realistic_geometry,
        time_step=time_step,
    )

    predictions = model.run(
        save_interval=save_interval,
        total_time=90., # 90 days 
        output_averages=output_averages,
        forcing=realistic_forcing,
    )

    return model, predictions