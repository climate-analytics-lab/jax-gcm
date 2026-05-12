"""Default-SPEEDY 90-day climatology generator and stats baseline.

``run_default_speedy_model`` is the runner used by both
``test_speedy_model_default_statistics`` (with ``save_interval=30.`` to
get the per-month averages the test compares) and
``generate()`` (with ``save_interval=None`` for per-timestep output that
becomes the stored mean/std climatology).

``generate()`` writes ``default_statistics.nc`` next to itself. Run it
once whenever the default integration scheme changes in a way that
shifts the 90-day climatology — most recently the operator-split
refactor (#471), which shifted lower-level humidity enough to break
the 3σ bands.

Usage (recommended on GPU, tens of minutes on a single A100)::

    CUDA_VISIBLE_DEVICES=0 python -m jcm.data.test.t30.generate_default_stats

Stats convention (matches ``notebooks/03_generate_speedy_default_stats.ipynb``)::

    mean[var, level] = pred_ds.resample(time='1ME').mean()
                              .isel(time=-1).mean(dim={'lon', 'lat'})
    std [var, level] = pred_ds.mean(dim={'lon', 'lat'}).std(dim='time')

i.e. the *mean* is the last monthly global mean and the *std* is the
std over time of the global-mean time series. The test then checks
``pred_ds.isel(time=-1).mean(dim={'lon','lat'})`` falls inside
``mean ± 3·std`` for every variable in ``default_stat_vars``.
"""

default_stat_vars = ['u_wind', 'v_wind', 'temperature', 'geopotential', 'specific_humidity',
                     'normalized_surface_pressure','humidity.rh','shortwave_rad.ftop','longwave_rad.ftop',
                     'shortwave_rad.cloudstr','shortwave_rad.qcloud','convection.precnv','condensation.precls']

def run_default_speedy_model(save_interval=None):
    """Run the speedy physics at default settings with realistic forcing and terrain
    T31, 40min timestep
    """
    from jcm.model import Model
    from jcm.terrain import TerrainData
    from jcm.physics.speedy.speedy_coords import get_speedy_coords
    from jcm.forcing import ForcingData
    from importlib import resources

    forcing_dir = resources.files('jcm.data.bc.t30.clim')

    # Load the terrain and forcing data

    coords = get_speedy_coords()
    realistic_terrain = TerrainData.from_file(forcing_dir / 'terrain.nc', coords=coords)
    realistic_forcing = ForcingData.from_file(forcing_dir / 'forcing.nc', coords=coords)

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
        coords=coords,
        terrain=realistic_terrain,
        time_step=time_step,
    )

    predictions = model.run(
        save_interval=save_interval,
        total_time=90., # 90 days
        output_averages=output_averages,
        forcing=realistic_forcing,
    )

    return model, predictions


def generate() -> None:
    """Regenerate ``default_statistics.nc`` from a fresh 90-day default-SPEEDY run."""
    from pathlib import Path
    import xarray as xr

    _, predictions = run_default_speedy_model(save_interval=None)
    pred_ds = predictions.to_xarray()

    pred_ds_mean = (
        pred_ds.resample(time="1ME").mean().isel(time=-1).mean(dim={"lon", "lat"})
    )
    pred_ds_std = pred_ds.mean(dim={"lon", "lat"}).std(dim="time")

    stats = xr.Dataset()
    for var in default_stat_vars:
        # Variable names may be nested (e.g. ``shortwave_rad.ftop``);
        # the prepared Dataset exposes them as a flat lookup.
        stats[f"{var}.mean"] = pred_ds_mean[var]
        stats[f"{var}.std"] = pred_ds_std[var]

    out_path = Path(__file__).parent / "default_statistics.nc"
    stats.to_netcdf(out_path)
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    generate()
