"""Regenerate ``default_statistics.nc`` under the current default integration scheme.

The frozen statistics are the climatology baseline used by
``test_speedy_model_default_statistics`` (see ``jcm/model_test.py``).
They must be regenerated when the default integration scheme changes
in a way that meaningfully shifts the 90-day climatology — most
recently when issue #471 switched physics from inside-RK to
operator-split, perturbing lower-level humidity by enough to break
the original 3σ bands.

Usage (recommended on GPU, ~tens of minutes on a single A100):

    CUDA_VISIBLE_DEVICES=0 python -m jcm.data.test.t30.regenerate_default_stats

The script writes ``default_statistics.nc`` next to itself. Commit
the result alongside the change that motivated the regen.

Statistics convention (matches ``notebooks/03_generate_speedy_default_stats.ipynb``):

    mean[var, level] = ``pred_ds.resample(time='1ME').mean().isel(time=-1).mean(dim={'lon','lat'})``
    std [var, level] = ``pred_ds.mean(dim={'lon','lat'}).std(dim='time')``

i.e. the *mean* is the last monthly global mean and the *std* is the
std over time of the global mean time series. The test then checks
that ``pred_ds.isel(time=-1).mean(dim={'lon','lat'})`` falls inside
``mean ± 3 * std`` for every variable in
``default_stat_vars``.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

import xarray as xr

from jcm.data.test.t30.generate_default_stats import default_stat_vars


def generate() -> None:
    """Run the default 90-day speedy integration and write the climatology file."""
    from jcm.model import Model
    from jcm.physics.speedy.speedy_coords import get_speedy_coords
    from jcm.terrain import TerrainData
    from jcm.forcing import ForcingData

    forcing_dir = resources.files("jcm.data.bc.t30.clim")
    coords = get_speedy_coords()
    terrain = TerrainData.from_file(forcing_dir / "terrain.nc", coords=coords)
    forcing = ForcingData.from_file(forcing_dir / "forcing.nc", coords=coords)

    # Match the test's run config: default 40-minute timestep, 90 days,
    # output_averages=False, save every timestep so the daily-ish time
    # series can be both resampled to monthly (for the mean) and
    # globally averaged then std'd (for the std).
    time_step = 40.0
    save_interval = time_step / 1440.0  # one save per timestep
    model = Model(coords=coords, terrain=terrain, time_step=time_step)
    predictions = model.run(
        save_interval=save_interval,
        total_time=90.0,
        output_averages=False,
        forcing=forcing,
    )

    pred_ds = predictions.to_xarray()

    pred_ds_mean = (
        pred_ds.resample(time="1ME").mean().isel(time=-1).mean(dim={"lon", "lat"})
    )
    pred_ds_std = pred_ds.mean(dim={"lon", "lat"}).std(dim="time")

    stats = xr.Dataset()
    for var in default_stat_vars:
        # Variable names may be nested (e.g. ``shortwave_rad.ftop``); xarray
        # accepts the dotted name as a flat lookup on the prepared Dataset.
        stats[f"{var}.mean"] = pred_ds_mean[var]
        stats[f"{var}.std"] = pred_ds_std[var]

    out_path = Path(__file__).parent / "default_statistics.nc"
    stats.to_netcdf(out_path)
    print(f"Wrote {out_path} ({out_path.stat().st_size} bytes)")


if __name__ == "__main__":
    generate()
