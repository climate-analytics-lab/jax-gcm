"""Default-ECHAM statistics harness, mirroring ``t30/generate_default_stats.py``.

Runs ``echam_physics()`` on a T31L8 aquaplanet and exposes the variable
list and run helper that the slow regression test in ``model_test.py``
compares against ``default_statistics.nc``.

The variable list intentionally mixes a few prognostic-state fields
(global means stay close to climate values) with a handful of
diagnostic scheme outputs (cloud fraction, top-of-atmosphere fluxes,
precipitation) so a regression in any of the scheme migrations shows up
on at least one of the asserted bands.
"""

from __future__ import annotations


default_echam_stat_vars = [
    # Prognostic state.
    "u_wind",
    "v_wind",
    "temperature",
    "specific_humidity",
    "normalized_surface_pressure",
    # Moist-air diagnostics produced by MoistAirColumnState.
    "pressure_full",
    "air_density",
    "layer_thickness",
    "relative_humidity",
    # Scheme outputs.
    "radiation.toa_lw_up",
    "radiation.surface_sw_down",
    "clouds.cloud_fraction",
    "clouds.precip_rain",
    "convection.precip_conv",
    # NB: ``chemistry.ozone_vmr`` (and the other chemistry VMRs) is
    # reset to a constant value by EchamBoundaryConditions every step
    # — its std is 0 so it can't carry a meaningful climatology band.
    # Listed here as a reminder that the chemistry path is currently
    # effectively a no-op (see IMPLEMENTATION_ROADMAP.md).
]


def run_default_echam_model(save_interval=None, total_time=5.0):
    """Run the default ECHAM aquaplanet at T31L8.

    Mirrors ``run_default_speedy_model`` in ``t30/generate_default_stats.py``:
    ``save_interval=None`` saves every timestep (raw snapshots) so the
    stat-generation notebook can compute per-step variability;
    ``save_interval=X`` saves every ``X`` days with time-averaging so
    the regression test gets a single ``X``-day mean to compare against
    the saved climatology band.

    Defaults to a 5-day run. Longer aquaplanet integrations at this
    resolution + default physics still blow up to NaN (the
    ``debug/echam-2m-micro-stability`` work the user is in the middle
    of); 5 days gives stable diagnostics with a meaningful
    daily-snapshot standard deviation across time.

    Args:
        save_interval: Save interval in days. ``None`` → save every
            timestep, no averaging.
        total_time: Total simulation length in days.

    Returns:
        Tuple ``(model, predictions)``.

    """
    import numpy as np

    from jcm.forcing import ForcingData
    from jcm.model import Model
    from jcm.physics.echam.echam_terms import echam_physics
    from jcm.terrain import TerrainData
    from jcm.utils import get_coords

    sigma_boundaries = np.linspace(0, 1, 9)  # 8 levels
    coords = get_coords(sigma_boundaries, nodal_shape=(64, 32))
    terrain = TerrainData.aquaplanet(coords)
    forcing = ForcingData.zeros((64, 32))

    time_step = 30.0  # minutes (Model default)
    output_averages = False
    if save_interval is None:
        save_interval = time_step / 1440.0  # one timestep, in days
    else:
        output_averages = True

    model = Model(
        coords=coords,
        terrain=terrain,
        physics=echam_physics(),
        time_step=time_step,
    )

    predictions = model.run(
        save_interval=save_interval,
        total_time=total_time,
        output_averages=output_averages,
        forcing=forcing,
    )
    return model, predictions
