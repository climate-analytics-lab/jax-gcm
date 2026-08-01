"""Prescribed-forcing builder for the pyses CAM-SE backend.

:func:`build_forcing` interpolates a jcm-canonical monthly climatological
forcing file (regular lon/lat grid, e.g. ``jcm/data/bc/t63/forcing.nc``) onto
the backend's scattered pg2 physics columns and wraps the result as a normal
:class:`jcm.forcing.ForcingData` in the ``(1, ncol)`` layout. Because the
monthly fields become ordinary ``TimeSeries`` leaves (``WRAP_YEAR``
climatology alignment), ``Model``'s existing per-step
``forcing.select(date)`` machinery — month selection, solar-geometry
population, GHG scalars — works unchanged; nothing downstream knows the
horizontal layout is a column list.

Interpolation is host-side numpy bilinear at build time (see
:mod:`jcm.dycore.pyses.interp` — note there about interpolating offline for
higher-resolution production forcing). ``ForcingData.from_file`` is *not*
reused because its interpolation pipeline is spectral-grid specific
(Gaussian-grid upsampling + monthly→daily interpolation); here the monthly
resolution is kept and ``WRAP_YEAR`` indexing selects the current month —
matching the developer prototype's forcing cadence.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from jcm.dycore.pyses.interp import interp_grid_to_points
from jcm.forcing import ForcingData, WRAP_YEAR, make_time_series


# Time-varying monthly fields expected in the jcm-canonical forcing file,
# mapped to their ForcingData field names.
_TIME_FIELDS = {
    "sst": "sea_surface_temperature",
    "icec": "sice_am",
    "stl": "stl_am",
    "soilw_am": "soilw_am",
    "snowc": "snowc_am",
}


def build_forcing(forcing_file: str, dycore, *, validate: bool = True) -> ForcingData:
    """Interpolate a monthly lon/lat forcing climatology onto the physics columns.

    Args:
        forcing_file: jcm-canonical forcing netCDF: monthly ``sst`` /
            ``icec`` / ``stl`` / ``soilw_am`` / ``snowc`` shaped
            ``(lon, lat, time)`` plus static ``alb`` ``(lon, lat)``.
        dycore: A :class:`~jcm.dycore.pyses.dycore.PysesCamSEDycore` (only
            its ``colmap`` column coordinates are read).
        validate: Run the host-side physical-range sanity check jcm applies
            to boundary data (``jcm.forcing._validate_bc_fields``). Disable
            only for synthetic test fixtures.

    Returns:
        :class:`ForcingData` whose spatial leaves are ``(1, ncol)`` (static
        albedo) or ``TimeSeries`` with values ``(12, 1, ncol)`` in
        ``WRAP_YEAR`` (climatology) alignment — the current month is
        selected by fraction-of-year, so the ``time_seconds`` axis (month
        starts, 365-day year) is informational.

    """
    import xarray as xr

    from jcm.forcing import _validate_bc_fields

    ds = xr.open_dataset(forcing_file)
    if validate:
        _validate_bc_fields(ds)

    lon = np.asarray(ds["lon"].values)
    lat = np.asarray(ds["lat"].values)
    cm = dycore.colmap
    ncol = cm.num_cols
    col_lon = np.degrees(cm.longitudes)
    col_lat = np.degrees(cm.latitudes)

    n_time = int(ds.sizes["time"])
    if n_time != 12:
        raise ValueError(
            f"build_forcing expects a 12-month climatology; {forcing_file} "
            f"has {n_time} time slices."
        )
    # Month-start seconds in a 365-day year: informational under WRAP_YEAR
    # (selection is by fraction-of-year), but kept physically labelled.
    month_days = np.array([0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334])
    time_seconds = month_days * 86400.0

    def monthly_to_columns(name):
        arr = np.asarray(ds[name].transpose("lon", "lat", "time").values)
        months = np.stack(
            [interp_grid_to_points(lon, lat, arr[:, :, m], col_lon, col_lat)
             for m in range(n_time)],
            axis=0,
        )                                                  # (12, ncol)
        return months.reshape(n_time, 1, ncol)

    fields = {dest: monthly_to_columns(src) for src, dest in _TIME_FIELDS.items()}
    # Fraction fields pick up interpolation noise at coast/ice edges; clip.
    fields["sice_am"] = np.clip(fields["sice_am"], 0.0, 1.0)
    fields["snowc_am"] = np.clip(fields["snowc_am"], 0.0, 20000.0)

    alb0 = np.clip(
        interp_grid_to_points(
            lon, lat, np.asarray(ds["alb"].transpose("lon", "lat").values),
            col_lon, col_lat,
        ),
        0.0, 1.0,
    ).reshape(1, ncol)

    def ts(values):
        return make_time_series(jnp.asarray(values), time_seconds,
                                align_mode=WRAP_YEAR)

    return ForcingData.zeros(
        nodal_shape=(1, ncol),
        alb0=jnp.asarray(alb0),
        sea_surface_temperature=ts(fields["sea_surface_temperature"]),
        sice_am=ts(fields["sice_am"]),
        stl_am=ts(fields["stl_am"]),
        soilw_am=ts(fields["soilw_am"]),
        snowc_am=ts(fields["snowc_am"]),
    )
