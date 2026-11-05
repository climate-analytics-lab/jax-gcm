"""SLIM's prescribed land-surface properties on the model grid.

SLIM reads its surface dataset (``mml_surdat``) once a month: record ``m`` of
every field from the first of month ``m`` (``mml_main.F90::nc_import``, called
at the first step of a run and at day 1, 00 UTC). :func:`read_slim_properties`
reads a file already regridded to the model grid (``slim_t63.nc``; how it is
made is in its ``provenance`` attribute and in
``docs/source/design/land_skin_energy_balance.md``) into the mapping
``ForcingData.slim_properties``: the monthly fields as ``WRAP_YEAR``
``TimeSeries`` (stepped by month, never interpolated, as SLIM does) and the
static ``glc_mask``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

#: Monthly fields SLIM reads (``nc_import``), all required. ``emissivity`` and
#: the dust fluxes are SLIM fields jcm does not read (the land emissivity is
#: the host's; jcm's dust is the JAM scheme's).
MONTHLY_FIELDS = (
    "alb_gvd", "alb_svd", "alb_gnd", "alb_snd",
    "alb_gvf", "alb_svf", "alb_gnf", "alb_snf",
    "bucketdepth", "snowmask", "evap_res", "roughness",
    "soil_tk_1d", "soil_cv_1d", "glc_tk_1d", "glc_cv_1d",
)
#: Static fields.
STATIC_FIELDS = ("glc_mask",)

_MONTHS = 12


def read_slim_properties(ds, lat_deg=None, lon_deg=None, source="slim_file"):
    """Read SLIM's surface properties into ``ForcingData.slim_properties``.

    Args:
        ds: an :class:`xarray.Dataset` on the model grid with the
            :data:`MONTHLY_FIELDS` dimensioned ``(time, lat, lon)`` (twelve
            month-start records) and :data:`STATIC_FIELDS` ``(lat, lon)``.
        lat_deg: the model's latitudes [deg], checked against the file's.
        lon_deg: the model's longitudes [deg], checked against the file's.
        source: name used in error messages.

    Returns:
        ``{name: TimeSeries | array}``, each field shaped ``(lon, lat)`` after
        :meth:`ForcingData.select`.

    """
    from jcm.forcing import _orient_to_model_grid, _times_and_mode, make_time_series

    missing = [v for v in MONTHLY_FIELDS + STATIC_FIELDS if v not in ds.data_vars]
    if missing:
        raise ValueError(f"{source}: missing SLIM field(s) {missing}.")
    if ds.sizes.get("time") != _MONTHS:
        raise ValueError(
            f"{source}: SLIM's surface dataset has {_MONTHS} monthly records "
            f"(nc_import reads record m in month m), found {ds.sizes.get('time')}.")
    out = {}
    for name in MONTHLY_FIELDS:
        arr = _orient_to_model_grid(ds[name], lat_deg, lon_deg, name=name)
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{source}: {name} has non-finite values.")
        out[name] = make_time_series(arr, *_times_and_mode(ds, "wrap_year", name))
    for name in STATIC_FIELDS:
        arr = _orient_to_model_grid(ds[name], lat_deg, lon_deg, name=name)
        out[name] = jnp.asarray((np.asarray(arr) > 0.5).astype(np.float32))
    return out
