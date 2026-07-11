"""Shared helpers for reading horizontal coordinates off a dycore grid.

Horizontally-aware physics terms (solar zenith in the radiation schemes,
MACv2-SP plume placement) cache one latitude/longitude pair per physics
column at construction time. Historically they assumed a *separable* lat/lon
grid (dinosaur: ``latitudes`` of shape ``(nlat,)``, ``longitudes`` of shape
``(nlon,)``, columns flattened lon-major from ``nodal_shape=(nlon, nlat)``)
and built the per-column pairs with ``jnp.meshgrid``. Scattered-column grids
(the pySES cubed-sphere adapter) cannot be represented by any separable
pair, so they expose the full per-column arrays as ``column_latitudes`` /
``column_longitudes`` instead; this helper prefers those when present and
reproduces the exact legacy meshgrid convention otherwise.
"""

import jax.numpy as jnp


def column_lat_lon(horizontal):
    """Per-column (latitude, longitude) arrays flattened to ``(ncols,)``.

    Units follow the input arrays (dycores conventionally store radians).

    Resolution order:

    1. ``column_latitudes`` / ``column_longitudes`` attributes — full
       per-column coordinates from scattered-column grids (pySES SE).
    2. Same-shaped ``latitudes`` / ``longitudes`` matching ``nodal_shape``
       (e.g. the fake cubed-sphere test grid) — flattened directly.
    3. Separable 1-D ``latitudes`` (nlat,) / ``longitudes`` (nlon,) —
       ``meshgrid(lat, lon)`` + ``reshape(-1)``, bit-identical to the
       convention every shipped scheme used before this helper existed
       (lon-major flattening, matching ``nodal_shape=(nlon, nlat)``).
    """
    col_lat = getattr(horizontal, "column_latitudes", None)
    col_lon = getattr(horizontal, "column_longitudes", None)
    if col_lat is not None and col_lon is not None:
        return (jnp.asarray(col_lat).reshape(-1),
                jnp.asarray(col_lon).reshape(-1))

    lat = jnp.asarray(horizontal.latitudes)
    lon = jnp.asarray(horizontal.longitudes)
    nodal_shape = tuple(getattr(horizontal, "nodal_shape", ()))
    if lat.shape == lon.shape and lat.shape == nodal_shape and lat.ndim > 1:
        return lat.reshape(-1), lon.reshape(-1)

    lat_2d, lon_2d = jnp.meshgrid(lat, lon)
    return lat_2d.reshape(-1), lon_2d.reshape(-1)
