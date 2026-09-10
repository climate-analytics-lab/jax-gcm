"""Host-side bilinear sampling of regular lon/lat fields at scattered points.

The pyses CAM-SE backend's physics columns are scattered cubed-sphere (lat,
lon) points, while jcm's bundled boundary-condition data (terrain, monthly
forcing climatology) lives on regular lon/lat grids. This module provides the
one bridge used at *construction time only* (never inside the jitted step):
a periodic-in-longitude, clamped-in-latitude bilinear sampler.

Design notes (per the developer's operational guidance):

* Bilinear interpolation of the T63 boundary data onto the columns is
  acceptable at the resolutions this backend currently targets (ne3 tests,
  ne30 production against 1.875° data). Higher-resolution runs against
  coarser forcing — or conservative remapping requirements — should be
  handled by interpolating the boundary data **offline** onto the column set
  and feeding the result in directly.
* Everything here is plain numpy (no scipy dependency, no jax): it runs once
  on the host while the dycore is being built.
"""

from __future__ import annotations

import numpy as np


def interp_grid_to_points(lon_deg, lat_deg, field_lonlat, pt_lon_deg, pt_lat_deg):
    """Bilinearly sample a regular lon/lat field at arbitrary ``(lon, lat)`` points.

    Args:
        lon_deg: 1-D longitudes of the source grid, ascending in ``[0, 360)``.
        lat_deg: 1-D latitudes of the source grid, ascending.
        field_lonlat: Source field with shape ``(nlon, nlat)`` (jcm-canonical
            boundary-condition layout).
        pt_lon_deg: Target-point longitudes (degrees, any real values —
            wrapped into ``[0, 360)``).
        pt_lat_deg: Target-point latitudes (degrees). Values beyond the
            source grid's latitude range are clamped so near-pole columns
            take the edge-row value rather than a wild extrapolation.

    Returns:
        A numpy array with the shape of ``pt_lon_deg`` containing the
        bilinear sample. Exact on source grid nodes; periodic across the
        0/360 longitude seam (one wrapped cell each side).

    """
    lon = np.asarray(lon_deg, dtype=np.float64)
    lat = np.asarray(lat_deg, dtype=np.float64)
    field = np.asarray(field_lonlat, dtype=np.float64)
    if field.shape != (lon.size, lat.size):
        raise ValueError(
            f"field shape {field.shape} does not match (nlon, nlat) = "
            f"({lon.size}, {lat.size})"
        )

    # Periodic longitude padding: one wrapped cell on each end so points in
    # the [lon[-1], lon[0]+360) seam interval interpolate across the wrap.
    lon_ext = np.concatenate([[lon[-1] - 360.0], lon, [lon[0] + 360.0]])
    field_ext = np.concatenate([field[-1:, :], field, field[:1, :]], axis=0)

    pt_lon = np.mod(np.asarray(pt_lon_deg, dtype=np.float64), 360.0)
    pt_lat = np.clip(np.asarray(pt_lat_deg, dtype=np.float64), lat[0], lat[-1])
    out_shape = pt_lon.shape
    pt_lon = pt_lon.ravel()
    pt_lat = np.broadcast_to(pt_lat, out_shape).ravel()

    # Bracketing indices along each axis. ``searchsorted - 1`` gives the cell
    # whose lower edge is <= the point; clip keeps the last point in-range.
    i0 = np.clip(np.searchsorted(lon_ext, pt_lon, side="right") - 1,
                 0, lon_ext.size - 2)
    j0 = np.clip(np.searchsorted(lat, pt_lat, side="right") - 1,
                 0, lat.size - 2)
    i1, j1 = i0 + 1, j0 + 1

    wx = (pt_lon - lon_ext[i0]) / (lon_ext[i1] - lon_ext[i0])
    wy = (pt_lat - lat[j0]) / (lat[j1] - lat[j0])
    wx = np.clip(wx, 0.0, 1.0)
    wy = np.clip(wy, 0.0, 1.0)

    val = ((1 - wx) * (1 - wy) * field_ext[i0, j0]
           + wx * (1 - wy) * field_ext[i1, j0]
           + (1 - wx) * wy * field_ext[i0, j1]
           + wx * wy * field_ext[i1, j1])
    return val.reshape(out_shape)
